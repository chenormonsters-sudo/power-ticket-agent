"""
监测 Agent 单元测试：规则引擎 / 去抖 / 归并 / 事件输出。
运行：python -m pytest backend/agents/monitor/test_monitor.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from datetime import datetime, timedelta

from backend.agents.monitor.schemas import Alarm, AlarmType, Measurement, Severity
from backend.agents.monitor.rules import CorrelationRule, MonitorRules, PointRule
from backend.agents.monitor.alarm_processor import (
    AlarmDebouncer, AlarmMerger, DEFAULT_RELATIONS, DeviceCorrelationMatrix,
)
from backend.agents.monitor.agent import MonitorAgent, build_default_specs
from backend.agents.monitor.simulator import DcsSimulator, FaultScenario


def _m(point_id: str, value: float, device: str = "磨煤机2#", team: str = "锅炉", ts: datetime | None = None) -> Measurement:
    return Measurement(point_id=point_id, device=device, team=team, value=value,
                       ts=ts or datetime(2026, 8, 10, 0, 0, 0))


def test_threshold_alarm_triggers():
    """越限告警触发且定级正确。"""
    rule = PointRule("P1", "磨煤机2#", "锅炉", high_limit=85.0)
    alarms = rule.feed(_m("P1", 86.0))
    assert len(alarms) == 1
    assert alarms[0].alarm_type == AlarmType.THRESHOLD
    assert alarms[0].severity == Severity.MAJOR


def test_trend_warning_is_hint_only():
    """趋势预警类型为 TREND（只提示，不触发诊断）。"""
    rule = PointRule("P1", "磨煤机2#", "锅炉", trend_window=5, trend_slope=2.0, window=10)
    alarms = []
    for i in range(10):
        alarms += rule.feed(_m("P1", 70.0 + i * 2.0))
    trends = [a for a in alarms if a.alarm_type == AlarmType.TREND]
    assert trends, "应产生趋势预警"
    # 通过 Agent 链路验证 TREND 不置 triggered
    agent = MonitorAgent()
    ev = None
    for i in range(10):
        for e in agent.process_measurement(_m("P1", 70.0 + i * 2.0, ts=datetime(2026, 8, 10, 0, 0, i))):
            ev = e
    assert ev is None or ev.triggered is False or len(ev.timeline) == 0


def test_debounce_suppresses_duplicates():
    """去抖：同设备同测点同类型在窗口内只放行一次。"""
    d = AlarmDebouncer(window_seconds=300)
    a1 = Alarm("P1", "磨煤机2#", "锅炉", 90.0, AlarmType.THRESHOLD, Severity.MAJOR,
               datetime(2026, 8, 10, 0, 0, 0))
    a2 = Alarm("P1", "磨煤机2#", "锅炉", 91.0, AlarmType.THRESHOLD, Severity.MAJOR,
               datetime(2026, 8, 10, 0, 1, 0))
    assert d.accept(a1) is True
    assert d.accept(a2) is False  # 60 秒内重复 → 抑制


def test_merge_correlation_event():
    """双条件归并：温度+振动（关联矩阵）在时间窗内归并为同一事件，含时间线。"""
    merger = AlarmMerger(DeviceCorrelationMatrix(DEFAULT_RELATIONS), window_seconds=900)
    t0 = datetime(2026, 8, 10, 0, 0, 0)
    a1 = Alarm("M02-BRG-TEMP", "磨煤机2#", "锅炉", 86.0, AlarmType.THRESHOLD, Severity.MAJOR, t0)
    a2 = Alarm("M02-BRG-VIB", "磨煤机2#", "锅炉", 9.0, AlarmType.RATE, Severity.GENERAL, t0 + timedelta(seconds=10))
    evs1 = merger.process(a1)
    evs2 = merger.process(a2)
    assert len(evs1) == 1
    assert len(evs2) == 1
    assert evs2[0].event_id == evs1[0].event_id  # 归并到同一事件
    assert len(evs2[0].timeline) == 2
    assert evs2[0].triggered is True


def test_independent_devices_not_merged():
    """时间紧邻但无设备关联 → 各自独立事件（防误并）。"""
    merger = AlarmMerger(DeviceCorrelationMatrix(DEFAULT_RELATIONS), window_seconds=900)
    t0 = datetime(2026, 8, 10, 0, 0, 0)
    a1 = Alarm("P1", "磨煤机2#", "锅炉", 86.0, AlarmType.THRESHOLD, Severity.MAJOR, t0)
    a2 = Alarm("P2", "汽轮机1#", "汽机", 86.0, AlarmType.THRESHOLD, Severity.MAJOR, t0 + timedelta(seconds=5))
    e1 = merger.process(a1)[0]
    e2 = merger.process(a2)[0]
    assert e1.event_id != e2.event_id


def test_simulator_fault_event_e2e():
    """端到端：注入磨煤机故障 → 产出触发诊断的缺陷事件。"""
    agent = MonitorAgent()
    scenarios = [
        FaultScenario("M02-BRG-TEMP", 120.0, "ramp", rate=0.30),
        FaultScenario("M02-BRG-VIB", 200.0, "spike", target_delta=6.0),
    ]
    events = agent.run_simulation(scenarios=scenarios, steps=80, time_scale=60.0)
    assert len(events) >= 1
    ev = events[0]
    assert ev.device == "磨煤机2#"
    assert ev.triggered is True
    assert ev.severity == Severity.SERIOUS
    assert len(ev.timeline) >= 4  # 越限+趋势+突变+组合


def test_specs_match_rules():
    """默认测点配置与规则构建不报错。"""
    specs = build_default_specs()
    from backend.agents.monitor.agent import build_default_rules
    rules = build_default_rules(specs)
    assert len(specs) == 5
    assert len(rules.point_rules) == 5
