"""
MonitorAgent：监测 Agent 组装。

数据流：DCS 模拟器 → 规则引擎（阈值/趋势/变化率/关联）→ 去抖 → 归并 → DefectEvent

设计要点：
- 24h 常驻，纯规则计算，零 LLM 成本
- 趋势预警仅作为运行提示（triggered=False），越限/突变/组合异常触发完整诊断
- 多实例并发入口：process_measurement 无状态依赖（除去抖/归并缓冲），可多线程消费
"""
from __future__ import annotations

from datetime import datetime

from backend.agents.monitor.schemas import DefectEvent, Measurement
from backend.agents.monitor.rules import CorrelationRule, MonitorRules, PointRule
from backend.agents.monitor.alarm_processor import (
    AlarmDebouncer, AlarmMerger, DEFAULT_RELATIONS, DeviceCorrelationMatrix,
)
from backend.agents.monitor.simulator import DcsSimulator, FaultScenario, PointSpec
from backend.base.logger import get_logger

logger = get_logger(__name__)


def build_default_specs() -> list[PointSpec]:
    """默认测点配置（演示用，覆盖典型关键辅机）。"""
    return [
        PointSpec("M02-BRG-TEMP", "磨煤机2#", "锅炉", baseline=70.0, noise=1.0, sample_interval=10.0),
        PointSpec("M02-BRG-VIB", "磨煤机2#", "锅炉", baseline=3.5, noise=0.3, sample_interval=10.0),
        PointSpec("PAF-A-VIB", "一次风机A", "锅炉", baseline=4.0, noise=0.4, sample_interval=10.0),
        PointSpec("T01-W1-TEMP", "汽轮机1#", "汽机", baseline=62.0, noise=0.8, sample_interval=10.0),
        PointSpec("T01-LUB-P", "润滑油泵A", "汽机", baseline=0.25, noise=0.01, sample_interval=10.0),
    ]


def build_default_rules(specs: list[PointSpec]) -> MonitorRules:
    """由测点配置构建规则引擎（阈值 = 基线 + 15 的典型高限）。"""
    point_rules = []
    for s in specs:
        point_rules.append(PointRule(
            point_id=s.point_id, device=s.device, team=s.team,
            high_limit=round(s.baseline + 15.0, 1),
            rate_limit=max(5.0, s.baseline * 0.15),
            trend_window=10, trend_slope=2.0, window=20,
        ))

    # 跨测点关联：磨煤机轴承温度+振动组合
    correlation_rules = [
        CorrelationRule(
            device="磨煤机2#", team="锅炉",
            related_points=["M02-BRG-TEMP", "M02-BRG-VIB"],
            window_seconds=300, min_hits=2,
        ),
    ]
    return MonitorRules(point_rules, correlation_rules)


class MonitorAgent:
    """监测 Agent：消费测点流，产出缺陷事件。"""

    def __init__(
        self,
        rules: MonitorRules | None = None,
        debouncer: AlarmDebouncer | None = None,
        merger: AlarmMerger | None = None,
    ):
        self.rules = rules or build_default_rules(build_default_specs())
        self.debouncer = debouncer or AlarmDebouncer(window_seconds=300)
        self.merger = merger or AlarmMerger(
            DeviceCorrelationMatrix(DEFAULT_RELATIONS), window_seconds=900,
        )

    def process_measurement(self, m: Measurement) -> list[DefectEvent]:
        """
        处理单条采样：规则检测 → 去抖 → 归并。
        返回：本次采样影响的缺陷事件（新建/更新）。
        注意：同一设备多个测点建议逐条喂入（时间戳一致）。
        """
        affected: list[DefectEvent] = []
        for alarm in self.rules.feed(m):
            if not self.debouncer.accept(alarm):
                logger.info("monitor.debounced", point=alarm.point_id, type=alarm.alarm_type.value)
                continue
            logger.info(
                "monitor.alarm",
                point=alarm.point_id, type=alarm.alarm_type.value,
                severity=alarm.severity.value, detail=alarm.detail,
            )
            events = self.merger.process(alarm)
            affected.extend(events)
        return affected

    def run_simulation(
        self,
        specs: list[PointSpec] | None = None,
        scenarios: list[FaultScenario] | None = None,
        steps: int = 60,
        time_scale: float = 60.0,
    ) -> list[DefectEvent]:
        """
        跑一段模拟数据流（快进模式），返回过程中产生的缺陷事件（按优先级排序）。
        time_scale: 快进倍数（60 表示 1 秒模拟 60 秒采样间隔流逝）。
        """
        specs = specs or build_default_specs()
        sim = DcsSimulator(specs, scenarios or [], time_scale=time_scale)
        all_events: dict[str, DefectEvent] = {}

        for _ in range(steps):
            for m in sim.step():
                for ev in self.process_measurement(m):
                    all_events[ev.event_id] = ev

        # 关闭过期事件
        for ev in self.merger.close_expired():
            all_events[ev.event_id] = ev

        order = {"重大": 3, "较大": 2, "一般": 1}
        return sorted(all_events.values(), key=lambda e: (-order[e.severity.value], e.created_at))


if __name__ == "__main__":
    """演示：注入磨煤机轴承温度爬升故障，观察告警→归并→事件输出。"""
    import asyncio
    from backend.base.logger import configure_logging
    configure_logging()

    agent = MonitorAgent()
    scenarios = [
        FaultScenario(
            point_id="M02-BRG-TEMP", start_offset=120.0,
            mode="ramp", rate=0.30,      # 每采样周期 +0.3，约 50 个周期后越限（70+15=85）
        ),
        FaultScenario(
            point_id="M02-BRG-VIB", start_offset=200.0,
            mode="spike", target_delta=6.0,   # 振动突变，与温度构成组合异常
        ),
    ]

    events = agent.run_simulation(scenarios=scenarios, steps=80, time_scale=60.0)
    print(f"\n{'='*60}\n共产生 {len(events)} 个缺陷事件：\n")
    for ev in events:
        print(f"事件 {ev.event_id} | 设备 {ev.device} | 班组 {ev.teams} | 严重度 {ev.severity.value} | 触发诊断: {ev.triggered}")
        print(f"  异常参数: {ev.params}")
        print(f"  时间线:")
        for t in ev.timeline:
            print(f"    {t['ts'][11:19]} {t['point_id']}: {t['event']}")
        print()
