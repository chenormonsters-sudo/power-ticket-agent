"""
监测 Agent 数据模型：测点、告警、缺陷事件。
监测层为纯规则计算，不调用 LLM。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Severity(str, Enum):
    """告警/事件严重度（用于优先级调度）。"""
    GENERAL = "一般"      # 低风险，自动放行
    MAJOR = "较大"        # 中风险，关键节点确认
    SERIOUS = "重大"      # 高风险，每步人工确认


class AlarmType(str, Enum):
    """告警类型。"""
    THRESHOLD = "越限"      # 阈值越限：触发完整诊断
    TREND = "趋势"          # 趋势预警：仅运行提示，不触发诊断
    RATE = "突变"           # 变化率突变：触发完整诊断
    CORRELATION = "组合"    # 跨测点组合异常：触发完整诊断


@dataclass
class Measurement:
    """单条测点采样值。"""
    point_id: str          # 测点编号，如 "M01-BRG-TEMP"
    device: str            # 所属设备，如 "磨煤机2#"
    team: str              # 所属班组：锅炉/汽机/电气/热控/输煤
    value: float           # 采样值
    ts: datetime           # 采样时间


@dataclass
class Alarm:
    """规则引擎产出的原始告警。"""
    point_id: str
    device: str
    team: str
    value: float
    alarm_type: AlarmType
    severity: Severity
    ts: datetime
    detail: str = ""       # 说明（如超限幅度/趋势斜率）
    threshold: float | None = None


@dataclass
class DefectEvent:
    """归并后的缺陷事件：诊断流程的入口。"""
    event_id: str
    device: str                    # 主设备
    teams: list[str]               # 涉及班组（用于动态激活专家）
    params: list[dict]             # 异常参数列表 [{point_id, value, detail}]
    severity: Severity
    timeline: list[dict]           # 事件时间线 [{ts, point_id, event}]
    triggered: bool = False        # 是否触发完整诊断（趋势预警不触发）
    created_at: datetime = field(default_factory=datetime.now)
