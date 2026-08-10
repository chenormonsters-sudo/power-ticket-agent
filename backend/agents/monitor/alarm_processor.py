"""
告警预处理：去抖（防抖）+ 双条件归并（时间窗口 + 设备关联矩阵）+ 事件时间线 + 优先级。

设计要点：
- 去抖：同一设备同一测点时间窗内重复告警合并，防刷屏空转
- 双条件归并：时间上紧邻（窗口内）+ 工艺/连锁关系关联（设备关联矩阵），双条件缺一不可
- 输出 DefectEvent（含事件时间线），作为诊断流程入口
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from backend.agents.monitor.schemas import Alarm, DefectEvent, Severity


class AlarmDebouncer:
    """告警去抖：时间窗内同设备同测点同类型告警合并。"""

    def __init__(self, window_seconds: int = 300):
        self.window_seconds = window_seconds
        self._last: dict[tuple[str, str, str], datetime] = {}  # (device, point, type) -> ts

    def accept(self, alarm: Alarm) -> bool:
        """返回 True 表示该告警通过去抖（应处理），False 表示重复应抑制。"""
        key = (alarm.device, alarm.point_id, alarm.alarm_type.value)
        now = alarm.ts
        last = self._last.get(key)
        if last is not None and (now - last).total_seconds() < self.window_seconds:
            return False
        self._last[key] = now
        return True


class DeviceCorrelationMatrix:
    """设备关联矩阵：工艺上下游 / 连锁逻辑关系。

    生产环境由工艺工程师配置；演示环境预置常见连锁关系。
    """

    def __init__(self, relations: dict[str, set[str]] | None = None):
        # relations: 设备 -> 关联设备集合（双向）
        self._relations: dict[str, set[str]] = relations or {}

    def related(self, device_a: str, device_b: str) -> bool:
        return device_b in self._relations.get(device_a, set()) or \
               device_a in self._relations.get(device_b, set())


# 演示用预置设备关联矩阵（磨煤机/一次风机/密封风机/给煤机为典型连锁链路）
DEFAULT_RELATIONS: dict[str, set[str]] = {
    "磨煤机2#": {"一次风机A", "密封风机A", "给煤机2#"},
    "一次风机A": {"磨煤机2#", "密封风机A", "锅炉本体"},
    "密封风机A": {"磨煤机2#", "一次风机A"},
    "给煤机2#": {"磨煤机2#"},
    "汽轮机1#": {"凝结水泵A", "润滑油泵A", "轴封系统"},
    "凝结水泵A": {"汽轮机1#"},
    "润滑油泵A": {"汽轮机1#"},
}


class AlarmMerger:
    """告警归并器：把去抖后的告警按 时间窗口+设备关联 双条件归并为缺陷事件。"""

    def __init__(
        self,
        matrix: DeviceCorrelationMatrix,
        window_seconds: int = 900,          # 归并时间窗
        severity_order: dict[Severity, int] | None = None,
    ):
        self.matrix = matrix
        self.window_seconds = window_seconds
        self.severity_order = severity_order or {
            Severity.SERIOUS: 3, Severity.MAJOR: 2, Severity.GENERAL: 1,
        }
        self._open_events: list[DefectEvent] = []

    def process(self, alarm: Alarm) -> list[DefectEvent]:
        """
        处理一条告警：
        - 若与某个未关闭事件满足 时间窗口 + 设备关联 双条件 → 归并进该事件
        - 否则新开一个缺陷事件
        返回：受影响的事件列表（新建或更新）
        """
        affected: list[DefectEvent] = []
        now = alarm.ts
        merged = False

        for ev in self._open_events:
            # 时间窗口条件
            if (now - ev.created_at).total_seconds() > self.window_seconds:
                continue
            # 设备关联条件（同设备或关联矩阵命中）
            if ev.device == alarm.device or self.matrix.related(ev.device, alarm.device):
                self._merge(ev, alarm)
                affected.append(ev)
                merged = True
                break

        if not merged:
            ev = DefectEvent(
                event_id=f"EV-{now.strftime('%Y%m%d-%H%M%S')}-{len(self._open_events) + 1}",
                device=alarm.device,
                teams=[alarm.team] if alarm.team else [],
                params=[{"point_id": alarm.point_id, "value": alarm.value, "detail": alarm.detail}],
                severity=alarm.severity,
                timeline=[{"ts": alarm.ts.isoformat(), "point_id": alarm.point_id, "event": f"{alarm.alarm_type.value}：{alarm.detail}"}],
                triggered=alarm.alarm_type.value != "趋势",
                created_at=alarm.ts,   # 时间轴与告警一致（模拟/生产语义统一）
            )
            self._open_events.append(ev)
            affected.append(ev)

        # 清理过期事件（超过窗口）
        self._open_events = [
            ev for ev in self._open_events
            if (now - ev.created_at).total_seconds() <= self.window_seconds
        ]
        return affected

    def _merge(self, ev: DefectEvent, alarm: Alarm) -> None:
        """把告警归并进事件：追加参数、时间线、升级严重度。"""
        ev.params.append({"point_id": alarm.point_id, "value": alarm.value, "detail": alarm.detail})
        ev.timeline.append({
            "ts": alarm.ts.isoformat(), "point_id": alarm.point_id,
            "event": f"{alarm.alarm_type.value}：{alarm.detail}",
        })
        if alarm.team and alarm.team not in ev.teams:
            ev.teams.append(alarm.team)
        # 严重度取最高
        if self.severity_order[alarm.severity] > self.severity_order[ev.severity]:
            ev.severity = alarm.severity
        # 趋势不触发；越限/突变/组合触发
        if alarm.alarm_type.value != "趋势":
            ev.triggered = True

    def close_expired(self, now: datetime | None = None) -> list[DefectEvent]:
        """返回并移除已过期的事件（窗口外不再归并）。"""
        now = now or datetime.now()
        done, keep = [], []
        for ev in self._open_events:
            if (now - ev.created_at).total_seconds() > self.window_seconds:
                done.append(ev)
            else:
                keep.append(ev)
        self._open_events = keep
        return done


def priority_key(ev: DefectEvent) -> tuple[int, datetime]:
    """事件优先级：重大 > 较大 > 一般；同级先发生先处理。"""
    order = {Severity.SERIOUS: 3, Severity.MAJOR: 2, Severity.GENERAL: 1}
    return (-order[ev.severity], ev.created_at)
