"""
监测规则引擎：阈值越限 / 滑动窗口趋势 / 变化率 / 跨测点关联。

设计要点：
- 纯规则计算，常驻 24h 零 LLM 成本
- 趋势预警（TREND）仅作为运行提示，不触发完整诊断
- 越限（THRESHOLD）、突变（RATE）、组合异常（CORRELATION）触发完整诊断
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from typing import Callable

from backend.agents.monitor.schemas import Alarm, AlarmType, Measurement, Severity


class PointRule:
    """单个测点的监控规则。"""

    def __init__(
        self,
        point_id: str,
        device: str,
        team: str,
        high_limit: float | None = None,
        low_limit: float | None = None,
        rate_limit: float | None = None,       # 相邻采样最大变化（突变检测）
        trend_window: int = 10,                # 趋势检测滑动窗口（采样点数）
        trend_slope: float | None = None,      # 窗口内斜率阈值（值/窗口）
        window: int = 20,                      # 缓冲窗口大小
    ):
        self.point_id = point_id
        self.device = device
        self.team = team
        self.high_limit = high_limit
        self.low_limit = low_limit
        self.rate_limit = rate_limit
        self.trend_window = min(trend_window, window)
        self.trend_slope = trend_slope
        self.buf: deque[Measurement] = deque(maxlen=window)

    def feed(self, m: Measurement) -> list[Alarm]:
        """喂入一条采样，返回触发的告警列表（可能为空）。"""
        alarms: list[Alarm] = []
        self.buf.append(m)

        # 1. 阈值越限（触发完整诊断）
        if self.high_limit is not None and m.value >= self.high_limit:
            alarms.append(Alarm(
                point_id=m.point_id, device=m.device, team=m.team,
                value=m.value, alarm_type=AlarmType.THRESHOLD,
                severity=self._severity_by_margin(m.value - self.high_limit, 0),
                ts=m.ts, threshold=self.high_limit,
                detail=f"越上限 {self.high_limit}，超幅 {m.value - self.high_limit:.1f}",
            ))
        elif self.low_limit is not None and m.value <= self.low_limit:
            alarms.append(Alarm(
                point_id=m.point_id, device=m.device, team=m.team,
                value=m.value, alarm_type=AlarmType.THRESHOLD,
                severity=self._severity_by_margin(self.low_limit - m.value, 0),
                ts=m.ts, threshold=self.low_limit,
                detail=f"越下限 {self.low_limit}，低幅 {self.low_limit - m.value:.1f}",
            ))

        # 2. 变化率突变（触发完整诊断）
        if self.rate_limit is not None and len(self.buf) >= 2:
            prev = self.buf[-2].value
            delta = abs(m.value - prev)
            if delta >= self.rate_limit:
                alarms.append(Alarm(
                    point_id=m.point_id, device=m.device, team=m.team,
                    value=m.value, alarm_type=AlarmType.RATE,
                    severity=Severity.GENERAL, ts=m.ts,
                    detail=f"相邻采样突变 {delta:.1f}（限 {self.rate_limit}）",
                ))

        # 3. 趋势检测（仅运行提示，不触发诊断）
        if self.trend_slope is not None and len(self.buf) >= self.trend_window:
            recent = list(self.buf)[-self.trend_window:]
            slope = self._linear_slope(recent)
            # 持续单调爬升/下降且斜率超阈值
            if abs(slope) >= self.trend_slope and self._monotonic(recent):
                direction = "爬升" if slope > 0 else "下降"
                alarms.append(Alarm(
                    point_id=m.point_id, device=m.device, team=m.team,
                    value=m.value, alarm_type=AlarmType.TREND,
                    severity=Severity.GENERAL, ts=m.ts,
                    detail=f"趋势{direction}，窗口斜率 {slope:.2f}（限 {self.trend_slope}）",
                ))

        return alarms

    @staticmethod
    def _linear_slope(ms: list[Measurement]) -> float:
        """最小二乘拟合斜率（按索引近似时间）。"""
        n = len(ms)
        if n < 2:
            return 0.0
        xs = list(range(n))
        ys = [m.value for m in ms]
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den = sum((x - x_mean) ** 2 for x in xs)
        return num / den if den else 0.0

    @staticmethod
    def _monotonic(ms: list[Measurement], tolerance: float = 0.5) -> bool:
        """判断序列是否单调（允许小幅波动 tolerance）。"""
        vals = [m.value for m in ms]
        up = all(b >= a - tolerance for a, b in zip(vals, vals[1:]))
        down = all(b <= a + tolerance for a, b in zip(vals, vals[1:]))
        return up or down

    @staticmethod
    def _severity_by_margin(margin: float, _: float = 0) -> Severity:
        """按超限幅度定级：大幅超限 → 重大，否则较大。"""
        if margin >= 10:
            return Severity.SERIOUS
        return Severity.MAJOR


class CorrelationRule:
    """跨测点关联规则：同一设备相关测点在时间窗内同时异常 → 组合告警（高置信）。"""

    def __init__(
        self,
        device: str,
        team: str,
        related_points: list[str],   # 关联测点集合
        window_seconds: int = 300,   # 组合判定时间窗
        min_hits: int = 2,           # 至少几个测点异常才触发
    ):
        self.device = device
        self.team = team
        self.related_points = set(related_points)
        self.window_seconds = window_seconds
        self.min_hits = min_hits
        self.recent: deque[tuple[datetime, str, str]] = deque()  # (ts, point_id, detail)

    def feed(self, alarm: Alarm) -> Alarm | None:
        """喂入告警，若满足组合条件返回 CORRELATION 告警。"""
        if alarm.device != self.device or alarm.point_id not in self.related_points:
            return None
        if alarm.alarm_type == AlarmType.TREND:
            return None  # 趋势预警不参与组合触发

        cutoff = alarm.ts - timedelta(seconds=self.window_seconds)
        while self.recent and self.recent[0][0] < cutoff:
            self.recent.popleft()
        self.recent.append((alarm.ts, alarm.point_id, alarm.detail))

        hit_points = {p for _, p, _ in self.recent}
        if len(hit_points) >= self.min_hits:
            detail = "组合异常：" + "、".join(
                f"{p}({d})" for _, p, d in list(self.recent)[-self.min_hits:]
            )
            return Alarm(
                point_id=", ".join(sorted(hit_points)), device=self.device, team=self.team,
                value=0.0, alarm_type=AlarmType.CORRELATION,
                severity=Severity.MAJOR, ts=alarm.ts, detail=detail,
            )
        return None


class MonitorRules:
    """监测规则引擎：管理所有测点规则与关联规则。"""

    def __init__(self, point_rules: list[PointRule], correlation_rules: list[CorrelationRule] | None = None):
        self.point_rules = {r.point_id: r for r in point_rules}
        self.correlation_rules = correlation_rules or []

    def feed(self, m: Measurement) -> list[Alarm]:
        """喂入采样，返回所有触发的告警（含组合告警）。"""
        rule = self.point_rules.get(m.point_id)
        if rule is None:
            return []
        alarms = rule.feed(m)
        for corr in self.correlation_rules:
            for a in list(alarms):
                combined = corr.feed(a)
                if combined:
                    alarms.append(combined)
        return alarms
