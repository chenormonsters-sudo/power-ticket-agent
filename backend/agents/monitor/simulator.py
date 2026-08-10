"""
模拟测点生成器：代替生产 DCS 数据接口（演示/开发环境）。

- 正常工况：稳定基线 + 小幅随机波动
- 故障工况：指定测点在指定时间开始按预设曲线演化（爬升/突变/波动）
- 快进能力：time_scale 控制时间流速
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from backend.agents.monitor.schemas import Measurement


@dataclass
class PointSpec:
    """测点定义。"""
    point_id: str
    device: str
    team: str
    baseline: float        # 正常基线值
    noise: float = 0.5     # 正常波动幅度
    sample_interval: float = 10.0  # 采样间隔（秒）


@dataclass
class FaultScenario:
    """故障场景：某测点从 start_offset 秒开始按曲线演化。"""
    point_id: str
    start_offset: float          # 从模拟开始多少秒后触发
    mode: str = "ramp"           # ramp(爬升) | spike(突变) | oscillate(波动)
    rate: float = 0.05           # ramp: 每秒上升量
    target_delta: float = 20.0   # spike: 突变幅度; ramp 终点偏移


class DcsSimulator:
    """DCS 测点模拟器：按时间生成测点采样流。"""

    def __init__(self, specs: list[PointSpec], scenarios: list[FaultScenario] | None = None,
                 seed: int = 42, time_scale: float = 1.0):
        self.specs = {s.point_id: s for s in specs}
        self.scenarios = scenarios or []
        self.rng = random.Random(seed)
        self.time_scale = time_scale        # 快进倍数
        self._t = 0.0                       # 模拟内部时间（秒）
        self._sim_start = datetime.now()    # 模拟时间轴起点（所有采样时间戳由它派生）
        self._last_ts: dict[str, datetime] = {}
        self._scenario_state = {s.point_id: s for s in self.scenarios}
        self._fault_phase: dict[str, float] = {}  # point_id -> 已持续秒数

    def start_time(self) -> datetime:
        return datetime.now()

    def step(self) -> list[Measurement]:
        """推进一个采样周期，返回所有测点的新采样。"""
        # 模拟时间轴按原始采样间隔推进（time_scale 只影响现实执行速度，不影响模拟时间轴）
        first = next(iter(self.specs.values()))
        self._t += first.sample_interval
        # 采样时间戳由模拟时间轴派生（保证去抖/归并的时间窗口语义与生产一致）
        now = self._sim_start + __import__("datetime").timedelta(seconds=self._t)
        out: list[Measurement] = []
        for pid, spec in self.specs.items():
            value = self._sample(spec, now)
            out.append(Measurement(
                point_id=pid, device=spec.device, team=spec.team,
                value=round(value, 2), ts=now,
            ))
        return out

    @property
    def sample_interval(self) -> float:
        """返回采样间隔（秒，按快进倍率折算）。"""
        first = next(iter(self.specs.values()))
        return first.sample_interval / self.time_scale

    def _sample(self, spec: PointSpec, now: datetime) -> float:
        """计算测点当前值（基线 + 噪声 + 故障叠加）。"""
        value = spec.baseline + self.rng.uniform(-spec.noise, spec.noise)
        for sc in self.scenarios:
            if sc.point_id != spec.point_id:
                continue
            # 故障相位
            if sc.point_id not in self._fault_phase:
                self._fault_phase[sc.point_id] = 0.0
            phase = self._fault_phase[sc.point_id]
            fault_start = sc.start_offset

            if self._t >= fault_start:
                elapsed = self._t - fault_start
                self._fault_phase[sc.point_id] = elapsed
                if sc.mode == "ramp":
                    value += sc.rate * elapsed
                elif sc.mode == "spike":
                    value += sc.target_delta if elapsed < 30 else 0
                elif sc.mode == "oscillate":
                    value += sc.target_delta * abs(
                        __import__("math").sin(elapsed / 20)
                    )
        return value
