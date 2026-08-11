"""
实测填充：拒答率（领域外查询）+ 引用准确率（真实诊断输出）。

运行：python eval/fill_live_metrics.py
会调用 DeepSeek API（一次诊断：1 专家 + 1 主控），耗时约 1-2 分钟。
结果写入 data/metrics/live_metrics.json，供 metrics_report.py 读取。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from eval import metrics  # noqa: E402


# 领域外/无答案查询（拒答率评测集）
OUT_OF_DOMAIN_QUERIES = [
    "今天天气怎么样",
    "帮我写一首诗",
    "asdkjhqwezx",
    "股票明天涨不涨",
    "什么是相对论",
]


def build_live_metrics() -> dict:
    out = {}

    # 1. 拒答率（纯检索判定，不调 LLM）
    from backend.agents.experts.retriever import get_retriever
    r = get_retriever()
    out["refusal"] = metrics.refusal_rate(OUT_OF_DOMAIN_QUERIES, r, threshold=0.45)

    # 2. 引用准确率（真实诊断：磨煤机事件 → 锅炉专家意见）
    from backend.agents.monitor.agent import MonitorAgent
    from backend.agents.monitor.simulator import FaultScenario
    from backend.agents.experts.expert import get_expert
    from backend.agents.experts.schemas import DiagnosisContext

    async def _diag():
        agent = MonitorAgent()
        sc = [FaultScenario("M02-BRG-TEMP", 120.0, "ramp", rate=0.30)]
        ev = agent.run_simulation(scenarios=sc, steps=60, time_scale=60.0)[0]
        ctx = DiagnosisContext(
            event_id=ev.event_id, device=ev.device, teams=ev.teams or ["锅炉"],
            params=ev.params, timeline=ev.timeline, severity=ev.severity.value,
        )
        opinion = await get_expert("锅炉").diagnose(ctx)
        return opinion

    opinion = asyncio.run(_diag())
    opinions = [opinion.model_dump()]
    out["citation"] = metrics.citation_accuracy(opinions)
    out["citation"]["sample_opinion"] = {
        "team": opinion.team,
        "sources": opinion.sources,
        "causes": opinion.possible_causes[:3],
    }

    # 落盘
    os.makedirs(os.path.join(BASE_DIR, "data", "metrics"), exist_ok=True)
    path = os.path.join(BASE_DIR, "data", "metrics", "live_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


if __name__ == "__main__":
    build_live_metrics()
