"""
综合指标报告：汇总 RAG 层 + Agent 层 + 质量层全部指标，输出 JSON + 可读摘要。

运行：python eval/metrics_report.py
说明：接管率/放行率/闭环通过率依赖试运行台账（data/metrics/run_log.jsonl），
     无台账时输出样例口径并标注"待试运行数据"。
"""
from __future__ import annotations

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from eval import failure_log, metrics
from eval.run_eval import run_eval  # V2 检索评测（Top-K 命中率）  # noqa: E402


def load_run_log() -> list[dict]:
    path = os.path.join(BASE_DIR, "data", "metrics", "run_log.jsonl")
    log = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    log.append(json.loads(line))
    return log


def build_report() -> dict:
    """生成综合指标报告。"""
    report = {}

    # 1. RAG 检索层（复用 run_eval）
    report["retrieval"] = run_eval()

    # 2. 引用准确率 & 拒答率（live_metrics.json 由 fill_live_metrics.py 生成）
    live_path = os.path.join(BASE_DIR, "data", "metrics", "live_metrics.json")
    if os.path.exists(live_path):
        with open(live_path, encoding="utf-8") as f:
            live = json.load(f)
        report["citation"] = live.get("citation", {})
        report["refusal"] = live.get("refusal", {})
    else:
        report["citation"] = metrics.citation_accuracy([])  # 框架：待 fill_live_metrics 填充
        report["refusal"] = {"hint": "运行 eval/fill_live_metrics.py 填充"}

    # 3. Agent 层（台账驱动）
    run_log = load_run_log()
    if run_log:
        report["human_takeover"] = metrics.human_takeover_metrics(run_log)
        report["e2e"] = metrics.e2e_closure_rate(
            closed=sum(1 for r in run_log if r.get("closed")), total=len(run_log))
    else:
        report["human_takeover"] = {"hint": "待试运行台账 data/metrics/run_log.jsonl"}
        report["e2e"] = {"hint": "待试运行台账"}

    # 4. 质量层：失败台账 + 缓存一致性自检
    report["failures"] = failure_log.count()
    report["cache_consistency"] = check_cache_consistency()

    return report


def check_cache_consistency() -> dict:
    """缓存一致性自检（F-001 事故的回归防线）。"""
    try:
        from backend.agents.experts.knowledge import TEAMS, build_team_index, load_kb
        meta_path = os.path.join(BASE_DIR, "knowledge_base", "index", "vectors_meta.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        idx = build_team_index(load_kb())
        mismatches = {t: (len(idx[t]), len(meta.get(t, []))) for t in TEAMS
                      if len(idx[t]) != len(meta.get(t, []))}
        return {"consistent": not mismatches, "mismatches": mismatches}
    except Exception as e:  # noqa: BLE001
        return {"consistent": False, "error": str(e)}


if __name__ == "__main__":
    report = build_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
