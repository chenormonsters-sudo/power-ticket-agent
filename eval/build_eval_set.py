"""
评测集构建：从 FAQ 知识库抽样生成测试集（查询 → 期望命中条目）。

策略：每个班组抽样 N 条 FAQ，以 fault_name 为查询，期望命中该 FAQ 自身。
输出：eval/test_set.json
"""
from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.experts.knowledge import TEAMS, build_team_index, load_kb

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SET_PATH = os.path.join(EVAL_DIR, "test_set.json")


def build_test_set(samples_per_team: int = 20, seed: int = 42) -> list[dict]:
    """构建测试集：每班组抽样 FAQ，查询=fault_name，期望=FAQ doc_id。"""
    docs = load_kb()
    team_index = build_team_index(docs)
    rng = random.Random(seed)
    test_set = []

    for team in TEAMS:
        faq_docs = [d for d in team_index[team] if d.source == "faq"]
        rng.shuffle(faq_docs)
        for d in faq_docs[:samples_per_team]:
            test_set.append({
                "query": d.meta.get("fault_name", d.text[:40]),
                "expected_doc_id": d.doc_id,
                "expected_team": team,
            })

    with open(TEST_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    print(f"评测集构建完成：{len(test_set)} 条（每班组 {samples_per_team} 条）→ {TEST_SET_PATH}")
    return test_set


if __name__ == "__main__":
    build_test_set()
