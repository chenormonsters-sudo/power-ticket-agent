"""
失败案例台账：每条失败记录 时间/环节/根因/影响/修复/规避。

内置首条真实事故：知识库缓存失配 → 全量重编码（load_kb 未加载复盘案例）。
"""
from __future__ import annotations

import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(BASE_DIR, "data", "metrics", "failures.json")

# 内置真实事故（开发期实际发生并修复）
_BUILTIN = [
    {
        "id": "F-001",
        "date": "2026-08-10",
        "stage": "知识库缓存加载",
        "phenomenon": "服务重启后 BGE-M3 CPU 全量重编码 3560 条（25-30 分钟），CPU 打满，测试/评测/演示启动极慢",
        "root_cause": "load_kb 只加载基础规程/FAQ，未加载 data/cases/cases.json 复盘案例 → 知识库文档数与向量缓存元数据条数不匹配 → 缓存被判失效触发全量重建",
        "impact": "pytest 每次运行 15 分钟+；多进程并发写缓存互相踩踏；演示无法快速启动",
        "fix": "load_kb 增加案例加载（案例成为可检索知识）；mark_vector_rebuild 废弃改为 apply_vector_increment 真增量追加；debrief 测试隔离向量缓存写入",
        "verification": "缓存一致性检查（meta 与 team_index 全班组条数比对 OK）；全量测试 32 passed / 24s；评测秒级",
        "prevention": "scripts/manage_procs.ps1 进程治理 + run_tests.ps1 标准流程；新增指标脚本每次跑缓存一致性校验",
    }
]


def load_failures() -> list[dict]:
    """加载失败台账（内置事故 + 新增记录）。"""
    records = list(_BUILTIN)
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, encoding="utf-8") as f:
            records.extend(json.load(f))
    return records


def record_failure(stage: str, phenomenon: str, root_cause: str,
                   impact: str, fix: str, prevention: str = "") -> dict:
    """记录新失败案例。"""
    entry = {
        "id": f"F-{len(load_failures()) + 1:03d}",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "stage": stage,
        "phenomenon": phenomenon,
        "root_cause": root_cause,
        "impact": impact,
        "fix": fix,
        "verification": "待验证",
        "prevention": prevention,
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    extra = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, encoding="utf-8") as f:
            extra = json.load(f)
    extra.append(entry)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(extra, f, ensure_ascii=False, indent=2)
    return entry


def count() -> dict:
    """台账统计（按环节分组）。"""
    records = load_failures()
    by_stage: dict[str, int] = {}
    for r in records:
        by_stage[r["stage"]] = by_stage.get(r["stage"], 0) + 1
    return {"total": len(records), "by_stage": by_stage}
