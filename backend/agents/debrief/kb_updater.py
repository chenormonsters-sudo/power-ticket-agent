"""
知识库增量更新器：复盘案例人工准入入库 + 索引增量。

入库规则（硬门控）：
- 人工确认（CaseFeedback.approved=True）才允许入库
- 入库流程：清洗 → 结构化 → 写案例库 → 标记向量索引需重建（增量）

存储：案例库 JSON（data/cases/cases.json，M7 可迁移 MySQL）。
向量增量：案例并入对应班组文档集后重建该班组向量缓存（案例量小，重建成本低）。
"""
from __future__ import annotations

import json
import os
import threading

from backend.agents.debrief.schemas import CaseFeedback, ReviewCase
from backend.base.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CASES_PATH = os.path.join(PROJECT_ROOT, "data", "cases", "cases.json")

_lock = threading.Lock()


def load_cases() -> list[ReviewCase]:
    """加载已入库案例。"""
    if not os.path.exists(CASES_PATH):
        return []
    with open(CASES_PATH, encoding="utf-8") as f:
        return [ReviewCase(**c) for c in json.load(f)]


def _save_cases(cases: list[ReviewCase]):
    os.makedirs(os.path.dirname(CASES_PATH), exist_ok=True)
    with open(CASES_PATH, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in cases], f, ensure_ascii=False, indent=2)


def case_exists(event_id: str) -> bool:
    """幂等查重：同一事件不重复入库（TraceID 幂等）。"""
    return any(c.event_id == event_id for c in load_cases())


def ingest_case(feedback: CaseFeedback, draft: ReviewCase) -> ReviewCase | None:
    """
    人工准入入库：approved=True 且原因/方案确认 → 写案例库。
    返回入库的案例；未批准/重复返回 None。
    """
    if not feedback.approved:
        logger.info("debrief.ingest_rejected", evt=draft.event_id, reason="not_approved")
        return None
    if not (feedback.cause_correct and feedback.solution_effective):
        logger.info("debrief.ingest_rejected", evt=draft.event_id, reason="not_confirmed")
        return None
    if case_exists(draft.event_id):
        logger.info("debrief.ingest_skipped", evt=draft.event_id, reason="duplicate_trace")
        return None

    case = ReviewCase(
        case_id=f"CASE-{draft.event_id}",
        event_id=draft.event_id,
        device=draft.device,
        teams=draft.teams,
        params=draft.params,
        timeline=draft.timeline,
        final_cause=feedback.final_cause or draft.final_cause,
        solution=feedback.solution or draft.solution,
        result=feedback.result,
        extra_notes=feedback.extra_notes,
    )

    with _lock:
        cases = load_cases()
        if any(c.event_id == case.event_id for c in cases):  # 双检防并发重复
            return None
        cases.append(case)
        _save_cases(cases)

    logger.info("debrief.ingested", evt=case.event_id, case=case.case_id)
    return case


def mark_vector_rebuild(team: str | None = None):
    """
    【已废弃】旧实现会删除向量缓存元数据导致全量重建（20+ 分钟）——已由 apply_vector_increment 替代。
    保留仅为兼容引用；新代码请用 apply_vector_increment。
    """
    logger.warning("debrief.vector_mark_deprecated", hint="use apply_vector_increment")
    apply_vector_increment(team)


def apply_vector_increment(team: str | None = None):
    """
    知识库增量更新：把已入库但尚未进向量缓存的案例，增量追加到对应班组缓存。
    - 不删除、不重建已有缓存（旧实现缺陷的修复）
    - 幂等：doc_id 已在缓存则跳过
    - team=None 时按案例所属班组逐个处理
    """
    from backend.agents.experts.knowledge import KbDoc
    from backend.agents.experts.retriever import get_retriever

    cases = load_cases()
    retriever = get_retriever()
    teams = [team] if team else sorted({c.teams[0] for c in cases if c.teams})
    total = 0
    for t in teams:
        docs = [
            KbDoc(
                doc_id=c.case_id, team=t,
                text=f"案例：{c.device}；原因：{c.final_cause}；处置：{c.solution}；结果：{c.result}；补充：{c.extra_notes}",
                source="case", meta={"event_id": c.event_id},
            )
            for c in cases if c.teams and c.teams[0] == t
        ]
        if docs:
            n = retriever.append_docs(t, docs)
            total += n
    logger.info("debrief.vector_increment_done", teams=teams, appended=total)
    return total
