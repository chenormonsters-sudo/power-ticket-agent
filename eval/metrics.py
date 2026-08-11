"""
V2 综合评测指标模块（RAG 层 + Agent 层 + 质量层）。

指标口径：
- RAG 层：Top-K 命中率（run_eval.py 已有）、引用准确率、拒答率
- Agent 层：路由准确率、结构化输出成功率、人工接管率、低风险自动放行率
- 质量层：端到端闭环通过率、失败案例台账（failure_log.py）、时延

说明：引用准确率/拒答率/路由准确率用脚本计算；接管率/放行率/闭环通过率
依赖试运行台账（data/metrics/run_log.jsonl，见 metrics_report.py 生成样例）。
"""
from __future__ import annotations

import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(BASE_DIR, "eval")
KB_INDEX_DIR = os.path.join(BASE_DIR, "knowledge_base", "index")


def load_doc_ids() -> set[str]:
    """知识库有效 doc_id 集合（引用有效性判定基准）。"""
    meta_path = os.path.join(KB_INDEX_DIR, "vectors_meta.json")
    ids: set[str] = set()
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            for team_ids in json.load(f).values():
                ids.update(team_ids)
    return ids


def citation_accuracy(opinions: list[dict], doc_ids: set[str] | None = None) -> dict:
    """
    引用准确率：检查每条意见的 sources 是否有效（存在）且与意见内容相关。
    - 存在性：doc_id 在知识库索引中
    - 相关性：sources 对应块文本与意见文本的关键词重叠度 > 阈值
    返回 {valid_citation_rate, content_related_rate, total, checked}
    """
    doc_ids = doc_ids or load_doc_ids()
    import jieba

    valid, related, checked = 0, 0, 0
    for op in opinions:
        sources = op.get("sources", [])
        if not sources:
            continue
        checked += 1
        if all(s in doc_ids for s in sources):
            valid += 1
        # 相关性：取第一个来源块文本与意见原因文本做分词重叠
        overlap = 0.0
        op_text = "".join(op.get("possible_causes", []))[:200]
        op_tokens = set(jieba.cut(op_text))
        if op_tokens and sources:
            # 简化：意见原因关键词出现在引用块文本中的比例（此处用 doc_id 前缀匹配 FAQ 名称）
            meta = _load_doc_meta()
            for s in sources[:1]:
                chunk_text = meta.get(s, "")
                hit = sum(1 for t in op_tokens if t and len(t) > 1 and t in chunk_text)
                overlap = hit / max(len(op_tokens), 1)
        if overlap >= 0.15:
            related += 1

    return {
        "valid_citation_rate": round(valid / checked, 4) if checked else 0,
        "content_related_rate": round(related / checked, 4) if checked else 0,
        "checked": checked,
    }


_doc_meta_cache: dict[str, str] | None = None


def _load_doc_meta() -> dict[str, str]:
    """doc_id -> 文本（用于相关性判定）。"""
    global _doc_meta_cache
    if _doc_meta_cache is not None:
        return _doc_meta_cache
    docs_path = os.path.join(EVAL_DIR, "..", "knowledge_base", "index", "chunks.json")
    meta: dict[str, str] = {}
    try:
        with open(os.path.normpath(docs_path), encoding="utf-8") as f:
            chunks = json.load(f)
        for i, c in enumerate(chunks):
            meta[f"CHUNK-{i}"] = c.get("text", "")
    except Exception:  # noqa: BLE001
        pass
    # FAQ 文本（fault_name 作为引用块代表文本）
    faq_path = os.path.join(BASE_DIR, "knowledge_base", "faq", "faq_knowledge.json")
    try:
        with open(faq_path, encoding="utf-8") as f:
            for item in json.load(f):
                meta[item.get("id", "")] = item.get("fault_name", "") + item.get("cause", "")
    except Exception:  # noqa: BLE001
        pass
    _doc_meta_cache = meta
    return meta


def refusal_rate(no_answer_queries: list[str], retriever, team: str = "锅炉", threshold: float = 0.45) -> dict:
    """
    拒答率：无答案/超纲查询是否被正确拒绝（has_answer=False 且不产出答案）。
    评测：随机字符/领域外问题不应命中知识库（用指定班组检索判定）。
    """
    refused = 0
    for q in no_answer_queries:
        if not retriever.has_answer(team, q, threshold=threshold):
            refused += 1
    return {
        "refusal_rate": round(refused / len(no_answer_queries), 4) if no_answer_queries else 0,
        "refused": refused,
        "total": len(no_answer_queries),
    }


def route_accuracy(cases: list[dict]) -> dict:
    """
    路由准确率：缺陷事件实际激活班组 vs 预期班组。
    cases: [{event_id, expected_teams[], actual_teams[]}]
    """
    hit = 0
    for c in cases:
        exp = set(c["expected_teams"])
        act = set(c["actual_teams"])
        if exp and exp.issubset(act):
            hit += 1
    return {
        "route_accuracy": round(hit / len(cases), 4) if cases else 0,
        "hit": hit,
        "total": len(cases),
    }


def structured_output_success(attempts: int, failures: int) -> dict:
    """结构化输出成功率：LLM 输出通过 Pydantic 校验的比例。"""
    return {
        "structured_success_rate": round((attempts - failures) / attempts, 4) if attempts else 0,
        "attempts": attempts,
        "failures": failures,
    }


def human_takeover_metrics(run_log: list[dict]) -> dict:
    """
    人工接管率 & 低风险自动放行率（试运行台账统计）。
    run_log: [{severity: 重大/较大/一般, manual_confirm: bool, auto_released: bool}]
    """
    high = [r for r in run_log if r.get("severity") in ("重大", "较大")]
    low = [r for r in run_log if r.get("severity") == "一般"]
    high_manual = sum(1 for r in high if r.get("manual_confirm"))
    low_auto = sum(1 for r in low if r.get("auto_released"))
    return {
        "high_risk_takeover_rate": round(high_manual / len(high), 4) if high else 0,   # 安全底线：重大/较大必须 100% 人工
        "low_risk_auto_release_rate": round(low_auto / len(low), 4) if low else 0,     # 效率面：低风险自动放行
        "high_events": len(high),
        "low_events": len(low),
    }


def e2e_closure_rate(closed: int, total: int) -> dict:
    """端到端闭环通过率：告警→会诊→两票→确认→复盘入库 全链路成功比例。"""
    return {
        "e2e_closure_rate": round(closed / total, 4) if total else 0,
        "closed": closed,
        "total": total,
    }
