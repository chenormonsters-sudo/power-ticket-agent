"""
Agent 1 的 LangGraph 节点函数。
"""
import re
from langchain_core.messages import SystemMessage, HumanMessage

from backend.base.llm_factory import get_structured_llm
from backend.base.retry import with_retry
from backend.base.logger import get_logger
from backend.agents.review.state import ReviewState
from backend.agents.review.schemas import DimensionReview, TicketReviewReport
from backend.agents.review.prompts import (
    SAFETY_SYSTEM_PROMPT, PROCEDURE_SYSTEM_PROMPT,
    HAZARD_SYSTEM_PROMPT, RISK_SYSTEM_PROMPT,
    check_format, build_user_prompt,
)

logger = get_logger(__name__)


# ── 辅助函数 ──

def extract_field(text: str, pattern: str, group: int = 2) -> str:
    """从文本中用正则提取字段值。"""
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(group).strip() if match else ""


# ── 解析节点 ──

async def parse_ticket_node(state: ReviewState) -> dict:
    """从 ticket_text 中提取结构化字段。"""
    text = state["ticket_text"]
    return {
        "ticket_id": extract_field(text, r"(WP|GK|DQ)[-\s]?\d+", group=0),
        "ticket_task": extract_field(text, r"(工作任务|工作内容)[：:](.*?)(?:\n|$)"),
        "risk_level": extract_field(text, r"(风险等级)[：:](.*?)(?:\n|$)"),
        "hazard_analysis": extract_field(text, r"(危险点|风险分析)[：:](.*?)(?:\n|$)"),
        "procedures": extract_field(text, r"(步骤|操作|工艺)[：:](.*?)$"),
    }


# ── LLM 审查节点工厂 ──

def _make_review_node(dimension_key: str, system_prompt: str):
    """工厂函数：生成一个 LLM 审查节点。"""

    async def review_node(state: ReviewState) -> dict:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=build_user_prompt(state["ticket_text"], dimension_key)),
        ]

        llm = get_structured_llm("ticket_review", DimensionReview)

        @with_retry(agent_type="ticket_review")
        async def _invoke():
            return await llm.ainvoke(messages)

        result = await _invoke()
        logger.info("review.dimension", dimension=dimension_key, score=result.score)

        return {f"{dimension_key}_review": result}

    return review_node


# ── 4 个 LLM 审查节点 ──

check_safety_node = _make_review_node("safety", SAFETY_SYSTEM_PROMPT)
check_procedure_node = _make_review_node("procedure", PROCEDURE_SYSTEM_PROMPT)
check_hazard_node = _make_review_node("hazard", HAZARD_SYSTEM_PROMPT)
check_risk_node = _make_review_node("risk", RISK_SYSTEM_PROMPT)


# ── 规则引擎审查节点 ──

async def check_format_node(state: ReviewState) -> dict:
    """格式规范性审查（不调 LLM）。"""
    return {"format_review": check_format(state["ticket_text"])}


# ── 聚合节点 ──

async def aggregate_node(state: ReviewState) -> dict:
    """汇总 5 个维度，生成最终报告。"""
    dims = {
        "safety": state["safety_review"],
        "procedure": state["procedure_review"],
        "hazard": state["hazard_review"],
        "risk": state["risk_review"],
        "format": state["format_review"],
    }

    weights = {"safety": 0.30, "procedure": 0.20, "hazard": 0.25, "risk": 0.15, "format": 0.10}
    total = sum(dims[k].score * w for k, w in weights.items() if dims[k] is not None)
    overall = round(total)

    # 否决项
    needs_manual = (dims["safety"] and not dims["safety"].passed) or \
                   (dims["hazard"] and not dims["hazard"].passed)

    all_issues = []
    for dim in dims.values():
        if dim:
            all_issues.extend(dim.issues)

    if needs_manual:
        summary = "安全措施或危险点分析存在不通过项，需要人工复核。"
    elif overall >= 80:
        summary = "整体情况良好，工作票内容规范。"
    elif overall >= 60:
        summary = "基本符合要求，部分问题已列出，建议修改后复审。"
    else:
        summary = "存在较多问题，建议重新填写。"

    report = TicketReviewReport(
        ticket_id=state.get("ticket_id", ""),
        ticket_task=state.get("ticket_task", ""),
        risk_level=state.get("risk_level", ""),
        hazard_analysis=state.get("hazard_analysis", ""),
        procedures=state.get("procedures", ""),
        overall_score=overall,
        passed=overall >= 60 and not needs_manual,
        needs_manual_review=needs_manual,
        summary=summary,
        dimensions={k: v for k, v in dims.items() if v is not None},
    )

    return {"report": report}