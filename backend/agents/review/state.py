"""
审查 Agent 的 State 定义。
StateGraph 中的 State 是一个 TypedDict，描述整个 Agent 的数据流。
"""
from typing import TypedDict, Optional
from backend.agents.review.schemas import DimensionReview, TicketReviewReport


class ReviewState(TypedDict):
    """工作票审查 Agent 的全局状态（填票前审查版本）。"""
    # ── 输入 ──
    ticket_text: str                    # 工作票原始文本
    ticket_id: str                      # 工作票编号
    ticket_task: str                    # 工作任务描述
    risk_level: str                     # 风险等级
    hazard_analysis: str                # 危险点分析原文
    procedures: str                     # 操作步骤明细

    # ── 中间状态（5 个维度各自的评审结果）──
    safety_review: Optional[DimensionReview]     # 维度1: 安全措施完备性
    procedure_review: Optional[DimensionReview]   # 维度2: 操作步骤逻辑
    hazard_review: Optional[DimensionReview]      # 维度3: 危险点分析充分性
    risk_review: Optional[DimensionReview]        # 维度4: 风险预控措施
    format_review: Optional[DimensionReview]      # 维度5: 格式规范性

    # ── 输出 ──
    report: Optional[TicketReviewReport]







