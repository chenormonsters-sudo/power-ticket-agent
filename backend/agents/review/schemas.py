"""
审查 Agent 的结构化输出 Schema。
用于 LLM 的 with_structured_output，让 LLM 返回固定结构的 JSON。
"""
from pydantic import BaseModel, Field
from typing import Optional


class DimensionReview(BaseModel):
    """单个维度的审查结果（LLM 输出的结构化格式）。"""
    score: int = Field(..., ge=0, le=100, description="该维度评分（0-100）")
    issues: list[str] = Field(default_factory=list, description="发现的问题列表")
    suggestions: list[str] = Field(default_factory=list, description="修改建议列表")
    passed: bool = Field(..., description="该维度是否通过审查")


class TicketReviewReport(BaseModel):
    """Agent 1 的最终输出：完整的审查报告。"""

    # ── 输入信息（来自票面，透传给报告）──
    ticket_id: str = Field("", description="工作票编号")
    ticket_task: str = Field("", description="工作任务描述")
    risk_level: str = Field("", description="风险等级（低风险/一般风险/较大风险/重大风险）")
    hazard_analysis: str = Field("", description="填写的危险点分析原文")
    procedures: str = Field("", description="操作步骤明细")

    # ── 审查结果 ──
    overall_score: int = Field(0, ge=0, le=100, description="综合评分")
    passed: bool = Field(False, description="是否通过审查")
    summary: str = Field("", description="综合评语")
    needs_manual_review: bool = Field(False, description="是否需要人工复核")
    dimensions: Optional[dict[str, DimensionReview]] = Field(
        None, description="各维度审查结果，key=维度名"
    )
