"""
主控 Agent（Orchestrator）：整合多专家研判素材、梳理分歧要点、生成结构化参考处置方案。

边界（无 AI 决策）：
- 不做最终结论判定，只做信息整合
- 专家意见分歧时输出分歧点清单，由人工裁决
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from backend.agents.experts.schemas import DiagnosisContext, ExpertOpinion


class DisagreementPoint(BaseModel):
    """分歧点：专家意见不一致之处。"""
    topic: str = Field(..., description="分歧主题")
    views: list[str] = Field(default_factory=list, description="各方观点（含班组与依据）")


class ReferencePlan(BaseModel):
    """结构化参考处置方案（仅供人工参考）。"""
    summary: str = Field("", description="综合研判摘要")
    likely_causes: list[str] = Field(default_factory=list, description="综合可能原因（按多数专家/置信度）")
    recommended_actions: list[str] = Field(default_factory=list, description="参考处置建议（运行侧）")
    maintenance_reference: list[str] = Field(default_factory=list, description="检修参考条目（来自知识库）")
    disagreements: list[DisagreementPoint] = Field(default_factory=list, description="分歧点清单（供人工裁决）")
    needs_field_check: bool = Field(True, description="是否需要现场核实（默认需要）")


class OrchestratorOutput(BaseModel):
    """主控输出：参考方案 + 分歧清单（最终结论由人工决定）。"""
    plan: ReferencePlan
    opinions: list[ExpertOpinion] = Field(default_factory=list, description="各专家原始意见")
    ctx: DiagnosisContext | None = None
