"""班组专家 Agent 数据模型。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ExpertOpinion(BaseModel):
    """班组专家对缺陷的诊断意见（结构化输出）。"""
    team: str = Field(..., description="班组名称")
    device: str = Field(..., description="设备名称")
    possible_causes: list[str] = Field(default_factory=list, description="可能原因（按可能性排序）")
    verification_methods: list[str] = Field(default_factory=list, description="验证方法")
    impact: str = Field("", description="影响评估")
    disposal_reference: list[str] = Field(default_factory=list, description="知识库处置参考条目")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="置信度")
    sources: list[str] = Field(default_factory=list, description="引用来源（doc_id）")


class DiagnosisContext(BaseModel):
    """传给专家 Agent 的缺陷上下文。"""
    event_id: str
    device: str
    teams: list[str]
    params: list[dict]          # [{point_id, value, detail}]
    timeline: list[dict]        # 事件时间线
    severity: str
