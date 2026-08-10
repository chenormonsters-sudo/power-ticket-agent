"""复盘 Agent 数据模型：案例、人工反馈、入库记录。"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ReviewCase(BaseModel):
    """复盘案例（人工确认通过后入库）。"""
    case_id: str = Field("", description="案例编号（事件号+日期）")
    event_id: str = Field("", description="关联缺陷事件 TraceID")
    device: str = Field("", description="设备")
    teams: list[str] = Field(default_factory=list, description="涉及班组")
    params: list[dict] = Field(default_factory=list, description="异常参数快照")
    timeline: list[dict] = Field(default_factory=list, description="事件时间线")
    final_cause: str = Field("", description="最终故障原因（人工确认）")
    solution: str = Field("", description="处置方案（人工确认）")
    result: str = Field("", description="执行结果（人工确认）")
    extra_notes: str = Field("", description="人工补充经验")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class CaseFeedback(BaseModel):
    """人工反馈（复盘确认节点）。"""
    cause_correct: bool = Field(False, description="故障原因是否确认正确")
    solution_effective: bool = Field(False, description="处置方案是否有效")
    final_cause: str = Field("", description="人工填写的最终原因")
    solution: str = Field("", description="人工填写的处置方案")
    result: str = Field("", description="执行结果")
    extra_notes: str = Field("", description="补充经验")
    approved: bool = Field(False, description="是否确认入库")


class DebriefSummary(BaseModel):
    """复盘 Agent LLM 生成的总结草稿。"""
    event_id: str = Field("", description="事件号")
    device: str = Field("", description="设备")
    suggested_cause: str = Field("", description="建议原因（来自前序诊断，待人工确认）")
    suggested_solution: str = Field("", description="建议方案")
    summary: str = Field("", description="过程摘要")
