"""两票辅助 Agent 数据模型：票草稿、审查结果、常见票型模板。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class TicketDraft(BaseModel):
    """工作票/操作票草稿（Agent 辅助生成，人工确认后走 ERP）。"""
    ticket_type: str = Field("工作票", description="工作票 | 操作票")
    ticket_id: str = Field("", description="票号（WP/GK/DQ 前缀）")
    device: str = Field("", description="作业设备")
    location: str = Field("", description="作业地点")
    task: str = Field("", description="工作任务")
    risk_level: str = Field("", description="风险等级：低风险/一般风险/较大风险/重大风险")
    hazard_analysis: str = Field("", description="危险点分析")
    safety_measures: str = Field("", description="安全措施")
    procedures: str = Field("", description="操作步骤")
    attachments: list[str] = Field(default_factory=list, description="关联证件：动火证/气体检测记录等")
    personnel: str = Field("", description="工作负责人/许可人/签发人")


class TicketCheckResult(BaseModel):
    """两票草稿审查结果。"""
    passed: bool = Field(False, description="是否通过")
    score: int = Field(0, ge=0, le=100, description="评分")
    issues: list[str] = Field(default_factory=list, description="问题清单")
    suggestions: list[str] = Field(default_factory=list, description="修改建议")
    required_attachments: list[str] = Field(default_factory=list, description="缺失的关联证件")


class CommonTicketTemplate(BaseModel):
    """常见票型模板（典型类型，非全覆盖）。"""
    template_id: str
    name: str                       # 如 "磨煤机检修"
    ticket_type: str = "工作票"
    task_template: str = Field("", description="任务模板（含 {device} 变量）")
    hazard_template: str = Field("", description="危险点模板")
    safety_template: str = Field("", description="安全措施模板")
    procedures_template: str = Field("", description="步骤模板")
    required_attachments: list[str] = Field(default_factory=list, description="常需证件")
