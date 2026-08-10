"""
复盘 Agent：处置闭环后生成复盘摘要 → 人工确认（准入）→ 入库 → 索引增量。

流程：
1. 监测确认参数恢复正常 → 触发复盘
2. LLM 生成复盘摘要草稿（引用前序诊断，标注"待人工确认"）
3. 人工确认节点（interrupt）：确认原因/方案/结果，填写最终结论
4. 人工准入通过 → 入库（TraceID 查重）→ 标记向量增量重建
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.debrief.kb_updater import ingest_case, mark_vector_rebuild
from backend.agents.debrief.schemas import CaseFeedback, DebriefSummary, ReviewCase
from backend.agents.experts.schemas import DiagnosisContext, ExpertOpinion
from backend.agents.orchestrator.schemas import OrchestratorOutput
from backend.base.llm_factory import get_structured_llm
from backend.base.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """你负责生成设备缺陷处置复盘摘要（草稿）。

基于前序诊断意见与参考方案，总结：
- 建议原因（标注为待人工确认）
- 建议方案
- 过程摘要（事件→诊断→方案）

要求：内容必须源自前序 Agent 输出，不得虚构；输出为结构化 JSON。"""


class DebriefAgent:
    """复盘 Agent。"""

    def __init__(self, agent_type: str = "review"):
        self.agent_type = agent_type

    async def generate_summary(
        self,
        ctx: DiagnosisContext,
        opinions: list[ExpertOpinion],
        output: OrchestratorOutput,
    ) -> DebriefSummary:
        """生成复盘摘要草稿（LLM，结果待人工确认）。"""
        causes = []
        for o in opinions:
            causes.extend(o.possible_causes)
        plan = output.plan

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"【事件】{ctx.event_id} | {ctx.device} | {ctx.severity}\n"
                f"【专家意见】{('；'.join(causes[:6])) or '无'}\n"
                f"【参考方案】{plan.summary}\n"
                f"【建议动作】{('；'.join(plan.recommended_actions[:4])) or '无'}"
            )),
        ]
        summary: DebriefSummary = await get_structured_llm(
            self.agent_type, DebriefSummary
        ).ainvoke(messages)
        summary.event_id = ctx.event_id
        summary.device = ctx.device
        logger.info("debrief.summary_generated", evt=ctx.event_id)
        return summary

    def build_draft_case(
        self, ctx: DiagnosisContext, summary: DebriefSummary
    ) -> ReviewCase:
        """由摘要构建待确认案例（final 字段留空，人工填写）。"""
        return ReviewCase(
            case_id=f"CASE-{ctx.event_id}",
            event_id=ctx.event_id,
            device=ctx.device,
            teams=ctx.teams,
            params=ctx.params,
            timeline=ctx.timeline,
            final_cause=summary.suggested_cause,
            solution=summary.suggested_solution,
            result="",
            extra_notes="",
        )

    def confirm_and_ingest(self, feedback: CaseFeedback, draft: ReviewCase) -> ReviewCase | None:
        """人工确认准入 → 入库 → 向量增量追加（不重建已有缓存）。"""
        case = ingest_case(feedback, draft)
        if case is not None:
            from backend.agents.debrief.kb_updater import apply_vector_increment
            apply_vector_increment(team=case.teams[0] if case.teams else None)
        return case
