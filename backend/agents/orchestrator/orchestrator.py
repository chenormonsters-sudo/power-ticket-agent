"""
主控 Agent 实现：整合专家意见（规则层分歧检测 + LLM 方案整合）。

分工：
- 规则层：分歧检测（专家原因集合重叠度低/置信度接近 → 记分歧点）
- LLM 层：生成综合研判摘要与参考处置方案
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.experts.schemas import DiagnosisContext, ExpertOpinion
from backend.agents.orchestrator.schemas import (
    DisagreementPoint, OrchestratorOutput, ReferencePlan,
)
from backend.base.llm_factory import get_structured_llm
from backend.base.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """你是火电厂运维值班长，负责综合各班组专家的研判意见，形成参考处置方案。

职责：
1. 综合多专家意见，梳理共识与分歧
2. 生成参考处置方案（运行侧建议 + 检修参考）
3. 输出分歧点清单

严格要求：
- 你不做最终结论判定，最终结论由运行人员结合现场决定
- 方案必须基于专家意见与知识库内容，不得凭空编造
- 意见分歧时如实呈现各方观点与依据"""

_USER_PROMPT_TMPL = """【缺陷事件】{event_id} | {device} | 严重度 {severity}
【异常参数】{params}
【事件时间线】{timeline}

【各班组专家意见】
{opinions}

请整合为参考处置方案。"""


class OrchestratorAgent:
    """主控 Agent。"""

    def __init__(self, agent_type: str = "orchestrator"):
        self.agent_type = agent_type

    @staticmethod
    def _detect_disagreements(opinions: list[ExpertOpinion]) -> list[DisagreementPoint]:
        """规则层分歧检测：两个专家可能原因集合重合度低 → 分歧点。"""
        points: list[DisagreementPoint] = []
        for i in range(len(opinions)):
            for j in range(i + 1, len(opinions)):
                a, b = opinions[i], opinions[j]
                if a.team == b.team:
                    continue
                set_a = {c[:6] for c in a.possible_causes}
                set_b = {c[:6] for c in b.possible_causes}
                overlap = len(set_a & set_b) / max(len(set_a | set_b), 1)
                if overlap < 0.3 and set_a and set_b:
                    points.append(DisagreementPoint(
                        topic=f"{a.team}与{b.team}对原因判断不一致",
                        views=[f"{a.team}: " + "；".join(a.possible_causes[:3]),
                               f"{b.team}: " + "；".join(b.possible_causes[:3])],
                    ))
        return points

    async def integrate(self, ctx: DiagnosisContext, opinions: list[ExpertOpinion]) -> OrchestratorOutput:
        """整合专家意见 → 参考方案。"""
        # 1. 规则层：分歧检测（不调 LLM）
        disagreements = self._detect_disagreements(opinions)

        # 2. LLM 生成参考方案
        opinions_block = "\n\n".join(
            f"[{o.team}班] 置信度{o.confidence:.2f}\n"
            f"可能原因: {'; '.join(o.possible_causes)}\n"
            f"验证方法: {'; '.join(o.verification_methods)}\n"
            f"影响: {o.impact}\n"
            f"处置参考: {'; '.join(o.disposal_reference[:3])}"
            for o in opinions
        ) or "（无专家意见）"

        params_block = "；".join(
            f"{p.get('point_id','')}={p.get('value','')}({p.get('detail','')})" for p in ctx.params
        )
        timeline_block = "\n".join(
            f"{t.get('ts','')} {t.get('point_id','')}: {t.get('event','')}" for t in ctx.timeline
        )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_USER_PROMPT_TMPL.format(
                event_id=ctx.event_id, device=ctx.device, severity=ctx.severity,
                params=params_block, timeline=timeline_block, opinions=opinions_block,
            )),
        ]

        plan: ReferencePlan = await get_structured_llm(self.agent_type, ReferencePlan).ainvoke(messages)

        # 3. 规则层补充：把规则检测到的分歧并入 LLM 结果（双保险）
        seen = {p.topic for p in plan.disagreements}
        for d in disagreements:
            if d.topic not in seen:
                plan.disagreements.append(d)

        logger.info(
            "orchestrator.integrated", evt=ctx.event_id,
            experts=len(opinions), disagreements=len(plan.disagreements),
        )
        return OrchestratorOutput(plan=plan, opinions=opinions, ctx=ctx)
