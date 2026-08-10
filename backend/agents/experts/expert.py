"""
班组专家 Agent：检索本班组知识库 + LLM 结构化诊断。

流程固定（可控优先）：检索（BM25+向量混合）→ 组装上下文 → LLM 输出结构化诊断意见。
专家只输出研判素材，不做最终结论（无 AI 决策）。
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.experts.retriever import get_retriever
from backend.agents.experts.schemas import DiagnosisContext, ExpertOpinion
from backend.base.llm_factory import get_structured_llm
from backend.base.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT_TMPL = """你是{team}班组资深专家，负责对设备缺陷给出专业研判。

工作方式：
1. 基于知识库检索结果（本班组规程/历史缺陷案例）分析
2. 输出可能原因（按可能性从高到低排序）、验证方法、影响评估
3. 从检索结果中引用处置参考条目
4. 给出置信度（0~1），证据不足时置信度降低

严格要求：
- 只输出研判素材，不做最终结论判定（最终结论由运行人员决定）
- 原因必须与检索到的知识相关，不得凭空编造
- 引用来源必须来自给定的检索结果"""

_USER_PROMPT_TMPL = """【缺陷事件】
事件编号：{event_id}
设备：{device}（{teams}）
严重度：{severity}

【异常参数】
{params}

【事件时间线】
{timeline}

【本班组知识库检索结果】
{search_results}

请基于以上材料输出{team}班组的专业研判意见。"""


class TeamExpertAgent:
    """班组专家 Agent。"""

    def __init__(self, team: str, agent_type: str, top_k: int = 5):
        self.team = team
        self.agent_type = agent_type      # llm_factory 路由键：expert_boiler 等
        self.top_k = top_k

    def _retrieve(self, query: str, trace_id: str | None = None) -> list[dict]:
        """混合检索本班组知识库（携带 TraceID 幂等）。"""
        retriever = get_retriever()
        results = retriever.search(self.team, query, top_k=self.top_k, trace_id=trace_id)
        logger.info("expert.retrieved", team=self.team, query=query[:30], hits=len(results))
        return results

    async def diagnose(self, ctx: DiagnosisContext, trace_id: str | None = None) -> ExpertOpinion:
        """对缺陷事件输出诊断意见（trace_id 缺省用 event_id）。"""
        trace_id = trace_id or ctx.event_id
        # 1. 检索：用设备+异常参数作为查询
        params_desc = "；".join(
            f"{p.get('point_id','')}={p.get('value','')}({p.get('detail','')})" for p in ctx.params
        )
        query = f"{ctx.device} {params_desc}"
        results = self._retrieve(query, trace_id=trace_id)

        # 2. 组装 Prompt
        search_block = "\n".join(
            f"[{i+1}] {r['doc'].text[:400]}（来源:{r['doc_id']}）"
            for i, r in enumerate(results)
        ) or "（本班组知识库无相关条目）"

        timeline_block = "\n".join(
            f"{t.get('ts','')} {t.get('point_id','')}: {t.get('event','')}" for t in ctx.timeline
        ) or "（无）"

        params_block = "\n".join(
            f"- {p.get('point_id','')}: {p.get('value','')} — {p.get('detail','')}" for p in ctx.params
        ) or "（无）"

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT_TMPL.format(team=self.team)),
            HumanMessage(content=_USER_PROMPT_TMPL.format(
                event_id=ctx.event_id, device=ctx.device,
                teams="、".join(ctx.teams), severity=ctx.severity,
                params=params_block, timeline=timeline_block,
                search_results=search_block, team=self.team,
            )),
        ]

        # 3. 结构化输出
        llm = get_structured_llm(self.agent_type, ExpertOpinion)
        opinion = await llm.ainvoke(messages)
        # 兜底：填充来源
        if not opinion.sources:
            opinion.sources = [r["doc_id"] for r in results[:3]]
        logger.info(
            "expert.diagnosed", team=self.team, device=ctx.device,
            causes=len(opinion.possible_causes), confidence=opinion.confidence,
        )
        return opinion


# 五班组专家注册（agent_type 对应 llm_factory 路由）
EXPERT_TEAMS: dict[str, str] = {
    "锅炉": "expert_boiler",
    "汽机": "expert_turbine",
    "电气": "expert_electric",
    "热控": "expert_hotcontrol",
    "燃除": "expert_coal",
}


def get_expert(team: str) -> TeamExpertAgent:
    """获取班组专家实例。"""
    agent_type = EXPERT_TEAMS.get(team, "expert_boiler")
    return TeamExpertAgent(team=team, agent_type=agent_type)
