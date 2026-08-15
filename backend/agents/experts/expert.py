"""
班组专家 Agent：ReAct 工具自主决策（检索策略由模型自主规划，预算兜底）。

升级说明：不再固定"检索一次→分析→输出"，而是绑定工具集（查案例/查规程/看时间线），
由 LLM 自主决定检索路径与轮次；每次工具调用留痕可审计；max_steps 兜底防发散。
专家只输出研判素材，不做最终结论（无 AI 决策）。
"""
from __future__ import annotations

from backend.agents.experts.schemas import DiagnosisContext, ExpertOpinion
from backend.agents.experts.tools import build_expert_tools
from backend.base.llm_factory import get_llm
from backend.base.logger import get_logger

logger = get_logger(__name__)


class TeamExpertAgent:
    """班组专家 Agent（ReAct 工具自主决策）。

    升级说明：不再固定"检索一次→分析→输出"，而是绑定工具集，
    由 LLM 自主决定检索策略（先查案例还是规程、查几轮、证据不足追加检索），
    预算由 recursion_limit 兜底（max_steps=5），每次工具调用留痕可审计。
    """

    def __init__(self, team: str, agent_type: str, top_k: int = 3, max_steps: int = 5):
        self.team = team
        self.agent_type = agent_type      # llm_factory 路由键：expert_boiler 等
        self.top_k = top_k
        self.max_steps = max_steps

    async def diagnose(self, ctx: DiagnosisContext, trace_id: str | None = None) -> ExpertOpinion:
        """对缺陷事件自主规划检索并输出诊断意见（ReAct 循环）。"""
        from langgraph.prebuilt import create_react_agent
        from langchain_core.messages import HumanMessage

        trace_id = trace_id or ctx.event_id
        tools, state = build_expert_tools(self.team, ctx, trace_id)

        system = (
            f"你是{self.team}班组资深专家，负责对设备缺陷给出专业研判。\n\n"
            "你可以自主调用工具收集证据：\n"
            "- 先用 search_cases 查历史案例（同类故障经验），再用 search_regulation 查规程知识\n"
            "- 用 get_timeline 查看事件时间线（先后顺序有诊断意义：先振动后温度指向机械磨损，先温度后振动指向润滑恶化）\n"
            "- 证据不足时可换关键词追加检索（最多 5 次工具调用）\n"
            "- 证据充分后必须调用 finalize 提交意见，并基于证据量给出置信度（证据不足则降低）\n\n"
            "严格要求：只输出研判素材，不做最终结论判定（最终结论由运行人员决定）；"
            "原因必须与检索到的知识相关，不得凭空编造；引用来源必须来自检索结果。"
        )

        params_desc = "；".join(
            f"{p.get('point_id', '')}={p.get('value', '')}({p.get('detail', '')})" for p in ctx.params
        )
        task = (
            f"【缺陷事件】{ctx.event_id} | {ctx.device} | 班组 {ctx.teams} | 严重度 {ctx.severity}\n"
            f"【异常参数】{params_desc or '无'}\n"
            f"请自主规划检索路径，收集充分证据后调用 finalize 输出 {self.team} 班组诊断意见。"
        )

        agent = create_react_agent(
            model=get_llm(self.agent_type),
            tools=tools,
            prompt=system,
        )
        # 预算兜底：recursion_limit 控制总步数（thought+action+observation 每步计数），
        # 防 Agent 发散循环（max_steps 次工具调用 ≈ 3×max_steps+4）
        await agent.ainvoke(
            {"messages": [HumanMessage(content=task)]},
            config={"recursion_limit": self.max_steps * 3 + 4},
        )

        opinion = state.get("opinion")
        if opinion is None:
            # 兜底：未正常 finalize（超步数/异常）——不抛错，返回空意见并告警，由规则质检标记
            logger.warning("expert.no_finalize", team=self.team, evt=ctx.event_id, hint="agent did not finalize")
            opinion = ExpertOpinion(
                team=self.team, device=ctx.device,
                possible_causes=[], verification_methods=[], impact="",
                disposal_reference=[], confidence=0.0, sources=[],
            )
        logger.info("expert.diagnosed", team=self.team, device=ctx.device,
                    causes=len(opinion.possible_causes), confidence=opinion.confidence)
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
