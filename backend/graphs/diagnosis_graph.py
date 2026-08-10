"""
诊断 LangGraph：事件 → 动态路由班组专家（Send 并行）→ 规则质检 → 主控整合。

并行：route_experts 按缺陷涉及班组 Send fan-out；expert_node 并行执行；fan-in 汇聚。
人机协同：本图为诊断阶段（不含人工确认节点，M4 接入 interrupt）。
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from backend.agents.experts.expert import EXPERT_TEAMS, get_expert
from backend.agents.experts.schemas import DiagnosisContext, ExpertOpinion
from backend.agents.orchestrator.orchestrator import OrchestratorAgent
from backend.agents.orchestrator.schemas import OrchestratorOutput
from backend.base.logger import get_logger

logger = get_logger(__name__)


class DiagnosisState(TypedDict):
    """诊断图共享状态。"""
    ctx: DiagnosisContext                 # 缺陷事件上下文
    opinions: Annotated[list[ExpertOpinion], operator.add]  # 专家意见累加器（并行写入）
    qc_warnings: list[str]                # 规则质检告警
    output: OrchestratorOutput | None     # 主控整合结果


# ── 节点 1：动态路由（Send fan-out）──
def route_experts(state: DiagnosisState):
    """按缺陷涉及班组动态激活专家（只激活相关班组，省算力）。"""
    ctx: DiagnosisContext = state["ctx"]
    teams = [t for t in ctx.teams if t in EXPERT_TEAMS] or ["锅炉"]  # 兜底
    logger.info("graph.route", evt=ctx.event_id, teams=teams)
    return [Send("expert_node", {"ctx": ctx, "team": t}) for t in teams]


# ── 节点 2：班组专家（并行执行）──
async def expert_node(state: dict) -> dict:
    """单个专家诊断（Send 目标节点）。trace_id 即缺陷事件 event_id（全局唯一）。"""
    ctx: DiagnosisContext = state["ctx"]
    team: str = state["team"]
    expert = get_expert(team)
    opinion = await expert.diagnose(ctx, trace_id=ctx.event_id)
    return {"opinions": [opinion]}


# ── 节点 3：规则质检（不调 LLM）──
def rule_qc(state: DiagnosisState) -> dict:
    """质检：来源引用 / 三要素覆盖 / 置信度标注。不达标记告警（不阻塞）。"""
    warnings: list[str] = []
    for o in state.get("opinions", []):
        if not o.sources:
            warnings.append(f"{o.team}班：缺少来源引用")
        if not o.possible_causes:
            warnings.append(f"{o.team}班：未给出可能原因")
        if o.confidence <= 0:
            warnings.append(f"{o.team}班：未标注置信度")
    logger.info("graph.qc", evt=state["ctx"].event_id, warnings=len(warnings))
    return {"qc_warnings": warnings}


# ── 节点 4：主控整合 ──
async def integrate_node(state: DiagnosisState) -> dict:
    """主控整合专家意见 → 参考处置方案 + 分歧清单。"""
    ctx: DiagnosisContext = state["ctx"]
    orchestrator = OrchestratorAgent()
    output = await orchestrator.integrate(ctx, state.get("opinions", []))
    logger.info("graph.integrated", evt=ctx.event_id)
    return {"output": output}


def build_diagnosis_graph():
    """构建诊断图。"""
    builder = StateGraph(DiagnosisState)

    builder.add_node("expert_node", expert_node)
    builder.add_node("rule_qc", rule_qc)
    builder.add_node("integrate_node", integrate_node)

    # 动态 fan-out：START 条件边返回 Send 列表 → 并行唤醒班组专家
    builder.add_conditional_edges(START, route_experts)
    # fan-in：全部专家完成后汇聚到主图继续
    builder.add_edge("expert_node", "rule_qc")
    builder.add_edge("rule_qc", "integrate_node")
    builder.add_edge("integrate_node", END)

    return builder.compile()


def event_to_ctx(event) -> DiagnosisContext:
    """把监测 Agent 的 DefectEvent 转为诊断上下文。"""
    return DiagnosisContext(
        event_id=event.event_id,
        device=event.device,
        teams=event.teams or ["锅炉"],
        params=event.params,
        timeline=event.timeline,
        severity=event.severity.value,
    )
