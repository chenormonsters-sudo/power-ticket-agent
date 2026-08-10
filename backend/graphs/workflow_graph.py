"""
完整工作流图（M4 版）：诊断 → 人工确认①（现场核实）→ 方案确认 → 两票辅助 → 许可 → 执行跟踪。

关键机制：
- 人机协同：interrupt() 挂起等待人工处理，Command(resume) 恢复执行（分级：低风险自动放行）
- 断点续跑：SqliteSaver checkpoint 落盘，服务重启可恢复
- TraceID：thread_id 使用缺陷事件 event_id（全局唯一，幂等）
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from backend.agents.debrief.debrief import DebriefAgent
from backend.agents.debrief.schemas import CaseFeedback, DebriefSummary, ReviewCase
from backend.agents.experts.schemas import DiagnosisContext, ExpertOpinion
from backend.agents.orchestrator.schemas import OrchestratorOutput
from backend.agents.ticket_assist.schemas import TicketCheckResult
from backend.agents.ticket_assist.ticket_assist import TicketAssistAgent
from backend.base.tracing import get_tracing_callbacks
from backend.graphs.diagnosis_graph import build_diagnosis_graph, event_to_ctx
from backend.base.logger import get_logger

logger = get_logger(__name__)


class WorkflowState(TypedDict):
    """完整工作流状态。"""
    ctx: DiagnosisContext
    opinions: Annotated[list[ExpertOpinion], operator.add]
    qc_warnings: list[str]
    output: OrchestratorOutput | None
    human_decision_1: dict | None      # 现场核实诊断
    human_decision_2: dict | None      # 确认处置方案（含是否需要检修开票）
    ticket_draft: dict | None          # 两票草稿
    ticket_check: TicketCheckResult | None
    human_decision_3: dict | None      # 草稿定稿确认
    execute_status: str                # 执行跟踪状态
    monitor_recovered: bool            # 监测确认参数恢复正常
    debrief_summary: DebriefSummary | None
    human_decision_4: dict | None      # 复盘人工确认（准入）
    ingested_case: ReviewCase | None   # 入库结果


# ── 节点 1：诊断（复用诊断子图）──
async def diagnose_node(state: WorkflowState) -> dict:
    """调用诊断子图：并行会诊 → 质检 → 主控整合。
    幂等 guard：state 已有诊断结果（断点恢复/测试注入）则跳过，不重复调 LLM。"""
    if state.get("output") is not None:
        logger.info("wf.diagnose_skipped", evt=state["ctx"].event_id, reason="output_exists")
        return {}
    diagnosis = build_diagnosis_graph()
    result = await diagnosis.ainvoke({
        "ctx": state["ctx"], "opinions": [], "qc_warnings": [], "output": None,
    })
    logger.info("wf.diagnosed", evt=state["ctx"].event_id)
    return {
        "opinions": result["opinions"],
        "qc_warnings": result["qc_warnings"],
        "output": result["output"],
    }


# ── 节点 2：人工确认① 现场核实诊断（interrupt）──
def human_confirm_1(state: WorkflowState) -> dict:
    """现场核实诊断：运行人员确认诊断结论与参考方案是否可行。"""
    decision = interrupt({
        "type": "confirm_diagnosis",
        "event_id": state["ctx"].event_id,
        "question": "请现场核实诊断结论：原因判断是否与现场情况一致？参考方案是否可行？",
        "plan": state["output"].plan.model_dump() if state["output"] else None,
        "qc_warnings": state.get("qc_warnings", []),
    })
    return {"human_decision_1": decision}


# ── 节点 3：方案确认（interrupt）──
def plan_confirm(state: WorkflowState) -> dict:
    """确认处置方案：运行人员结合现场判断是否按方案执行、是否需要检修开票。"""
    decision = interrupt({
        "type": "confirm_plan",
        "event_id": state["ctx"].event_id,
        "question": "请确认处置方案：是否按参考方案执行？是否需要检修作业（开工作票）？",
        "plan": state["output"].plan.model_dump() if state["output"] else None,
    })
    return {"human_decision_2": decision}


def _need_ticket(state: WorkflowState) -> str:
    """条件边：是否需要检修开票（人工在方案确认中决定）。"""
    d = state.get("human_decision_2") or {}
    if d.get("need_ticket") is True:
        return "ticket_assist_node"
    return "execute_track"


# ── 节点 4：两票辅助（草稿生成 + 规则校验）──
async def ticket_assist_node(state: WorkflowState) -> dict:
    """按常见票型生成草稿 + 规则硬校验（LLM 语义审查按需开启）。"""
    agent = TicketAssistAgent()
    device = state["ctx"].device
    # 从诊断结论提取检修关键词（取参考方案维护参考第一条作为模板匹配词）
    keyword = ""
    plan = state["output"].plan if state["output"] else None
    if plan and plan.maintenance_reference:
        keyword = plan.maintenance_reference[0][:8]
    draft = agent.assist_draft(device=device, task_keyword=keyword or device)
    if draft is None:
        # 无模板命中：生成最小草稿供人工填写
        draft = {
            "ticket_type": "工作票", "device": device, "location": "",
            "task": f"{device}检修作业", "hazard_analysis": "", "safety_measures": "",
            "procedures": "", "attachments": [], "template_ref": "manual",
        }
    from backend.agents.ticket_assist.schemas import TicketDraft
    td = TicketDraft(**{k: draft.get(k, "") for k in
                        ["ticket_type", "ticket_id", "device", "location", "task",
                         "risk_level", "hazard_analysis", "safety_measures",
                         "procedures", "attachments", "personnel"]})
    check = agent.check_rule_only(td)
    logger.info("wf.ticket_assist", evt=state["ctx"].event_id, template=draft.get("template_ref"))
    return {"ticket_draft": draft, "ticket_check": check}


# ── 节点 5：人工确认② 草稿定稿确认（interrupt）──
def human_confirm_2(state: WorkflowState) -> dict:
    """草稿定稿确认：填票人/负责人核对草稿无误后，自行走 ERP 真实开票流程。
    边界：系统只辅助填票与检查，不提交、不签发、不模拟真实许可环节。"""
    decision = interrupt({
        "type": "confirm_draft",
        "event_id": state["ctx"].event_id,
        "question": "请核对工作票草稿：内容是否正确完整？确认后由填票人/负责人在 ERP 系统中人工完成正式开票（系统不提交、不签发、不参与后续流程）。",
        "ticket_draft": state.get("ticket_draft"),
        "ticket_check": state["ticket_check"].model_dump() if state.get("ticket_check") else None,
    })
    return {"human_decision_3": decision}


# ── 节点 6：执行跟踪 ──
def execute_track(state: WorkflowState) -> dict:
    """记录执行状态（消缺执行中）。"""
    status = "executing"
    logger.info("wf.execute_start", evt=state["ctx"].event_id)
    return {"execute_status": status}


# ── 节点 7：监测恢复确认 ──
def monitor_verify(state: WorkflowState) -> dict:
    """监测 Agent 确认参数恢复正常（生产：监测规则检测恢复；演示：模拟/人工确认）。"""
    logger.info("wf.monitor_recovered", evt=state["ctx"].event_id)
    return {"monitor_recovered": True}


# ── 节点 8：复盘摘要生成（LLM，幂等 guard）──
async def debrief_summary_node(state: WorkflowState) -> dict:
    """生成复盘摘要草稿（待人工确认）。已有摘要则跳过（断点恢复/测试注入）。"""
    if state.get("debrief_summary") is not None:
        return {}
    agent = DebriefAgent()
    summary = await agent.generate_summary(
        state["ctx"], state.get("opinions", []), state["output"],
    )
    return {"debrief_summary": summary}


# ── 节点 9：复盘人工确认（interrupt，准入门控）──
def human_confirm_4(state: WorkflowState) -> dict:
    """人工确认复盘：确认故障原因/方案/结果是否正确，批准后入库。
    防止前序 Agent 判断错误污染知识库——硬门控。"""
    decision = interrupt({
        "type": "confirm_debrief",
        "event_id": state["ctx"].event_id,
        "question": "请确认复盘结论：故障原因与处置方案是否正确？有无补充？确认通过后将入库沉淀为案例。",
        "debrief_summary": state["debrief_summary"].model_dump() if state.get("debrief_summary") else None,
    })
    return {"human_decision_4": decision}


# ── 节点 10：人工准入入库 ──
def ingest_node(state: WorkflowState) -> dict:
    """按人工反馈准入入库（TraceID 查重 + 向量增量标记）。"""
    feedback = CaseFeedback(**{k: state["human_decision_4"].get(k) for k in
                               ["cause_correct", "solution_effective", "final_cause",
                                "solution", "result", "extra_notes", "approved"]})
    draft = DebriefAgent().build_draft_case(state["ctx"], state["debrief_summary"])
    case = DebriefAgent().confirm_and_ingest(feedback, draft)
    logger.info("wf.ingest_done", evt=state["ctx"].event_id, ingested=case is not None)
    return {"ingested_case": case}


def build_workflow_graph(checkpointer: AsyncSqliteSaver | None = None):
    """构建完整工作流图（可选持久化 checkpoint）。"""
    builder = StateGraph(WorkflowState)

    builder.add_node("diagnose_node", diagnose_node)
    builder.add_node("human_confirm_1", human_confirm_1)
    builder.add_node("plan_confirm", plan_confirm)
    builder.add_node("ticket_assist_node", ticket_assist_node)
    builder.add_node("human_confirm_2", human_confirm_2)
    builder.add_node("execute_track", execute_track)
    builder.add_node("monitor_verify", monitor_verify)
    builder.add_node("debrief_summary_node", debrief_summary_node)
    builder.add_node("human_confirm_4", human_confirm_4)
    builder.add_node("ingest_node", ingest_node)

    builder.add_edge(START, "diagnose_node")
    builder.add_edge("diagnose_node", "human_confirm_1")
    builder.add_edge("human_confirm_1", "plan_confirm")
    builder.add_conditional_edges("plan_confirm", _need_ticket,
                                  {"ticket_assist_node": "ticket_assist_node",
                                   "execute_track": "execute_track"})
    builder.add_edge("ticket_assist_node", "human_confirm_2")
    builder.add_edge("human_confirm_2", "execute_track")
    builder.add_edge("execute_track", "monitor_verify")
    builder.add_edge("monitor_verify", "debrief_summary_node")
    builder.add_edge("debrief_summary_node", "human_confirm_4")
    builder.add_edge("human_confirm_4", "ingest_node")
    builder.add_edge("ingest_node", END)

    return builder.compile(checkpointer=checkpointer)


async def run_workflow(initial: dict, thread_id: str, checkpointer: AsyncSqliteSaver | None = None):
    """
    运行工作流直至人工中断点；返回 (结果, 中断点列表)。
    调用方在中断后收集人工决策，用 Command(resume=...) 恢复。
    可选携带 Langfuse 追踪回调（故障事后溯源/审计）。
    """
    graph = build_workflow_graph(checkpointer)
    callbacks = get_tracing_callbacks()
    config = {"configurable": {"thread_id": thread_id}}
    if callbacks:
        config["callbacks"] = callbacks
    result = await graph.ainvoke(initial, config)
    return result, config
