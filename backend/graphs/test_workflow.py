"""
M4 测试：两票规则/模板 + 工作流 interrupt 人机协同 + checkpoint 断点续跑。

单元测试默认不调 LLM（诊断结果注入 mock）；全链路 LLM 测试标记 integration。
"""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from backend.agents.monitor.schemas import DefectEvent, Severity
from backend.agents.experts.schemas import DiagnosisContext
from backend.agents.ticket_assist.schemas import TicketDraft
from backend.agents.ticket_assist.rules import (
    aggregate_rules, check_attachments, check_hazard_coverage, check_required_fields, run_rule_checks,
)
from backend.agents.ticket_assist.templates import COMMON_TEMPLATES, match_template
from backend.agents.ticket_assist.ticket_assist import TicketAssistAgent
from backend.graphs.workflow_graph import build_workflow_graph


# ── 两票规则测试 ──

def _draft(**over):
    base = dict(
        ticket_type="工作票", ticket_id="WP-20260810-001", device="磨煤机2#",
        location="锅炉房0米", task="磨煤机2#动火检修：检查磨辊磨损", risk_level="较大风险",
        hazard_analysis="机械伤害、高温烫伤、煤粉自燃、受限空间", safety_measures="隔离煤源、停电挂牌、通风检测、专人监护",
        procedures="1.办理工作票 2.隔离煤源 3.通风检测 4.检修 5.验收恢复",
        attachments=["动火证", "气体检测记录"], personnel="工作负责人：张三",
    )
    base.update(over)
    return TicketDraft(**base)


def test_rule_required_fields_ok():
    """完整票面通过必填项校验。"""
    r = check_required_fields(_draft())
    assert r.passed and r.score >= 60


def test_rule_required_fields_missing():
    """缺票号/任务 → 不通过。"""
    r = check_required_fields(_draft(ticket_id="", task=""))
    assert not r.passed
    assert any("票号" in i for i in r.issues)


def test_rule_attachments_fire():
    """动火作业缺动火证 → 拦截。"""
    r = check_attachments(_draft(attachments=[]))
    assert not r.passed
    assert any("动火证" in m for m in r.required_attachments)


def test_rule_hazard_coverage():
    """危险点缺恢复阶段 → 告警。"""
    r = check_hazard_coverage(_draft(hazard_analysis="隔离煤源、通风检测"))
    assert not r.passed


def test_rule_aggregate():
    """聚合：任一硬性失败即不通过。"""
    r = aggregate_rules(run_rule_checks(_draft(attachments=[])))
    assert not r.passed


# ── 模板测试 ──

def test_template_index():
    """常见票型库 5 个，磨煤机检修命中。"""
    assert len(COMMON_TEMPLATES) == 5
    t = match_template("磨煤机")
    assert t is not None and t.name == "磨煤机检修"


def test_template_generate_draft():
    """草稿变量预填（设备名替换）。"""
    agent = TicketAssistAgent()
    draft = agent.assist_draft(device="磨煤机3#", task_keyword="磨煤机检修")
    assert draft is not None
    assert "磨煤机3#" in draft["task"]
    assert "动火证" in draft["attachments"]


def test_rule_check_via_agent():
    """Agent 规则校验入口。"""
    agent = TicketAssistAgent()
    r = agent.check_rule_only(_draft(ticket_id=""))
    assert not r.passed


def _interrupt_value(r: dict) -> dict:
    """从 ainvoke 返回的 state 中提取 interrupt payload。"""
    it = r.get("__interrupt__")
    assert it, "预期中断但未中断"
    return it[0].value


# ── 工作流 interrupt 测试（不调 LLM：注入 mock 诊断输出）──

def _ctx():
    return DiagnosisContext(
        event_id="EV-WF-1", device="磨煤机2#", teams=["锅炉"],
        params=[{"point_id": "M02-BRG-TEMP", "value": 88.0, "detail": "越限"}],
        timeline=[{"ts": "2026-08-10T00:00:00", "point_id": "P1", "event": "越限"}],
        severity="重大",
    )


def _fake_output():
    from backend.agents.orchestrator.schemas import OrchestratorOutput, ReferencePlan
    plan = ReferencePlan(
        summary="测试方案", likely_causes=["轴承磨损"],
        recommended_actions=["现场检查确认"],
        maintenance_reference=["磨煤机检修"],
    )
    return OrchestratorOutput(plan=plan, opinions=[])


def _initial():
    return {
        "ctx": _ctx(), "opinions": [], "qc_warnings": [],
        "output": _fake_output(),
        "human_decision_1": None, "human_decision_2": None,
        "ticket_draft": None, "ticket_check": None,
        "human_decision_3": None, "execute_status": "",
    }


@pytest.mark.integration
def test_workflow_full_llm():
    """全链路（调 LLM）：监测事件 → 完整工作流 → 人工确认 → 执行。"""
    from backend.agents.monitor.agent import MonitorAgent
    from backend.agents.monitor.simulator import FaultScenario
    from backend.graphs.workflow_graph import run_workflow

    async def _run():
        agent = MonitorAgent()
        sc = [FaultScenario("M02-BRG-TEMP", 120.0, "ramp", rate=0.30)]
        events = agent.run_simulation(scenarios=sc, steps=60, time_scale=60.0)
        ev = events[0]
        from backend.graphs.diagnosis_graph import event_to_ctx
        initial = _initial()
        initial["ctx"] = event_to_ctx(ev)
        initial["output"] = None  # 让诊断子图真实执行

        graph = build_workflow_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": ev.event_id}}
        r = await graph.ainvoke(initial, config)
        v = _interrupt_value(r)
        assert v["type"] == "confirm_diagnosis"
        r = await graph.ainvoke(Command(resume={"confirmed": True, "note": "现场确认"}), config)
        v = _interrupt_value(r)
        assert v["type"] == "confirm_plan"
        r = await graph.ainvoke(Command(resume={"need_ticket": True}), config)
        v = _interrupt_value(r)
        assert v["type"] == "confirm_draft"
        r = await graph.ainvoke(Command(resume={"approved": True}), config)
        assert r["execute_status"] == "executing"

    asyncio.run(_run())


def test_workflow_interrupt_flow_mock():
    """人机协同流程（mock 诊断）：三次 interrupt 挂起/恢复，走完执行。"""
    async def _run():
        graph = build_workflow_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "EV-WF-MOCK"}}
        initial = _initial()

        # 中断① 现场核实诊断
        r = await graph.ainvoke(initial, config)
        v = _interrupt_value(r)
        assert v["type"] == "confirm_diagnosis"
        assert v["event_id"] == "EV-WF-1"

        # 恢复① → 中断② 方案确认
        r = await graph.ainvoke(Command(resume={"confirmed": True, "note": "现场确认轴承异常"}), config)
        v = _interrupt_value(r)
        assert v["type"] == "confirm_plan"

        # 恢复②（需检修）→ 两票辅助 → 中断③ 草稿定稿确认
        r = await graph.ainvoke(Command(resume={"need_ticket": True}), config)
        v = _interrupt_value(r)
        assert v["type"] == "confirm_draft"
        assert "ticket_draft" in r and r["ticket_draft"] is not None

        # 恢复③ → 执行跟踪
        r = await graph.ainvoke(Command(resume={"approved": True}), config)
        assert r["execute_status"] == "executing"

    asyncio.run(_run())


def test_workflow_no_ticket_branch():
    """不需要检修分支：跳过两票，直接执行。"""
    async def _run():
        graph = build_workflow_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "EV-WF-NOTICKET"}}
        r = await graph.ainvoke(_initial(), config)
        r = await graph.ainvoke(Command(resume={"confirmed": True}), config)
        r = await graph.ainvoke(Command(resume={"need_ticket": False}), config)
        assert r["execute_status"] == "executing"

    asyncio.run(_run())


def test_checkpoint_resume_persistence():
    """checkpoint 断点续跑：中断后新建图实例（同 saver）可恢复。"""
    async def _run():
        saver = MemorySaver()
        thread = "EV-WF-CP-1"
        g1 = build_workflow_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": thread}}
        r = await g1.ainvoke(_initial(), config)
        v = _interrupt_value(r)
        assert v["type"] == "confirm_diagnosis"

        # 模拟"服务重启"：新图实例 + 同一 saver + 同 thread_id → 恢复
        g2 = build_workflow_graph(checkpointer=saver)
        r2 = await g2.ainvoke(Command(resume={"confirmed": True}), config)
        v2 = _interrupt_value(r2)
        assert v2["type"] == "confirm_plan"
        r3 = await g2.ainvoke(Command(resume={"need_ticket": True}), config)
        v3 = _interrupt_value(r3)
        assert v3["type"] == "confirm_draft"
        r4 = await g2.ainvoke(Command(resume={"approved": True}), config)
        assert r4["execute_status"] == "executing"

    asyncio.run(_run())
