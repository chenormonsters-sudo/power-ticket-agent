"""
M5 测试：复盘闭环 + 人工准入门控 + TraceID 查重 + 完整工作流。

单元测试不调 LLM（摘要注入 mock）；LLM 全链路标记 integration。
"""
import sys, os, asyncio, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from backend.agents.debrief.kb_updater import case_exists, ingest_case, load_cases
from backend.agents.debrief.schemas import CaseFeedback, DebriefSummary, ReviewCase
from backend.agents.experts.schemas import DiagnosisContext
from backend.graphs.workflow_graph import build_workflow_graph


@pytest.fixture(autouse=True)
def _mock_vector_increment(monkeypatch):
    """单元测试隔离向量缓存写入（真实向量增量由 integration 测试覆盖），
    避免测试污染 knowledge_base/index 缓存导致全量重编码。"""
    monkeypatch.setattr(
        'backend.agents.debrief.kb_updater.apply_vector_increment',
        lambda team=None: 0,
    )


@pytest.fixture(autouse=True)
def _clean_cases():
    """每个测试前后：备份/恢复案例库（防测试污染真实数据）。
    向量缓存不再备份——单元测试已隔离其写入（见 _mock_vector_increment）。"""
    import shutil
    from backend.agents.debrief.kb_updater import CASES_PATH
    backups = {}
    if os.path.exists(CASES_PATH):
        tmp = CASES_PATH + '.bak'
        shutil.copy2(CASES_PATH, tmp)
        backups['cases.json'] = tmp
    yield
    # 恢复备份；无备份则清理测试残留
    if 'cases.json' in backups:
        tmp = backups['cases.json']
        if os.path.exists(CASES_PATH):
            os.remove(CASES_PATH)
        shutil.move(tmp, CASES_PATH)
    elif os.path.exists(CASES_PATH):
        os.remove(CASES_PATH)


def _feedback(**over):
    base = dict(
        cause_correct=True, solution_effective=True,
        final_cause="轴承磨损", solution="更换轴承", result="参数恢复正常",
        extra_notes="", approved=True,
    )
    base.update(over)
    return CaseFeedback(**base)


def _draft_case(event_id="EV-RV-1"):
    return ReviewCase(
        case_id=f"CASE-{event_id}", event_id=event_id, device="磨煤机2#",
        teams=["锅炉"], params=[], timeline=[],
        final_cause="轴承磨损", solution="更换轴承",
    )


# ── 入库门控测试 ──

def test_ingest_requires_approval():
    """未批准（approved=False）不入库。"""
    c = ingest_case(_feedback(approved=False), _draft_case("EV-RV-NO-1"))
    assert c is None
    assert not case_exists("EV-RV-NO-1")


def test_ingest_requires_confirmation():
    """原因/方案未确认不入库。"""
    c = ingest_case(_feedback(cause_correct=False), _draft_case("EV-RV-NC-1"))
    assert c is None


def test_ingest_duplicate_trace():
    """TraceID 幂等：同事件不重复入库。"""
    c1 = ingest_case(_feedback(), _draft_case("EV-RV-DUP"))
    c2 = ingest_case(_feedback(), _draft_case("EV-RV-DUP"))
    assert c1 is not None
    assert c2 is None
    assert len([c for c in load_cases() if c.event_id == "EV-RV-DUP"]) == 1
    # 清理测试数据
    _cleanup("EV-RV-DUP")


def test_ingest_success():
    """批准+确认 → 入库成功。"""
    c = ingest_case(_feedback(final_cause="轴瓦磨损"), _draft_case("EV-RV-OK"))
    assert c is not None
    assert c.final_cause == "轴瓦磨损"
    _cleanup("EV-RV-OK")


def _cleanup(event_id: str):
    """清理测试产生的案例（保持环境干净）。"""
    from backend.agents.debrief.kb_updater import CASES_PATH
    if os.path.exists(CASES_PATH):
        cases = [c for c in load_cases() if c.event_id != event_id]
        from backend.agents.debrief.kb_updater import _save_cases
        _save_cases(cases)


# ── 完整闭环工作流测试（mock 摘要，不调 LLM）──

def _mock_summary(event_id="EV-RV-FLOW"):
    return DebriefSummary(
        event_id=event_id, device="磨煤机2#",
        suggested_cause="轴承磨损", suggested_solution="更换轴承",
        summary="测试复盘摘要",
    )


def _ctx():
    return DiagnosisContext(
        event_id="EV-RV-FLOW", device="磨煤机2#", teams=["锅炉"],
        params=[{"point_id": "M02-BRG-TEMP", "value": 88.0, "detail": "越限"}],
        timeline=[{"ts": "2026-08-10T00:00:00", "point_id": "P1", "event": "越限"}],
        severity="重大",
    )


def _initial():
    from backend.agents.orchestrator.schemas import OrchestratorOutput, ReferencePlan
    return {
        "ctx": _ctx(), "opinions": [], "qc_warnings": [],
        "output": OrchestratorOutput(plan=ReferencePlan(summary="测试", likely_causes=["轴承磨损"],
                                                        recommended_actions=["检查"], maintenance_reference=["磨煤机检修"]),
                                     opinions=[]),
        "human_decision_1": None, "human_decision_2": None,
        "ticket_draft": None, "ticket_check": None,
        "human_decision_3": None, "execute_status": "",
        "monitor_recovered": False,
        "debrief_summary": _mock_summary(),
        "human_decision_4": None, "ingested_case": None,
    }


def _interrupt_value(r: dict) -> dict:
    it = r.get("__interrupt__")
    assert it, "预期中断但未中断"
    return it[0].value


def test_workflow_full_loop_mock():
    """完整闭环（mock）：诊断→确认①→方案→两票→草稿确认→执行→恢复→复盘→准入入库。"""
    async def _run():
        graph = build_workflow_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "EV-RV-FLOW"}}

        r = await graph.ainvoke(_initial(), config)
        assert _interrupt_value(r)["type"] == "confirm_diagnosis"
        r = await graph.ainvoke(Command(resume={"confirmed": True}), config)
        assert _interrupt_value(r)["type"] == "confirm_plan"
        r = await graph.ainvoke(Command(resume={"need_ticket": True}), config)
        assert _interrupt_value(r)["type"] == "confirm_draft"
        r = await graph.ainvoke(Command(resume={"approved": True}), config)
        # 执行 → 监测恢复 → 复盘摘要 → 中断④
        assert _interrupt_value(r)["type"] == "confirm_debrief"
        # 复盘准入通过 → 入库
        r = await graph.ainvoke(Command(resume={
            "cause_correct": True, "solution_effective": True,
            "final_cause": "轴承磨损", "solution": "更换轴承",
            "result": "参数恢复正常", "extra_notes": "轴承间隙超标", "approved": True,
        }), config)
        assert r["execute_status"] == "executing"
        assert r["monitor_recovered"] is True
        assert r["ingested_case"] is not None
        assert r["ingested_case"].final_cause == "轴承磨损"
        assert r["ingested_case"].extra_notes == "轴承间隙超标"
        _cleanup("EV-RV-FLOW")

    asyncio.run(_run())


def test_workflow_reject_ingest():
    """复盘未批准 → 不入库（闭环但案例丢弃）。"""
    async def _run():
        graph = build_workflow_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "EV-RV-REJ"}}
        r = await graph.ainvoke(_initial(), config)
        r = await graph.ainvoke(Command(resume={"confirmed": True}), config)
        r = await graph.ainvoke(Command(resume={"need_ticket": False}), config)
        assert _interrupt_value(r)["type"] == "confirm_debrief"
        r = await graph.ainvoke(Command(resume={
            "cause_correct": False, "solution_effective": False,
            "final_cause": "", "solution": "", "result": "", "extra_notes": "",
            "approved": False,
        }), config)
        assert r["ingested_case"] is None

    asyncio.run(_run())


@pytest.mark.integration
def test_workflow_full_loop_llm():
    """完整闭环（调 LLM）：真实诊断 → 四级人工确认 → 复盘入库。"""
    from backend.agents.monitor.agent import MonitorAgent
    from backend.agents.monitor.simulator import FaultScenario
    from backend.graphs.diagnosis_graph import event_to_ctx

    async def _run():
        agent = MonitorAgent()
        sc = [FaultScenario("M02-BRG-TEMP", 120.0, "ramp", rate=0.30)]
        events = agent.run_simulation(scenarios=sc, steps=60, time_scale=60.0)
        initial = _initial()
        initial["ctx"] = event_to_ctx(events[0])
        initial["output"] = None
        initial["debrief_summary"] = None  # 复盘摘要真实生成

        graph = build_workflow_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": events[0].event_id}}
        r = await graph.ainvoke(initial, config)
        assert _interrupt_value(r)["type"] == "confirm_diagnosis"
        r = await graph.ainvoke(Command(resume={"confirmed": True}), config)
        assert _interrupt_value(r)["type"] == "confirm_plan"
        r = await graph.ainvoke(Command(resume={"need_ticket": False}), config)
        assert _interrupt_value(r)["type"] == "confirm_debrief"
        r = await graph.ainvoke(Command(resume={
            "cause_correct": True, "solution_effective": True,
            "final_cause": "测试确认原因", "solution": "测试方案",
            "result": "恢复", "extra_notes": "", "approved": True,
        }), config)
        assert r["ingested_case"] is not None
        _cleanup(r["ingested_case"].event_id)

    asyncio.run(_run())
