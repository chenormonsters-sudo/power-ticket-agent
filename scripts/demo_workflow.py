"""
一条龙业务闭环演示脚本（无 UI，验证全链路）：
告警 -> 建档 -> 并行会诊 -> 质检 -> 主控整合 -> 人工确认①（现场核实）
-> 方案确认 -> 两票草稿（辅助填票）-> 草稿确认 -> 执行 -> 监测恢复
-> 复盘摘要 -> 人工准入 -> 案例入库（知识增量）

运行：python scripts/demo_workflow.py
"""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from backend.base.logger import configure_logging
from backend.agents.monitor.agent import MonitorAgent
from backend.agents.monitor.simulator import FaultScenario
from backend.graphs.diagnosis_graph import event_to_ctx
from backend.graphs.workflow_graph import build_workflow_graph

configure_logging()


def _interrupt_value(r: dict) -> dict:
    it = r.get("__interrupt__")
    return it[0].value if it else None


async def main():
    print("=" * 70)
    print("火电两票协同智能审查与设备运维多 Agent 系统 — 业务闭环演示")
    print("=" * 70)

    # 1. 监测 Agent：注入磨煤机轴承温度爬升 + 振动突变
    print("\n[1] 监测 Agent 模拟 DCS 测点流（磨煤机2# 轴承温度爬升 + 振动突变）...")
    agent = MonitorAgent()
    sc = [
        FaultScenario("M02-BRG-TEMP", 120.0, "ramp", rate=0.30),
        FaultScenario("M02-BRG-VIB", 200.0, "spike", target_delta=6.0),
    ]
    events = agent.run_simulation(scenarios=sc, steps=80, time_scale=60.0)
    ev = events[0]
    print(f"    -> 缺陷事件 {ev.event_id} | {ev.device} | 班组 {ev.teams} | 严重度 {ev.severity.value}")
    print(f"      异常参数 {len(ev.params)} 个，时间线 {len(ev.timeline)} 条")

    # 2. 完整工作流（MemorySaver checkpoint，thread_id = event_id）
    print("\n[2] 启动诊断工作流（thread_id = event_id，断点可恢复）...")
    graph = build_workflow_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": ev.event_id}}
    initial = {
        "ctx": event_to_ctx(ev), "opinions": [], "qc_warnings": [],
        "output": None,
        "human_decision_1": None, "human_decision_2": None,
        "ticket_draft": None, "ticket_check": None,
        "human_decision_3": None, "execute_status": "",
        "monitor_recovered": False, "debrief_summary": None,
        "human_decision_4": None, "ingested_case": None,
    }

    # 2a. 诊断 + 人工确认①
    r = await graph.ainvoke(initial, config)
    v = _interrupt_value(r)
    print(f"\n[3] 会诊完成，待人工确认①（{v['type']}）：{v['question']}")
    plan = r["output"].plan
    print(f"    综合可能原因: {plan.likely_causes[:3]}")
    print(f"    参考建议: {plan.recommended_actions[:3]}")
    print(f"    分歧点: {len(plan.disagreements)}")
    for d in plan.disagreements:
        print(f"      · {d.topic}")
    print("    -> 运行人员现场核实，确认诊断正确...")
    r = await graph.ainvoke(Command(resume={"confirmed": True, "note": "现场确认轴承温度异常"}), config)

    # 2b. 方案确认
    v = _interrupt_value(r)
    print(f"\n[4] 待人工确认②（{v['type']}）：{v['question']}")
    print("    -> 确认按方案执行，需要检修开票...")
    r = await graph.ainvoke(Command(resume={"need_ticket": True}), config)

    # 2c. 两票辅助 + 草稿确认
    v = _interrupt_value(r)
    print(f"\n[5] 两票辅助生成草稿，待人工确认③（{v['type']}）")
    draft = r["ticket_draft"]
    check = r["ticket_check"]
    print(f"    草稿: {draft.get('template_ref', 'manual')} | {draft.get('task', '')[:30]}")
    print(f"    规则校验: 通过={check.passed} 评分={check.score}")
    for issue in check.issues:
        print(f"      · {issue}")
    print("    -> 填票人核对草稿无误，人工走 ERP 开票流程（系统不参与）...")
    r = await graph.ainvoke(Command(resume={"approved": True}), config)

    # 2d. 执行 -> 监测恢复 -> 复盘
    v = _interrupt_value(r)
    print(f"\n[6] 检修执行 -> 监测确认参数恢复 -> 复盘摘要，待人工确认④（{v['type']}）")
    summary = r["debrief_summary"]
    print(f"    复盘摘要: {summary.summary[:80]}")
    print(f"    建议原因: {summary.suggested_cause[:40]}")
    print("    -> 维护人员确认原因/方案/结果正确，批准入库...")
    r = await graph.ainvoke(Command(resume={
        "cause_correct": True, "solution_effective": True,
        "final_cause": "轴承磨损（确认）", "solution": "更换轴承",
        "result": "参数恢复正常", "extra_notes": "轴承间隙超标，下次检修提前安排",
        "approved": True,
    }), config)

    # 3. 闭环结果
    print("\n[7] 业务闭环完成 [OK]")
    print(f"    执行状态: {r['execute_status']} | 监测恢复: {r['monitor_recovered']}")
    case = r["ingested_case"]
    if case:
        print(f"    案例入库: {case.case_id}")
        print(f"      原因: {case.final_cause}")
        print(f"      方案: {case.solution}")
        print(f"      补充: {case.extra_notes}")
    else:
        print("    案例未入库（未批准/重复）")

    print("\n" + "=" * 70)
    print("演示结束:告警 -> 会诊 -> 方案 -> 两票 -> 人工确认 -> 执行 -> 复盘 -> 知识沉淀")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
