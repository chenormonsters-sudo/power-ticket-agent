"""
Streamlit 演示面板：火电两票协同智能审查与设备运维多 Agent 系统

运行：streamlit run web/app.py
功能：
- 监控看板：模拟 DCS 测点曲线 + 告警事件流
- 诊断工作台：缺陷事件 → 会诊过程 → 人工确认 → 两票草稿 → 复盘入库（全流程可视化）
- 知识闭环：已入库案例展示

依赖：streamlit、backend 模块（同 conda 环境）
"""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from backend.base.logger import configure_logging
from backend.agents.monitor.agent import MonitorAgent
from backend.agents.monitor.simulator import FaultScenario
from backend.agents.debrief.kb_updater import load_cases
from backend.graphs.diagnosis_graph import event_to_ctx
from backend.graphs.workflow_graph import build_workflow_graph

configure_logging()

st.set_page_config(page_title="火电两票协同智能审查系统", page_icon="⚡", layout="wide")
st.title("⚡ 火电两票协同智能审查与设备运维多 Agent 系统")


# ── 会话状态初始化 ──
def _init_state():
    st.session_state.setdefault("monitor_agent", None)
    st.session_state.setdefault("event", None)
    st.session_state.setdefault("graph", None)
    st.session_state.setdefault("config", None)
    st.session_state.setdefault("stage", "idle")   # idle|diagnosed|plan|draft|permit|executed|debrief|done
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("history", [])


_init_state()

# ── 侧边栏：系统说明与操作 ──
with st.sidebar:
    st.header("系统控制")
    st.caption("事件驱动编排式多 Agent：监测/班组专家×5/主控/两票辅助/复盘")
    if st.button("🚀 注入故障并启动流程", type="primary"):
        with st.spinner("监测 Agent 模拟 DCS 测点流（磨煤机2# 轴承温度爬升 + 振动突变）..."):
            agent = MonitorAgent()
            sc = [
                FaultScenario("M02-BRG-TEMP", 120.0, "ramp", rate=0.30),
                FaultScenario("M02-BRG-VIB", 200.0, "spike", target_delta=6.0),
            ]
            events = agent.run_simulation(scenarios=sc, steps=80, time_scale=60.0)
            ev = events[0]
            graph = build_workflow_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": ev.event_id}}
            st.session_state.monitor_agent = agent
            st.session_state.event = ev
            st.session_state.graph = graph
            st.session_state.config = config
            st.session_state.stage = "diagnosed"
            st.session_state.result = None
            st.session_state.history = [f"缺陷事件 {ev.event_id} | {ev.device} | 班组 {ev.teams} | 严重度 {ev.severity.value}"]
        st.rerun()

    if st.button("🧹 清空状态"):
        for k in ["monitor_agent", "event", "graph", "config", "stage", "result", "history"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.divider()
    st.caption("设计原则：无 AI 决策 | 人工分级确认 | 私有化合规 | 知识闭环")


# ── 工具函数：异步工作流调用 ──
def _run_graph(resume: dict | None = None):
    """调用工作流图；返回 (state, interrupt_value)。"""
    graph = st.session_state.graph
    config = st.session_state.config
    if resume is not None:
        state = asyncio.run(graph.ainvoke(Command(resume=resume), config))
    else:
        ev = st.session_state.event
        initial = {
            "ctx": event_to_ctx(ev), "opinions": [], "qc_warnings": [], "output": None,
            "human_decision_1": None, "human_decision_2": None,
            "ticket_draft": None, "ticket_check": None,
            "human_decision_3": None, "execute_status": "",
            "monitor_recovered": False, "debrief_summary": None,
            "human_decision_4": None, "ingested_case": None,
        }
        state = asyncio.run(graph.ainvoke(initial, config))
    it = state.get("__interrupt__")
    return state, (it[0].value if it else None)


# ── Tab1 监控看板 ──
tab1, tab2, tab3 = st.tabs(["📈 监控看板", "🔬 诊断工作台", "📚 知识闭环"])

with tab1:
    st.subheader("模拟 DCS 测点实时状态")
    cols = st.columns(5)
    points = [
        ("M02-BRG-TEMP", "磨煤机2# 轴承温度", "℃", 70, 85),
        ("M02-BRG-VIB", "磨煤机2# 振动", "mm/s", 3.5, 8.5),
        ("PAF-A-VIB", "一次风机A 振动", "mm/s", 4.0, 9.0),
        ("T01-W1-TEMP", "汽轮机1# 轴瓦温度", "℃", 62, 77),
        ("T01-LUB-P", "润滑油泵A 压力", "MPa", 0.25, 0.15),
    ]
    for col, (pid, name, unit, base, hi) in zip(cols, points):
        col.metric(label=name, value=f"{base}{unit}", delta=f"高限 {hi}{unit}")
    st.caption("注：演示环境为模拟测点数据（生产对接脱敏 DCS 接口）；监测规则引擎 24h 常驻、LLM 按需唤醒")

    if st.session_state.history:
        st.subheader("事件流")
        for h in st.session_state.history:
            st.info(h)

with tab2:
    st.subheader("诊断工作台（全流程人机协同）")
    stage = st.session_state.stage
    ev = st.session_state.event

    if ev is None:
        st.info("点击左侧「注入故障并启动流程」开始演示")
    else:
        st.success(f"当前事件：{ev.event_id} | {ev.device} | 严重度 {ev.severity.value}")
        st.caption("异常参数：" + "；".join(f"{p['point_id']}={p['value']}({p['detail']})" for p in ev.params[:4]))

        # 阶段 1：诊断完成 → 人工确认①
        if stage == "diagnosed":
            if st.session_state.result is None:
                with st.spinner("班组专家并行会诊中（LangGraph Send fan-out）..."):
                    state, iv = _run_graph()
                    st.session_state.result = state
            state = st.session_state.result
            plan = state["output"].plan
            st.markdown("#### 主控整合输出（参考，非结论）")
            st.write("**综合可能原因：**", plan.likely_causes)
            st.write("**参考建议：**", plan.recommended_actions)
            if plan.disagreements:
                st.warning(f"**分歧点 {len(plan.disagreements)} 个（交人工裁决）：**")
                for d in plan.disagreements:
                    st.write(f"- {d.topic}：{d.views}")
            st.caption("QC 告警：" + str(state.get("qc_warnings", [])))
            if st.button("✅ 人工确认①：现场核实诊断无误（interrupt 恢复）"):
                state, iv = _run_graph(resume={"confirmed": True, "note": "现场确认"})
                st.session_state.result = state
                st.session_state.stage = "plan"
                st.session_state.history.append("人工确认①通过：现场核实诊断")
                st.rerun()

        # 阶段 2：方案确认
        elif stage == "plan":
            if st.button("✅ 人工确认②：按方案执行，需要检修开票"):
                state, iv = _run_graph(resume={"need_ticket": True})
                st.session_state.result = state
                st.session_state.stage = "draft"
                st.session_state.history.append("人工确认②通过：按方案执行，需要检修")
                st.rerun()

        # 阶段 3：两票草稿
        elif stage == "draft":
            draft = st.session_state.result["ticket_draft"]
            check = st.session_state.result["ticket_check"]
            st.markdown("#### 两票辅助 Agent：草稿（辅助填票，走 ERP 前人工确认）")
            st.write("**任务：**", draft.get("task"))
            st.write("**危险点：**", draft.get("hazard_analysis") or "（待补充）")
            st.write("**安全措施：**", draft.get("safety_measures") or "（待补充）")
            st.write("**关联证件：**", draft.get("attachments"))
            st.warning(f"规则校验：通过={check.passed}，评分={check.score}")
            for i in check.issues:
                st.write(f"- {i}")
            st.caption("边界：系统只辅助填票与检查，确认后由填票人/负责人在 ERP 系统人工完成正式开票")
            if st.button("✅ 人工确认③：草稿核对无误（人工走 ERP 开票）"):
                state, iv = _run_graph(resume={"approved": True})
                st.session_state.result = state
                st.session_state.stage = "debrief"
                st.session_state.history.append("人工确认③通过：草稿定稿，人工走 ERP 开票")
                st.rerun()

        # 阶段 4：复盘确认
        elif stage == "debrief":
            summary = st.session_state.result["debrief_summary"]
            st.markdown("#### 复盘 Agent：处置闭环摘要（人工准入后入库）")
            st.write("**过程摘要：**", summary.summary)
            st.write("**建议原因：**", summary.suggested_cause)
            note = st.text_input("补充经验（可选）", value="轴承间隙超标，下次检修提前安排")
            if st.button("✅ 人工确认④：原因/方案/结果正确，批准入库"):
                state, iv = _run_graph(resume={
                    "cause_correct": True, "solution_effective": True,
                    "final_cause": "轴承磨损（确认）", "solution": "更换轴承",
                    "result": "参数恢复正常", "extra_notes": note, "approved": True,
                })
                st.session_state.result = state
                st.session_state.stage = "done"
                st.session_state.history.append("人工确认④通过：复盘批准，案例入库")
                st.rerun()

        # 阶段 5：完成
        elif stage == "done":
            state = st.session_state.result
            st.success("🎉 业务闭环完成")
            st.write("执行状态：", state["execute_status"], "| 监测恢复：", state["monitor_recovered"])
            if state.get("ingested_case"):
                c = state["ingested_case"]
                st.markdown("**案例已入库（知识闭环）：**")
                st.write(f"- 原因：{c.final_cause}")
                st.write(f"- 方案：{c.solution}")
                st.write(f"- 补充：{c.extra_notes}")
                st.caption("入库后成为知识库文档，后续同类故障检索匹配度提升（TraceID 幂等防重复）")
            st.button("🔁 再来一轮", on_click=lambda: st.session_state.update(stage="diagnosed", result=None))


with tab3:
    st.subheader("知识闭环：已入库案例")
    cases = load_cases()
    if not cases:
        st.info("暂无案例（完成一次闭环后自动入库）")
    else:
        df = pd.DataFrame([{
            "case_id": c.case_id, "device": c.device, "teams": ",".join(c.teams),
            "final_cause": c.final_cause[:30], "solution": c.solution[:30],
            "created_at": c.created_at,
        } for c in cases])
        st.dataframe(df, use_container_width=True)
