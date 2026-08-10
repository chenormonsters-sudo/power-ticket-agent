"""
M3 诊断图测试：单元（默认，不调 LLM）+ 集成（标记 integration，调 DeepSeek API）。

运行：
- 单元：python -m pytest backend/graphs/test_diagnosis.py -v
- 全量（含 LLM）：python -m pytest backend/graphs/test_diagnosis.py -v -m integration
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime

import pytest

from backend.agents.experts.knowledge import TEAMS, build_team_index, load_kb
from backend.agents.experts.retriever import get_retriever
from backend.agents.experts.schemas import DiagnosisContext, ExpertOpinion
from backend.agents.orchestrator.orchestrator import OrchestratorAgent
from backend.agents.monitor.schemas import DefectEvent, Severity
from backend.graphs.diagnosis_graph import build_diagnosis_graph, event_to_ctx


# ── 单元测试：不调 LLM ──

def test_build_graph():
    """诊断图可编译（结构正确）。"""
    graph = build_diagnosis_graph()
    assert graph is not None


def test_kb_team_split():
    """知识库按五班组分库，共享库并入各班组。"""
    docs = load_kb()
    idx = build_team_index(docs)
    assert len(TEAMS) == 5
    for t in TEAMS:
        assert t in idx
        assert len(idx[t]) > 100  # 每个班组都有足够文档


def test_retriever_bm25_hit():
    """BM25 检索命中（不依赖向量）。"""
    r = get_retriever()
    results = r._bm25_search("锅炉", "磨煤机轴承温度高", top_k=5)
    assert len(results) > 0
    assert all(s > 0 for _, s in results)


def test_retriever_hybrid_hit():
    """混合检索返回结构化结果（用缓存向量）。"""
    r = get_retriever()
    results = r.search("锅炉", "磨煤机轴承温度高", top_k=3)
    assert len(results) > 0
    assert "doc" in results[0] and "score" in results[0] and "doc_id" in results[0]


def test_trace_idempotency():
    """TraceID 幂等：同 trace_id 重复检索直接复用缓存，不重复执行。"""
    r = get_retriever()
    q = "汽轮机轴瓦温度异常"
    r1 = r.search("汽机", q, top_k=3, trace_id="EV-TRACE-1")
    r2 = r.search("汽机", q, top_k=3, trace_id="EV-TRACE-1")
    assert r1 is r2  # 同一对象（缓存复用）
    r3 = r.search("汽机", q, top_k=3, trace_id="EV-TRACE-2")
    assert r3 is not r1  # 不同 TraceID 重新执行


def test_disagreement_detection():
    """规则层分歧检测：原因集合重叠度低 → 分歧点。"""
    a = ExpertOpinion(team="汽机", device="汽轮机1#",
                      possible_causes=["轴瓦磨损", "油膜振荡"], confidence=0.6,
                      sources=["FAQ-汽机-1"])
    b = ExpertOpinion(team="电气", device="汽轮机1#",
                      possible_causes=["定子绕组过流", "信号干扰"], confidence=0.45,
                      sources=["FAQ-电气-1"])
    points = OrchestratorAgent._detect_disagreements([a, b])
    assert len(points) >= 1


def test_no_ai_decision_schema():
    """无 AI 决策：参考方案默认需现场核实，结论由人决定。"""
    from backend.agents.orchestrator.schemas import ReferencePlan
    plan = ReferencePlan()
    assert plan.needs_field_check is True


def test_event_to_ctx():
    """DefectEvent → DiagnosisContext 转换正确。"""
    ev = DefectEvent(event_id="EV-1", device="磨煤机2#", teams=["锅炉"],
                     params=[{"point_id": "P1", "value": 90.0, "detail": "越限"}],
                     severity=Severity.SERIOUS,
                     timeline=[{"ts": "2026-08-10T00:00:00", "point_id": "P1", "event": "越限"}],
                     triggered=True)
    ctx = event_to_ctx(ev)
    assert ctx.event_id == "EV-1"
    assert ctx.teams == ["锅炉"]
    assert ctx.severity == "重大"


# ── 集成测试：调 LLM（标记 integration，默认跳过）──

@pytest.mark.integration
def test_cross_team_consultation_e2e():
    """端到端：跨班组事件 → 并行会诊 → 质检 → 主控整合（无 AI 决策）。"""
    import asyncio

    ev = DefectEvent(
        event_id="EV-TEST-CROSS-1", device="汽轮机1#", teams=["汽机", "电气"],
        params=[{"point_id": "T01-W1-TEMP", "value": 88.0, "detail": "越上限 77.0，超幅 11.0"},
                {"point_id": "T01-GEN-CURR", "value": 12.5, "detail": "定子电流波动"}],
        severity=Severity.SERIOUS,
        timeline=[{"ts": "2026-08-10T00:10:00", "point_id": "T01-W1-TEMP", "event": "越限"}],
        triggered=True,
    )
    graph = build_diagnosis_graph()

    async def _run():
        result = await graph.ainvoke({
            "ctx": event_to_ctx(ev), "opinions": [], "qc_warnings": [], "output": None,
        })
        return result

    result = asyncio.run(_run())
    out = result["output"]

    # 并行会诊：两个班组都出意见
    assert {o.team for o in out.opinions} == {"汽机", "电气"}
    # 质检无告警（来源/原因/置信度齐全）
    assert result["qc_warnings"] == []
    # 参考方案输出
    assert out.plan.likely_causes
    # 无 AI 决策：需要现场核实
    assert out.plan.needs_field_check is True
