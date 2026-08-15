"""
班组专家 Agent 工具集（ReAct 自主工具决策）。

每个专家绑定本班组知识库工具，LLM 自主决定调用顺序与次数：
- search_regulation：查规程/FAQ 知识
- search_cases：查历史案例（知识闭环沉淀）
- get_timeline：查事件时间线（时序证据）
- finalize：提交最终结构化意见（必须调用以结束）

预算兜底：create_react_agent 的 recursion_limit 限制总步数（max_steps=5）。
审计：全部 thought/action/observation 留存在 messages（可溯源）。
"""
from __future__ import annotations

import json
from typing import Callable

from langchain_core.tools import tool

from backend.agents.experts.retriever import get_retriever
from backend.agents.experts.schemas import DiagnosisContext, ExpertOpinion
from backend.base.logger import get_logger

logger = get_logger(__name__)


def build_expert_tools(team: str, ctx: DiagnosisContext, trace_id: str | None = None) -> tuple[list, dict]:
    """
    构建专家工具集 + 共享状态（finalized 意见）。
    返回 (tools, state)；state["opinion"] 在 agent 完成后读取。
    """
    retriever = get_retriever()
    state: dict = {"opinion": None}

    def _fmt(results: list[dict], limit: int = 3) -> str:
        """检索结果格式化为工具观察文本（含来源编号）。"""
        lines = []
        for i, r in enumerate(results[:limit]):
            d = r["doc"]
            lines.append(f"[{i + 1}] 来源:{d.doc_id} 班组:{d.team} 匹配:{r['score']}\n{d.text[:200]}")
        return "\n".join(lines) if lines else "（无匹配结果）"

    @tool
    def search_regulation(query: str) -> str:
        """检索本班组规程与 FAQ 知识库。查询应包含设备名与异常现象，例如'磨煤机轴承温度高 原因 处置'。"""
        results = retriever.search(team, query, top_k=3, trace_id=trace_id)
        logger.info("expert.tool", team=team, tool="search_regulation", query=query[:30], hits=len(results))
        return _fmt(results)

    @tool
    def search_cases(query: str) -> str:
        """检索历史复盘案例库（同类故障的既往处置经验）。"""
        results = retriever.search(team, query, top_k=3, trace_id=trace_id)
        cases = [r for r in results if r["doc"].source == "case"]
        logger.info("expert.tool", team=team, tool="search_cases", query=query[:30], hits=len(cases))
        return _fmt(cases) + ("\n（提示：如无案例命中，可改用 search_regulation 查规程知识）" if not cases else "")

    @tool
    def get_timeline() -> str:
        """获取当前缺陷事件的时间线（各告警先后顺序，用于判断故障演化方向）。"""
        tl = "\n".join(
            f"{t.get('ts', '')} {t.get('point_id', '')}: {t.get('event', '')}" for t in ctx.timeline
        ) or "（无时间线）"
        return f"事件:{ctx.event_id} 设备:{ctx.device} 严重度:{ctx.severity}\n异常参数: {json.dumps(ctx.params, ensure_ascii=False)}\n时间线:\n{tl}"

    @tool
    def finalize(
        possible_causes: list[str],
        verification_methods: list[str],
        impact: str,
        disposal_reference: list[str],
        confidence: float,
        sources: list[str],
    ) -> str:
        """提交最终诊断意见。必须调用此工具结束诊断；confidence 须基于证据量给出（证据不足时降低）。"""
        state["opinion"] = ExpertOpinion(
            team=team,
            device=ctx.device,
            possible_causes=possible_causes,
            verification_methods=verification_methods,
            impact=impact,
            disposal_reference=disposal_reference,
            confidence=float(confidence),
            sources=sources,
        )
        logger.info("expert.finalized", team=team, causes=len(possible_causes), confidence=confidence)
        return "意见已提交，诊断结束。"

    return [search_regulation, search_cases, get_timeline, finalize], state
