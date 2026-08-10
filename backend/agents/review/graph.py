"""
Agent 1：工作票审查 Agent 的 LangGraph 图装配。
"""
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from langgraph.graph import StateGraph, START, END

from backend.agents.review.state import ReviewState
from backend.agents.review.nodes import (
    parse_ticket_node,
    check_safety_node,
    check_procedure_node,
    check_hazard_node,
    check_risk_node,
    check_format_node,
    aggregate_node,
)


def build_review_graph():
    """构建工作票审查 Agent 的 LangGraph。"""

    # 1. 创建图构建器
    builder = StateGraph(ReviewState)

    # 2. 注册节点
    builder.add_node("parse_ticket", parse_ticket_node)
    builder.add_node("check_safety", check_safety_node)
    builder.add_node("check_procedure", check_procedure_node)
    builder.add_node("check_hazard", check_hazard_node)
    builder.add_node("check_risk", check_risk_node)
    builder.add_node("check_format", check_format_node)
    builder.add_node("aggregate", aggregate_node)

    # 3. 连边
    # START → 解析
    builder.add_edge(START, "parse_ticket")

    # 解析 → 5 个审查节点（fan-out）
    builder.add_edge("parse_ticket", "check_safety")
    builder.add_edge("parse_ticket", "check_procedure")
    builder.add_edge("parse_ticket", "check_hazard")
    builder.add_edge("parse_ticket", "check_risk")
    builder.add_edge("parse_ticket", "check_format")

    # 5 个审查节点 → 聚合（fan-in）
    builder.add_edge("check_safety", "aggregate")
    builder.add_edge("check_procedure", "aggregate")
    builder.add_edge("check_hazard", "aggregate")
    builder.add_edge("check_risk", "aggregate")
    builder.add_edge("check_format", "aggregate")

    # 聚合 → END
    builder.add_edge("aggregate", END)

    # 4. 编译
    return builder.compile()


if __name__ == "__main__":
    import asyncio
    from backend.base.logger import configure_logging

    configure_logging()


    async def test():
        graph = build_review_graph()

        # 模拟一份工作票
        ticket = """工作任务：#1机组A汽泵检修
风险等级：较大风险
危险点：高空坠落、机械伤害、高温烫伤
操作步骤：1. 停运A汽泵  2. 隔离电源  3. 挂禁止操作牌  4. 办理检修交代"""

        result = await graph.ainvoke({
            "ticket_text": ticket,
            "ticket_id": "",
            "ticket_task": "",
            "risk_level": "",
            "hazard_analysis": "",
            "procedures": "",
            "safety_review": None,
            "procedure_review": None,
            "hazard_review": None,
            "risk_review": None,
            "format_review": None,
            "report": None,
        })

        report = result["report"]
        print(f"\n{'=' * 50}")
        print(f"工作票: {report.ticket_id}")
        print(f"综合评分: {report.overall_score}/100")
        print(f"是否通过: {'✅' if report.passed else '❌'}")
        print(f"人工复核: {'⚠️ 需要' if report.needs_manual_review else '✅ 不需要'}")
        print(f"综合评语: {report.summary}")
        print(f"\n各维度评分:")
        for name, dim in report.dimensions.items():
            print(f"  {name}: {dim.score}分 {'✅' if dim.passed else '❌'}")
            if dim.issues:
                for issue in dim.issues:
                    print(f"    - {issue}")


    asyncio.run(test())