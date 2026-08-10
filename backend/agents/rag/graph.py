"""
Agent 2：规程知识库 RAG 的 LangGraph 图装配。
"""
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from langgraph.graph import StateGraph, START, END
from backend.agents.rag.state import RagState
from backend.agents.rag.nodes import retrieve_node, generate_node


def build_rag_graph():
    """构建 RAG 问答 Agent 的 LangGraph。"""
    builder = StateGraph(RagState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


if __name__ == "__main__":
    import asyncio
    from backend.base.logger import configure_logging
    configure_logging()

    async def test():
        graph = build_rag_graph()
        result = await graph.ainvoke({
            "question": "汽泵检修需要哪些安全措施？",
            "ticket_text": "",
            "retrieved_docs": None,
            "answer": "",
            "sources": None,
        })
        print(f"\n问题: {result['question']}")
        print(f"回答: {result['answer']}")
        if result.get("sources"):
            print(f"\n来源: {len(result['sources'])} 篇")

    asyncio.run(test())
