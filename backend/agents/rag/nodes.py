"""
Agent 2：规程知识库 RAG 的节点函数。
"""
from langchain_core.messages import SystemMessage, HumanMessage
from backend.base.llm_factory import get_llm
from backend.base.logger import get_logger
from backend.agents.rag.state import RagState
from backend.agents.rag.retriever import search_knowledge_base

logger = get_logger(__name__)

RAG_SYSTEM_PROMPT = """你是一位火力发电厂规程专家，根据提供的规程文档内容回答问题。

规则：
1. 只能基于提供的文档内容回答，不要编造
2. 如果文档内容不足以回答，明确说"规程中未找到相关内容"
3. 回答要具体，引用文档中的条款编号
4. 回答要简洁，直接给出结论和依据"""


async def retrieve_node(state: RagState) -> dict:
    """检索节点：从知识库中检索相关文档片段。"""
    question = state["question"]

    # 如果有 ticket_text，把票面内容也作为检索上下文
    query = question
    if state.get("ticket_text"):
        query = f"{question}\n关联工作票：{state['ticket_text'][:200]}"

    docs = search_knowledge_base(query, top_k=3)
    logger.info("rag.retrieved", query=question[:50], count=len(docs))

    return {"retrieved_docs": docs}


async def generate_node(state: RagState) -> dict:
    """生成节点：基于检索结果生成回答。"""
    docs = state.get("retrieved_docs") or []
    question = state["question"]

    if not docs:
        return {
            "answer": "规程库中未找到相关内容，请补充知识库或咨询专工。",
            "sources": [],
        }

    context = "\n\n".join(docs)
    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=f"规程文档内容：\n{context}\n\n问题：{question}"),
    ]

    llm = get_llm("qa", temperature=0.1)
    response = await llm.ainvoke(messages)

    return {
        "answer": response.content,
        "sources": docs[:3],
    }
