"""
规程知识库 RAG 问答 API 接口。
"""
from fastapi import APIRouter
from pydantic import BaseModel
from backend.base.logger import get_logger
from backend.agents.rag.graph import build_rag_graph

logger = get_logger(__name__)
router = APIRouter(prefix="/api/qa", tags=["规程问答"])
graph = build_rag_graph()


class QARequest(BaseModel):
    question: str
    ticket_text: str = ""


class QAResponse(BaseModel):
    question: str
    answer: str
    sources: list


@router.post("/", response_model=QAResponse)
async def ask_question(req: QARequest):
    """向规程知识库提问。"""
    logger.info("api.qa_requested", question=req.question[:50])

    result = await graph.ainvoke({
        "question": req.question,
        "ticket_text": req.ticket_text,
        "retrieved_docs": None,
        "answer": "",
        "sources": None,
    })

    return QAResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result.get("sources") or [],
    )
