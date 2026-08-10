"""
工作票审查 API 接口。
"""
from fastapi import APIRouter
from pydantic import BaseModel
from backend.base.logger import get_logger
from backend.agents.review.graph import build_review_graph

logger = get_logger(__name__)
router = APIRouter(prefix="/api/review", tags=["工作票审查"])
graph = build_review_graph()


class ReviewRequest(BaseModel):
    ticket_text: str


class ReviewResponse(BaseModel):
    ticket_id: str
    overall_score: int
    passed: bool
    needs_manual_review: bool
    summary: str
    dimensions: dict
    message: str


@router.post("/", response_model=ReviewResponse)
async def review_ticket(req: ReviewRequest):
    """审查工作票内容。"""
    logger.info("api.review_requested", text_len=len(req.ticket_text))

    result = await graph.ainvoke({
        "ticket_text": req.ticket_text,
        "ticket_id": "", "ticket_task": "", "risk_level": "",
        "hazard_analysis": "", "procedures": "",
        "safety_review": None, "procedure_review": None,
        "hazard_review": None, "risk_review": None,
        "format_review": None, "report": None,
    })

    report = result["report"]

    # 维度信息转可序列化格式
    dims = {}
    for key, dim in report.dimensions.items():
        dims[key] = {
            "score": dim.score,
            "passed": dim.passed,
            "issues": dim.issues,
            "suggestions": dim.suggestions,
        }

    return ReviewResponse(
        ticket_id=report.ticket_id,
        overall_score=report.overall_score,
        passed=report.passed,
        needs_manual_review=report.needs_manual_review,
        summary=report.summary,
        dimensions=dims,
        message="审查完成",
    )
