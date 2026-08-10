"""
火电两票协同智能审查与设备运维多 Agent 系统
后端入口：FastAPI 应用
"""
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from backend.base.logger import configure_logging, get_logger
from backend.base.config import get_settings
from backend.api.review import router as review_router
from backend.api.qa import router as qa_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    configure_logging()
    logger = get_logger(__name__)
    s = get_settings()
    logger.info("app.starting", provider=s.llm_provider, port=s.app_port)
    yield
    logger.info("app.stopping")


app = FastAPI(
    title="火电两票协同智能审查与设备运维多 Agent 系统",
    description="基于 LangGraph + FastAPI 的事件驱动多 Agent 系统：设备状态感知、跨班组故障会诊、两票智能辅助审查、处置闭环与知识沉淀",
    version="2.0.0",
    lifespan=lifespan,
)

# 注册路由
app.include_router(review_router)
app.include_router(qa_router)


@app.get("/")
async def root():
    return {"service": "火电两票协同智能审查与设备运维多 Agent 系统", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "ok", "provider": get_settings().llm_provider}
