"""
Langfuse 链路追踪（可选组件）。

核心价值（生产/审计视角）：故障事后溯源——核查某条诊断结论由哪段规程、哪条历史案例支撑，
满足电厂事故复盘审计要求；同时是开发调试与面试演示工具（展示每个 Agent 的调用链）。

启用方式：
1. 启动服务端：docker compose -f docker-compose.langfuse.yml up -d
2. .env.local 配置 LANGFUSE_ENABLED=true / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
3. 工作流调用自动携带 callbacks（见 workflow_graph.run_workflow / tracing 工具）
"""
from __future__ import annotations

from backend.base.config import get_settings
from backend.base.logger import get_logger

logger = get_logger(__name__)


def get_tracing_callbacks() -> list | None:
    """返回 Langfuse 回调（未启用返回 None）。"""
    s = get_settings()
    if not s.langfuse_enabled:
        return None
    try:
        from langfuse.langgraph import LangfuseLanggraphCallbackHandler

        handler = LangfuseLanggraphCallbackHandler(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            host=s.langfuse_host,
        )
        logger.info("tracing.enabled", host=s.langfuse_host)
        return [handler]
    except Exception as e:  # noqa: BLE001
        logger.warning("tracing.disabled", error=str(e))
        return None
