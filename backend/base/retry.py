"""
三层兜底机制：自动重试 → Agent 级降级 → 系统级兜底
"""
import asyncio
from functools import wraps
from typing import Callable, Any, Optional

from backend.base.logger import get_logger
from backend.base.exceptions import (
    LLMAPIError,
    DatabaseError,
    InvalidInputError,
    AuthenticationError,
)

logger = get_logger(__name__)

# ── 异常分类 ──
RETRYABLE_ERRORS = (
    LLMAPIError,
    DatabaseError,
    TimeoutError,
    ConnectionError,
)

NON_RETRYABLE_ERRORS = (
    InvalidInputError,
    AuthenticationError,
)

MAX_RETRIES = 3        # 最多重试几次？（参考 EduAgent）
RETRY_DELAYS = [1.0,3.0,5.0]   # 每次重试前等多久？（参考 EduAgent）
TIMEOUT_PER_ATTEMPT = 30  # 单次调用超时多少秒？

def with_retry(agent_type: str = ""):
    """三层兜底装饰器工厂。"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_error = None

            # ── 第一层：重试 ──
            for attempt in range(MAX_RETRIES + 1):
                try:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=TIMEOUT_PER_ATTEMPT,
                    )
                    if attempt > 0:
                        logger.info("retry.succeeded", agent_type=agent_type, attempt=attempt + 1)
                    return result

                except NON_RETRYABLE_ERRORS:
                    raise  # 不可重试，直接抛

                except Exception as e:
                    last_error = e
                    if attempt < MAX_RETRIES:
                        delay = RETRY_DELAYS[attempt]
                        logger.warning("retry.attempt_failed",
                            agent_type=agent_type, attempt=attempt + 1, delay=delay)
                        await asyncio.sleep(delay)
                    else:
                        logger.error("retry.all_failed", agent_type=agent_type)

            # ── 第二层：降级 ──
            try:
                fallback = await AgentFallbackHandler.handle(
                    agent_type=agent_type, original_error=last_error
                )
                return fallback
            except Exception:
                logger.error("retry.fallback_failed", agent_type=agent_type)

            # ── 第三层：系统兜底 ──
            return _system_fallback(agent_type)

        return wrapper
    return decorator

class AgentFallbackHandler:
    @classmethod
    async def handle(cls, agent_type: str, original_error: Exception) -> Any:
        fallback_map = {
            "ticket_review": cls._ticket_review_fallback,
            # 以后加 Agent 时在这里加一行
        }
        handler = fallback_map.get(agent_type)
        if handler:
            return await handler()
        raise original_error

    @classmethod
    async def _ticket_review_fallback(cls) -> dict:
        """操作票审查降级：标记需人工复核。"""
        return {
            "fallback_used": True,
            "needs_manual_review": True,
            "content": "智能审查服务暂不可用，已标记需人工复核。"
        }

def _system_fallback(agent_type: str) -> dict:
    """第三层：系统级兜底。永远不会失败。"""
    return {
        "fallback_used": True,
        "system_fallback": True,
        "content": "服务暂不可用，请稍后再试。"
    }





