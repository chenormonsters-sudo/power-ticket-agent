"""
LLM Factory：统一的大模型工厂（模型网关）。
所有 Agent 必须通过此模块获取模型，禁止直接调用 init_chat_model。
双模式：deepseek（开发/演示 API）、vllm（生产私有化 72B）。
"""
from typing import Type, Any
from pydantic import BaseModel
import httpx
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from backend.base.config import get_settings
from backend.base.logger import get_logger

logger = get_logger(__name__)

# ── 自定义 httpx 客户端（绕过代理）──
# 为什么要这个？Windows 系统代理可能干扰 DeepSeek 直连。
_HTTP_ASYNC_CLIENT = httpx.AsyncClient(
    trust_env=False,
    timeout=httpx.Timeout(120.0, connect=15.0),
)
_HTTP_SYNC_CLIENT = httpx.Client(
    trust_env=False,
    timeout=httpx.Timeout(120.0, connect=15.0),
)

# ── Agent 类型 → 逻辑模型标识符 的路由表（config.agent_model_routing 为准）──
_AGENT_MODEL_ROUTING: dict[str, str] = {}

# ── 逻辑模型标识符 → DeepSeek API 实际模型名（仅 deepseek 模式使用）──
_MODEL_ID_MAP: dict[str, str] = {
    "deepseek-chat": "deepseek-v4-flash",
}


class LLMFactory:
    """大模型工厂（统一获取模型的唯一入口）。"""

    _instances: dict[str, BaseChatModel] = {}  # 模型实例缓存

    @classmethod
    def get_llm(cls, agent_type: str, temperature: float = 0, streaming: bool = False) -> BaseChatModel:
        """按 Agent 类型获取模型实例（带缓存）。"""
        # ① 查路由表（以 config 为准，回退模块级表）
        settings = get_settings()
        routing = settings.agent_model_routing or _AGENT_MODEL_ROUTING
        if agent_type in routing:
            model_key = routing[agent_type]
        elif agent_type in _AGENT_MODEL_ROUTING:
            model_key = _AGENT_MODEL_ROUTING[agent_type]
        else:
            raise ValueError(f"未知 agent_type: '{agent_type}'")
        cache_key = f"{model_key}_{temperature}_{streaming}"

        # ② 缓存命中 → 直接返回
        if cache_key in cls._instances:
            return cls._instances[cache_key]

        # ③ 缓存未命中 → 创建新实例

        # 关键：根据 llm_provider 切换不同的配置（模型网关双模式）
        if settings.llm_provider == "vllm":
            # vLLM 私有化模式（生产，断网内网运行）：OpenAI 兼容接口
            kwargs = {
                "model": settings.vllm_model,
                "model_provider": "openai",
                "temperature": temperature,
                "base_url": settings.vllm_base_url,
                "api_key": settings.vllm_api_key,
                "streaming": streaming,
            }
        else:
            # DeepSeek API 模式（开发/演示）：需要 api_key
            actual_model = _MODEL_ID_MAP.get(model_key, model_key)
            kwargs = {
                "model": actual_model,
                "model_kwargs": {
                    "extra_body": {"thinking": {"type": "disabled"}}
                },
                "model_provider": "openai",
                "temperature": temperature,
                "api_key": settings.deepseek_api_key,
                "base_url": settings.deepseek_base_url,
                "streaming": streaming,
            }

        # 共享的 httpx 客户端
        kwargs["http_async_client"] = _HTTP_ASYNC_CLIENT
        kwargs["http_client"] = _HTTP_SYNC_CLIENT

        llm = init_chat_model(**kwargs)
        cls._instances[cache_key] = llm
        logger.info("llm_factory.initialized", agent_type=agent_type, model=model_key)
        return llm

    @classmethod
    def get_structured_llm(cls, agent_type: str, output_schema: Type[BaseModel], temperature: float = 0) -> Runnable:
        """获取绑定了结构化输出 Schema 的模型。"""
        llm = cls.get_llm(agent_type, temperature=temperature)
        return llm.with_structured_output(output_schema, method="function_calling")


# ── 模块级便捷函数 ──
def get_llm(agent_type: str, temperature: float = 0, streaming: bool = False) -> BaseChatModel:
    return LLMFactory.get_llm(agent_type, temperature=temperature, streaming=streaming)


def get_structured_llm(agent_type: str, output_schema: Type[BaseModel]) -> Runnable:
    return LLMFactory.get_structured_llm(agent_type, output_schema)


if __name__ == "__main__":
    import asyncio, sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from langchain_core.messages import HumanMessage
    from backend.base.logger import configure_logging
    configure_logging()

    async def test():
        llm = get_llm("ticket_review")
        resp = await llm.ainvoke([HumanMessage(content="用一句话介绍你自己")])
        print(f"回复: {resp.content}")

    asyncio.run(test())