#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: config.py
项目: 火电两票协同智能审查与设备运维多 Agent 系统
描述: 全项目唯一的配置中心，从 .env.local 读取所有配置项。
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
import sys, os

# 从 config.py 所在位置向上走 3 层到项目根目录
# backend/base/config.py → backend/ → 项目根目录
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(root_path, ".env.local")


class Settings(BaseSettings):
    """配置模型：每个类属性对应 .env.local 里的一项配置。"""

    # ── LLM 提供者切换 ──
    # deepseek: 开发/演示（外网 API）；vllm: 生产（内网私有化 72B，断网运行）
    llm_provider: str = "deepseek"   # "deepseek" | "vllm"

    # ── DeepSeek（开发/演示）──
    deepseek_api_key: str = ""                      # vLLM 模式可不填
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-v4-flash"

    # ── vLLM（生产私有化，OpenAI 兼容接口）──
    vllm_base_url: str = "http://127.0.0.1:8001/v1"   # 内网推理服务地址
    vllm_model: str = "Qwen2.5-72B-Instruct"          # 私有化部署模型
    vllm_api_key: str = "vllm"                        # vLLM 通常不需要真实 key

    # ── Agent 类型 → 模型标识符 路由表（生产/开发共用同一套逻辑模型）──
    # 逻辑模型标识符通过模型网关映射到具体 provider 的模型名
    agent_model_routing: dict[str, str] = {
        "orchestrator": "deepseek-chat",   # 主控整合
        "expert_boiler": "deepseek-chat",  # 锅炉班组专家
        "expert_turbine": "deepseek-chat", # 汽机班组专家
        "expert_electric": "deepseek-chat",# 电气班组专家
        "expert_hotcontrol": "deepseek-chat", # 热控班组专家
        "expert_coal": "deepseek-chat",    # 输煤班组专家
        "ticket_assist": "deepseek-chat",  # 两票辅助审查
        "review": "deepseek-chat",         # 复盘总结
        "rag": "deepseek-chat",            # 规程问答（保留原 RAG 接口）
    }

    # ── 知识库配置 ──
    kb_root: str = "knowledge_base"   # 知识库根目录（相对项目根）
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    vector_db: str = "milvus"         # "milvus" | "faiss"（演示降级）
    embed_model_path: str = "models/embedding/bge-m3"
    rerank_model_path: str = "models/reranker/bge-reranker-large"

    # ── Langfuse 链路追踪（可选，故障事后溯源/审计）──
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://127.0.0.1:3000"

    # ── 应用基础配置 ──
    app_env: str = "local"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    class Config:
        """Pydantic 元配置。"""
        env_file = env_path
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """获取全局唯一的配置对象。"""
    return Settings()


if __name__ == "__main__":
    s = get_settings()
    print(f"LLM Provider:  {s.llm_provider}")
    print(f"DeepSeek Key:  {'***' if s.deepseek_api_key else '(empty)'}")
    print(f"Ollama URL:    {s.ollama_base_url}")
    print(f"Ollama Model:  {s.ollama_model}")
    print(f"Log Level:     {s.log_level}")
    print(f"App Port:      {s.app_port}")
