"""
Agent 2：规程知识库 RAG 的 State 定义。
"""
from typing import TypedDict, Optional, List


class RagState(TypedDict):
    """RAG 问答 Agent 的全局状态。"""
    # ── 输入 ──
    question: str                      # 用户提出的问题
    ticket_text: str                   # 关联的工作票原文（可选）

    # ── 检索中间状态 ──
    retrieved_docs: Optional[List[str]]  # 检索到的相关文档片段

    # ── 输出 ──
    answer: str                          # 最终回答
    sources: Optional[List[str]]         # 引用来源
