"""
知识库检索工具：对知识库进行混合检索（关键词 + 语义）。
"""

import os, json
from backend.base.logger import get_logger

logger = get_logger(__name__)

# 知识库索引路径
KB_INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "knowledge_base", "index")
KB_INDEX_FILE = os.path.join(KB_INDEX_DIR, "chunks.json")


def _load_chunks() -> list[dict]:
    """加载知识库文档块。"""
    if not os.path.exists(KB_INDEX_FILE):
        logger.warning("kb.index_not_found", path=KB_INDEX_FILE)
        return []
    with open(KB_INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _keyword_search(chunks: list[dict], query: str, top_k: int = 3) -> list[dict]:
    """简单的关键词检索（不依赖向量库，内网可用）。"""
    query_words = set(query.lower().split())
    scored = []

    for chunk in chunks:
        text = chunk["text"].lower()
        # 计算关键词命中率
        hits = sum(1 for w in query_words if w in text)
        if hits > 0:
            scored.append((hits / max(len(query_words), 1), chunk))

    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:top_k]]


def search_knowledge_base(query: str, top_k: int = 3) -> list[str]:
    """检索知识库，返回最相关的文档片段列表。"""
    chunks = _load_chunks()
    if not chunks:
        return []

    results = _keyword_search(chunks, query, top_k)
    texts = [r["text"] for r in results]

    logger.info("kb.search_complete", query=query[:50], results=len(texts))
    return texts
