"""
混合检索器：jieba BM25（稀疏）+ BGE-M3（稠密向量）双路检索 + 可选 BGE-Reranker 精排。

- 按班组分库检索（team_index 预构建）
- 向量索引首次构建后缓存到 knowledge_base/index/vectors.npy（避免重复编码）
- Reranker 默认关闭（内存/算力受限环境可选加载，与 dianchang 一致）
"""
from __future__ import annotations

import json
import os
import threading

import numpy as np

from backend.agents.experts.knowledge import KbDoc, TEAMS, build_team_index, load_kb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
VECTOR_CACHE = os.path.join(PROJECT_ROOT, "knowledge_base", "index", "vectors.npy")
META_CACHE = os.path.join(PROJECT_ROOT, "knowledge_base", "index", "vectors_meta.json")


class HybridRetriever:
    """混合检索器（BM25 + 向量 + 可选 Reranker）。"""

    def __init__(
        self,
        team_index: dict[str, list[KbDoc]] | None = None,
        enable_rerank: bool = False,
        embed_model_name: str = "models/embedding/bge-m3",
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ):
        self.team_index = team_index or build_team_index(load_kb())
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.enable_rerank = enable_rerank
        self._embedder = None
        self._reranker = None
        self._lock = threading.Lock()
        self._trace_cache: dict[tuple, list[dict]] = {}   # (trace_id, team, query, top_k) -> 结果（幂等复用）

        # 惰性构建 BM25 索引
        self._bm25_index: dict[str, object] = {}
        self._tokenized: dict[str, list[list[str]]] = {}
        self._build_bm25()

        # 惰性加载向量缓存/模型
        self._vectors: dict[str, np.ndarray] | None = None
        self._vector_docs: dict[str, list[KbDoc]] | None = None
        self._load_vectors_or_lazy()

    # ── BM25 ──
    def _build_bm25(self):
        import jieba

        for team in TEAMS:
            self._build_bm25_team(team)

    def _build_bm25_team(self, team: str):
        """重建单个班组的 BM25 索引（jieba 分词，千条级 <1s，增量成本可忽略）。"""
        from rank_bm25 import BM25Okapi
        import jieba

        docs = self.team_index[team]
        tokenized = [list(jieba.cut(d.text)) for d in docs]
        self._tokenized[team] = tokenized
        self._bm25_index[team] = BM25Okapi(tokenized)

    def _bm25_search(self, team: str, query: str, top_k: int) -> list[tuple[int, float]]:
        import jieba
        q_tokens = list(jieba.cut(query))
        scores = self._bm25_index[team].get_scores(q_tokens)
        # 返回 (doc_idx, score) 按分降序
        order = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    # ── 向量 ──
    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            model_path = os.path.join(PROJECT_ROOT, self._embed_model_path())
            self._embedder = SentenceTransformer(model_path)
        return self._embedder

    def _embed_model_path(self) -> str:
        return "models/embedding/bge-m3"

    def _load_vectors_or_lazy(self):
        """尝试加载向量缓存；不存在则标记惰性构建。"""
        if os.path.exists(VECTOR_CACHE) and os.path.exists(META_CACHE):
            try:
                arr = np.load(VECTOR_CACHE, allow_pickle=False)
                with open(META_CACHE, encoding="utf-8") as f:
                    meta = json.load(f)
                # meta: {"team": [doc_ids...]}
                self._vectors = {}
                self._vector_docs = {}
                offset = 0
                for team, ids in meta.items():
                    n = len(ids)
                    if team in self.team_index and n == len(self.team_index[team]):
                        self._vectors[team] = arr[offset:offset + n]
                        self._vector_docs[team] = self.team_index[team]
                    offset += n
                print(f"[retriever] 向量缓存加载: {len(self._vectors)} 个班组")
                return
            except Exception as e:  # noqa: BLE001
                print(f"[retriever] 缓存加载失败({e})，将重建")
        self._vectors = None
        self._vector_docs = None

    def _ensure_vectors(self, team: str):
        """为指定班组构建/加载向量（首次编码，缓存落盘）。"""
        if self._vectors is not None and team in self._vectors:
            return
        embedder = self._get_embedder()
        with self._lock:
            if self._vectors is None:
                self._vectors = {}
                self._vector_docs = {}
            if team not in self._vectors:
                docs = self.team_index[team]
                embs = embedder.encode([d.text for d in docs], show_progress_bar=True, convert_to_numpy=True)
                self._vectors[team] = embs.astype(np.float32)
                self._vector_docs[team] = docs
                self._save_cache()

    def _save_cache(self):
        """把已构建班组向量持久化（增量合并）。"""
        try:
            meta = {}
            arrs = []
            for team in TEAMS:
                if team in self._vectors:
                    meta[team] = [d.doc_id for d in self._vector_docs[team]]
                    arrs.append(self._vectors[team])
            if arrs:
                np.save(VECTOR_CACHE, np.concatenate(arrs, axis=0))
                with open(META_CACHE, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False)
                print(f"[retriever] 向量缓存已保存: {sum(len(v) for v in arrs)} 条")
        except Exception as e:  # noqa: BLE001
            print(f"[retriever] 缓存保存失败: {e}")

    def _vector_search(self, team: str, query: str, top_k: int) -> list[tuple[int, float]]:
        self._ensure_vectors(team)
        embedder = self._get_embedder()
        q_vec = embedder.encode([query], convert_to_numpy=True)[0]
        return self._vector_search_encoded(team, q_vec, top_k)

    def _vector_search_encoded(self, team: str, q_vec, top_k: int) -> list[tuple[int, float]]:
        """用预编码查询向量检索（评测批量场景避免重复编码）。"""
        self._ensure_vectors(team)
        embs = self._vectors[team]
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        sims = (embs @ q_vec) / (norms[:, 0] * (np.linalg.norm(q_vec) + 1e-9) + 1e-9)
        order = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in order if sims[i] > 0]

    def append_docs(self, team: str, docs: list[KbDoc]) -> int:
        """增量更新：编码新文档追加到该班组向量缓存 + 重建 BM25（不删除/重建已有）。
        幂等：doc_id 已在缓存中则跳过。返回实际追加数。"""
        if team not in self.team_index:
            raise ValueError(f"未知班组: {team}")
        self._ensure_vectors(team)

        existing_ids = {d.doc_id for d in self._vector_docs[team]}
        new_docs = [d for d in docs if d.doc_id not in existing_ids]
        if not new_docs:
            return 0

        embedder = self._get_embedder()
        new_embs = embedder.encode([d.text for d in new_docs], convert_to_numpy=True).astype(np.float32)
        with self._lock:
            self._vectors[team] = np.vstack([self._vectors[team], new_embs])
            self._vector_docs[team] = self._vector_docs[team] + new_docs
            self._save_cache()
        # BM25 增量：重建该班组（快）
        self.team_index[team] = self.team_index[team] + new_docs
        self._build_bm25_team(team)
        print(f"[retriever] 增量追加 {len(new_docs)} 条到 [{team}]，缓存已更新")
        return len(new_docs)

    # ── Reranker（可选）──
    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            path = os.path.join(PROJECT_ROOT, "models/reranker/bge-reranker-large")
            self._reranker = CrossEncoder(path)
        return self._reranker

    # ── 融合检索 ──
    def search(self, team: str, query: str, top_k: int = 5, trace_id: str | None = None) -> list[dict]:
        """
        混合检索：BM25 + 向量 加权融合 → 可选 Reranker 精排。
        幂等：携带 trace_id 时先查重，同一 TraceID+查询 直接复用历史结果，不重复执行检索逻辑。
        返回 [{doc, score, source, doc_id}]
        """
        if trace_id:
            cache_key = (trace_id, team, query, top_k)
            if cache_key in self._trace_cache:
                return self._trace_cache[cache_key]

        docs = self.team_index[team]
        bm25_hits = self._bm25_search(team, query, top_k * 3)
        vec_hits = self._vector_search(team, query, top_k * 3)

        # 分数归一化（min-max 到 0~1）
        def _norm(hits):
            if not hits:
                return {}
            max_s = max(s for _, s in hits)
            min_s = min(s for _, s in hits)
            span = max_s - min_s or 1.0
            return {i: (s - min_s) / span for i, s in hits}

        bm25_n = _norm(bm25_hits)
        vec_n = _norm(vec_hits)
        fused: dict[int, float] = {}
        for i, s in bm25_n.items():
            fused[i] = fused.get(i, 0.0) + self.bm25_weight * s
        for i, s in vec_n.items():
            fused[i] = fused.get(i, 0.0) + self.vector_weight * s

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        results = [
            {
                "doc": docs[i],
                "score": round(s, 4),
                "doc_id": docs[i].doc_id,
                "source": docs[i].source,
            }
            for i, s in ranked
        ]

        # 可选 Reranker 精排
        if self.enable_rerank and results:
            pairs = [(query, r["doc"].text[:512]) for r in results]
            scores = self._get_reranker().predict(pairs)
            for r, s in zip(results, scores):
                r["score"] = round(float(s), 4)
            results.sort(key=lambda r: -r["score"])

        if trace_id:
            self._trace_cache[cache_key] = results
        return results


_retriever_singleton: HybridRetriever | None = None
_retriever_lock = threading.Lock()


def get_retriever(enable_rerank: bool = False) -> HybridRetriever:
    """全局单例检索器（班组索引/BM25 只构建一次）。"""
    global _retriever_singleton
    with _retriever_lock:
        if _retriever_singleton is None:
            _retriever_singleton = HybridRetriever(enable_rerank=enable_rerank)
        return _retriever_singleton


if __name__ == "__main__":
    import time
    t0 = time.time()
    r = get_retriever()
    print(f"检索器初始化: {time.time() - t0:.1f}s")
    for q in ["磨煤机轴承温度高", "汽轮机轴瓦磨损"]:
        t1 = time.time()
        team = "锅炉" if "磨煤机" in q else "汽机"
        res = r.search(team, q, top_k=3)
        print(f"\n[{team}班] 查询: {q} ({time.time() - t1:.2f}s)")
        for item in res:
            print(f"  {item['score']:.3f} | {item['doc_id']} | {item['doc'].text[:60]}")
