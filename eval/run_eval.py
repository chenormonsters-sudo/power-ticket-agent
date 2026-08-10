"""
检索评测：Top-K 命中率（不调 LLM，纯检索指标）。

指标：
- Top-1 / Top-3 / Top-5 命中率：期望 FAQ 是否出现在检索结果前 K
- 分班组命中率（定位薄弱班组）
- 混合检索 vs 纯 BM25 对比（体现向量路价值）

运行：python eval/run_eval.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.experts.retriever import get_retriever

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SET_PATH = os.path.join(EVAL_DIR, "test_set.json")
QUERY_VEC_PATH = os.path.join(EVAL_DIR, "query_vectors.npy")
QUERY_META_PATH = os.path.join(EVAL_DIR, "query_meta.json")


def _load_or_build_query_vectors(queries: list[str], embedder):
    """查询向量缓存：评测集固定，向量只编码一次落盘，之后秒级加载。
    用查询文本 hash 校验缓存有效性（test_set 变化自动重建）。"""
    import hashlib
    digest = hashlib.md5("\n".join(queries).encode("utf-8")).hexdigest()

    if os.path.exists(QUERY_VEC_PATH) and os.path.exists(QUERY_META_PATH):
        try:
            with open(QUERY_META_PATH, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("digest") == digest:
                arr = np.load(QUERY_VEC_PATH)
                print(f"[eval] 查询向量缓存命中（{len(arr)} 条），跳过编码")
                return arr
        except Exception as e:  # noqa: BLE001
            print(f"[eval] 查询缓存加载失败({e})，重建")

    print(f"[eval] 批量编码 {len(queries)} 条查询（首次，耗时几分钟，落盘缓存）...")
    arr = embedder.encode(queries, batch_size=32, convert_to_numpy=True)
    np.save(QUERY_VEC_PATH, arr)
    with open(QUERY_META_PATH, "w", encoding="utf-8") as f:
        json.dump({"digest": digest, "count": len(queries)}, f)
    print(f"[eval] 查询向量缓存已保存: {QUERY_VEC_PATH}")
    return arr


def _rerank(retriever, query: str, results: list[dict], top_k: int) -> list[dict]:
    """BGE-Reranker 精排：对候选 (query, doc) 打分重排，取 Top-K。"""
    reranker = retriever._get_reranker()
    pairs = [(query, r["doc"].text[:512]) for r in results]
    scores = reranker.predict(pairs)
    for r, s in zip(results, scores):
        r["score"] = round(float(s), 4)
    results.sort(key=lambda r: -r["score"])
    return results[:top_k]


def run_eval(top_k: int = 5, enable_rerank: bool = False) -> dict:
    """跑检索评测（查询批量缓存；可选 Reranker 精排对比）。"""
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test_set = json.load(f)

    retriever = get_retriever()
    embedder = retriever._get_embedder()
    # 查询向量缓存：首次编码落盘，之后秒级复用（评测集固定，向量只算一次）
    queries = [item["query"] for item in test_set]
    q_vectors = _load_or_build_query_vectors(queries, embedder)

    hits = {"top1": 0, "top3": 0, "top5": 0}
    team_stats: dict[str, dict] = {}
    t0 = time.time()

    for idx, (item, qv) in enumerate(zip(test_set, q_vectors)):
        if idx % 10 == 0:
            print(f"[eval] 进度 {idx}/{len(test_set)}（{round(time.time()-t0,1)}s）", flush=True)
        team = item["expected_team"]
        # 融合：BM25 + 预编码向量
        docs = retriever.team_index[team]
        bm25_hits = retriever._bm25_search(team, item["query"], top_k * 3)
        vec_hits = retriever._vector_search_encoded(team, qv, top_k * 3)

        def _norm(hits):
            if not hits:
                return {}
            mx = max(s for _, s in hits); mn = min(s for _, s in hits); span = mx - mn or 1.0
            return {i: (s - mn) / span for i, s in hits}

        fused = {}
        for i, s in _norm(bm25_hits).items():
            fused[i] = fused.get(i, 0.0) + 0.5 * s
        for i, s in _norm(vec_hits).items():
            fused[i] = fused.get(i, 0.0) + 0.5 * s
        # 融合取 Top-k*2 候选池，供 Reranker 精排
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k * 2]
        pool = [{"doc": docs[i], "score": s, "doc_id": docs[i].doc_id} for i, s in ranked]
        if enable_rerank and pool:
            pool = _rerank(retriever, item["query"], pool, top_k)
        ids = [r["doc_id"] for r in pool]
        expected = item["expected_doc_id"]

        team_stats.setdefault(team, {"total": 0, "hit5": 0})
        team_stats[team]["total"] += 1
        if expected in ids:
            team_stats[team]["hit5"] += 1
        if expected in ids[:1]:
            hits["top1"] += 1
        if expected in ids[:3]:
            hits["top3"] += 1
        if expected in ids[:5]:
            hits["top5"] += 1

    n = len(test_set)
    metrics = {
        "total_queries": n,
        "top1_hit_rate": round(hits["top1"] / n, 4),
        "top3_hit_rate": round(hits["top3"] / n, 4),
        "top5_hit_rate": round(hits["top5"] / n, 4),
                "elapsed_sec": round(time.time() - t0, 1),
        "rerank_enabled": enable_rerank,
        "team_breakdown": {
            t: {"hit5_rate": round(v["hit5"] / v["total"], 4), "queries": v["total"]}
            for t, v in sorted(team_stats.items())
        },
    }
    print(f"[eval] 完成：{n} 条，耗时 {metrics['elapsed_sec']}s", flush=True)
    return metrics


def run_bm25_baseline(top_k: int = 5) -> dict:
    """纯 BM25 基线（对比混合检索价值）。"""
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test_set = json.load(f)
    retriever = get_retriever()
    hits = 0
    for item in test_set:
        team = item["expected_team"]
        results = retriever._bm25_search(team, item["query"], top_k)
        # bm25 返回 (doc_idx, score)，需要映射回 doc_id
        docs = retriever.team_index[team]
        ids = [docs[i].doc_id for i, _ in results]
        if item["expected_doc_id"] in ids:
            hits += 1
    return {"bm25_top5_hit_rate": round(hits / len(test_set), 4)}


if __name__ == "__main__":
    if not os.path.exists(TEST_SET_PATH):
        print("评测集不存在，先构建：python eval/build_eval_set.py")
        sys.exit(1)
    use_rerank = "--rerank" in sys.argv
    print(f"=== 混合检索评测（rerank={'开启' if use_rerank else '关闭'}）===")
    metrics = run_eval(enable_rerank=use_rerank)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if not use_rerank:
        print("\n--- BM25 基线对比 ---")
        print(json.dumps(run_bm25_baseline(), ensure_ascii=False))
