"""檢索評估:recall@1、recall@3、MRR。

為什麼只評檢索、不評生成
    兩者混在一起量,分數變差時分不出是「撈錯了」還是「模型講錯了」。檢索是可以離線、
    無 API key、可重複量測的那一半,先把它釘住;生成品質另外用 golden_set 那類題庫評。

為什麼建在暫存目錄
    評估不能碰 vector_store/ ——那是使用者的索引,而評估要換模型、換切塊參數反覆跑。
    每次跑都在 tempdir 建一份,跑完就丟。

CLI:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --embedding-model BAAI/bge-m3
    python scripts/evaluate_retrieval.py --embedding-provider openai
    python scripts/evaluate_retrieval.py --chunk-size 1000 --chunk-overlap 50
    python scripts/evaluate_retrieval.py --threshold 0.2 --json out/retrieval.json

`--threshold` 不影響 recall/MRR 的計算,它另外回答一個問題:
「這個門檻會讓多少本來撈得到的題目,對使用者顯示成『找不到相關內容』?」
選門檻應該看這個數字,不是憑感覺。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from config import settings  # noqa: E402
from memory.factory import create_embedder  # noqa: E402
from memory.reranker import create_reranker  # noqa: E402
from memory.vector_database.chroma import Chroma  # noqa: E402
from services.ingest_documents_service.document_loader.loader import DirectoryLoader  # noqa: E402
from services.ingest_documents_service.document_loader.text_splitter import split_chunks  # noqa: E402

GOLDEN_PATH = ROOT / "tests" / "retrieval" / "golden_queries.yaml"


@dataclass
class QueryResult:
    qid: str
    lang: str
    query: str
    relevant: list[str]
    ranked_sources: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def first_hit_rank(self) -> int | None:
        """1-indexed rank of the first relevant document, or None if it never appears."""
        for rank, source in enumerate(self.ranked_sources, start=1):
            if source in self.relevant:
                return rank
        return None

    @property
    def best_relevant_score(self) -> float | None:
        hits = [self.scores[s] for s in self.ranked_sources if s in self.relevant]
        return max(hits) if hits else None

    def recall_at(self, n: int) -> int:
        rank = self.first_hit_rank
        return 1 if rank is not None and rank <= n else 0

    @property
    def reciprocal_rank(self) -> float:
        rank = self.first_hit_rank
        return 1.0 / rank if rank else 0.0


def load_config() -> dict:
    if not GOLDEN_PATH.exists():
        raise SystemExit(f"找不到評估集:{GOLDEN_PATH}")
    with GOLDEN_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def relativise(path: str) -> str:
    """把 loader 寫進 metadata 的絕對路徑轉成 repo 相對路徑,才能跟 GT 比對。"""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def build_index(corpus_paths: list[str], chunk_size: int, chunk_overlap: int, embedder, work_dir: Path):
    documents = []
    for rel in corpus_paths:
        directory = ROOT / rel
        if not directory.is_dir():
            raise SystemExit(f"語料目錄不存在:{directory}")
        documents.extend(DirectoryLoader(path=directory, glob="**/*.md", show_progress=False).load())

    if not documents:
        raise SystemExit("語料是空的,沒有東西可以評估。")

    chunks = split_chunks(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    index = Chroma(
        is_persistent=True,
        persist_directory=str(work_dir),
        embedding=embedder,
        collection_name="retrieval-eval",
    )
    index.from_chunks(chunks)
    return index, len(documents), len(chunks)


def evaluate(index, queries: list[dict], k: int, reranker=None, candidates: int = 20) -> list[QueryResult]:
    results = []
    for item in queries:
        if reranker is None:
            docs_and_scores = index.similarity_search_with_relevance_scores(query=item["query"], k=k)
        else:
            # 二階段:dense 撈寬(candidates),cross-encoder 重排,再取前 k。
            #
            # candidates 是 rerank 的上限 —— dense 沒撈到的文件,重排救不回來。
            # 但「開大一定更好」是錯的:實測把 candidates 從 20 開到全部 87 個 chunk,
            # 英文 recall@3 從 80% 掉到 70%、MRR 從 0.775 掉到 0.742,中文持平。
            # 多給的候選大多是雜訊,reranker 在它不擅長的語言上會把雜訊拉上來。
            # 所以 candidates 要調,而且要用這支腳本量,不能假設越大越好。
            pool = index.similarity_search_with_relevance_scores(query=item["query"], k=candidates)
            docs_and_scores = reranker.rerank(item["query"], [doc for doc, _ in pool], top_k=k)

        # 以「文件」而非「chunk」排名:同一份文件被撈到三個 chunk 仍然只算一個名次,
        # 否則一份長文件塞滿 top-k,recall@3 就失去意義了。
        ranked, scores = [], {}
        for doc, score in docs_and_scores:
            source = relativise(doc.metadata.get("source", ""))
            scores.setdefault(source, score)
            if source not in ranked:
                ranked.append(source)

        results.append(
            QueryResult(
                qid=item["id"],
                lang=item.get("lang", "?"),
                query=item["query"],
                relevant=[Path(p).as_posix() for p in item["relevant"]],
                ranked_sources=ranked,
                scores=scores,
            )
        )
    return results


def summarise(results: list[QueryResult]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "recall@1": sum(r.recall_at(1) for r in results) / n,
        "recall@3": sum(r.recall_at(3) for r in results) / n,
        "mrr": sum(r.reciprocal_rank for r in results) / n,
    }


def pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def report(results: list[QueryResult], threshold: float | None, meta: dict) -> dict:
    line = "=" * 74
    print(line)
    print("檢索評估結果")
    print(line)
    for key, value in meta.items():
        print(f"  {key:<22}{value}")

    overall = summarise(results)
    by_lang = {}
    for lang in sorted({r.lang for r in results}):
        by_lang[lang] = summarise([r for r in results if r.lang == lang])

    print()
    print(f"  {'範圍':<10}{'題數':>5}{'recall@1':>11}{'recall@3':>11}{'MRR':>9}")
    print("  " + "-" * 46)
    print(
        f"  {'全部':<10}{overall['n']:>5}{pct(overall['recall@1']):>11}"
        f"{pct(overall['recall@3']):>11}{overall['mrr']:>9.3f}"
    )
    for lang, m in by_lang.items():
        print(f"  {lang:<10}{m['n']:>5}{pct(m['recall@1']):>11}{pct(m['recall@3']):>11}{m['mrr']:>9.3f}")

    misses = [r for r in results if r.first_hit_rank is None]
    weak = [r for r in results if r.first_hit_rank and r.first_hit_rank > 3]
    if misses or weak:
        print()
        print("  撈不到 / 排名太後面的題目")
        for r in misses:
            print(f"    {r.qid}  完全沒撈到    {r.query}")
        for r in weak:
            print(f"    {r.qid}  第 {r.first_hit_rank} 名      {r.query}")

    filtered = []
    if threshold is not None:
        for r in results:
            best = r.best_relevant_score
            if best is not None and best <= threshold:
                filtered.append((r, best))
        print()
        print(f"  門檻 {threshold} 的影響")
        print(f"    本來撈得到、但會被門檻濾成「找不到相關內容」:{len(filtered)} / {len(results)} 題")
        for r, score in filtered:
            print(f"      {r.qid}  最佳相關分數 {score:.3f}  {r.query}")
        if not filtered:
            print("      無 —— 這個門檻不會誤殺任何一題")

    print(line)
    return {
        "meta": meta,
        "overall": overall,
        "by_language": by_lang,
        "threshold": threshold,
        "filtered_by_threshold": [r.qid for r, _ in filtered],
        "queries": [
            {
                "id": r.qid,
                "lang": r.lang,
                "query": r.query,
                "relevant": r.relevant,
                "first_hit_rank": r.first_hit_rank,
                "best_relevant_score": r.best_relevant_score,
                "top_sources": r.ranked_sources[:5],
            }
            for r in results
        ],
    }


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate retrieval quality against the golden query set.")
    p.add_argument("--embedding-provider", default=None, help="覆寫 EMBEDDING_PROVIDER")
    p.add_argument("--embedding-model", default=None, help="覆寫 embedding 模型名稱")
    p.add_argument("--chunk-size", type=int, default=None, help="覆寫評估集裡的 chunk_size")
    p.add_argument("--chunk-overlap", type=int, default=None, help="覆寫評估集裡的 chunk_overlap")
    p.add_argument("--k", type=int, default=10, help="每題撈幾筆再排名。預設 10")
    p.add_argument("--threshold", type=float, default=None, help="額外報告這個相關度門檻會濾掉多少本來撈得到的題目")
    p.add_argument("--rerank", action="store_true", help="開啟第二階段 cross-encoder 重排")
    p.add_argument("--rerank-model", default=None, help="覆寫 reranker 模型,指定即視為開啟 --rerank")
    p.add_argument("--candidates", type=int, default=None, help="第一階段撈幾筆給 reranker。預設取 RERANK_CANDIDATES")
    p.add_argument("--json", default=None, help="把結果寫成 JSON,方便比較兩次跑的差異")
    return p.parse_args()


def main() -> None:
    args = get_args()
    config = load_config()
    corpus = config.get("corpus", {})
    queries = config.get("queries", [])
    if not queries:
        raise SystemExit("評估集裡沒有 queries。")

    chunk_size = args.chunk_size or corpus.get("chunk_size", 400)
    chunk_overlap = args.chunk_overlap or corpus.get("chunk_overlap", 80)

    embedder = create_embedder(provider=args.embedding_provider, model_name=args.embedding_model)

    rerank_on = args.rerank or args.rerank_model is not None
    reranker = create_reranker(model_name=args.rerank_model, enabled=rerank_on)
    candidates = args.candidates or settings.RERANK_CANDIDATES

    work_dir = Path(tempfile.mkdtemp(prefix="retrieval-eval-"))
    try:
        index, n_docs, n_chunks = build_index(corpus.get("paths", []), chunk_size, chunk_overlap, embedder, work_dir)
        results = evaluate(index, queries, args.k, reranker=reranker, candidates=candidates)
        payload = report(
            results,
            args.threshold,
            {
                "embedder": type(embedder).__name__,
                "model": args.embedding_model or "(依 .env 設定)",
                "provider": args.embedding_provider or "(依 .env 設定)",
                "chunk_size / overlap": f"{chunk_size} / {chunk_overlap}",
                "語料": f"{n_docs} 份文件 -> {n_chunks} 個 chunk",
                "k": args.k,
                "rerank": (f"{reranker.model_name} (candidates={candidates})" if reranker else "關閉"),
            },
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  已寫出 {out}")


if __name__ == "__main__":
    main()
