"""Сравнение lexical/hybrid на фиксированных RU/EN вопросах без вызовов LLM."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def evaluate(data, modes=("lexical", "hybrid")):
    from app.core.knowledge_base import _lexical_retrieve, build_retrieval_query, chunk_text
    from app.core.semantic_retrieval import hybrid_retrieve, MODEL
    chunks = [{"filename": document["filename"], "content": part}
              for document in data["documents"] for part in chunk_text(document["content"])]
    report = {"dataset_sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
              "embedding_model": MODEL, "threshold": 0.6, "margin": 0.05,
              "runtime": {"python": platform.python_version(), "numpy": version("numpy"),
                          "scikit-learn": version("scikit-learn")}, "results": {}}
    for mode in modes:
        rows = []
        for case in data["cases"]:
            started = time.monotonic()
            query = build_retrieval_query(case["q"], case.get("history"))
            found = _lexical_retrieve(query, chunks)
            if mode == "hybrid":
                # Прямой вызов: отсутствие модели должно провалить benchmark, а не скрыто вернуть lexical.
                found = hybrid_retrieve(query, chunks, found)
            files = [c["filename"] for c in found]
            expected = case["expected"]
            rows.append({**case, "retrieved": files, "top1": bool(files and files[0] == expected) if expected else not files,
                         "top3": expected in files if expected else not files,
                         "latency_ms": round((time.monotonic() - started) * 1000, 3)})
        groups = {}
        for split in ("dev", "heldout"):
            subset = [r for r in rows if r["split"] == split]
            answerable = [r for r in subset if r["expected"]]
            unknown = [r for r in subset if not r["expected"]]
            groups[split] = {"answerable": len(answerable), "top1_hits": sum(r["top1"] for r in answerable),
                             "top3_hits": sum(r["top3"] for r in answerable), "unanswerable": len(unknown),
                             "correct_abstentions": sum(r["top3"] for r in unknown),
                             "median_latency_ms": statistics.median(r["latency_ms"] for r in subset)}
        report["results"][mode] = {"summary": groups, "cases": rows}
    if "hybrid" in modes:
        from app.core.semantic_retrieval import embedding_model
        report["runtime"]["fastembed"] = version("fastembed")
        directory = Path(embedding_model().model._model_dir)
        assets = {}
        for path in sorted(directory.rglob("*.onnx")):
            with path.open("rb") as handle:
                assets[str(path.relative_to(directory))] = hashlib.file_digest(handle, "sha256").hexdigest()
        report["model_assets_sha256"] = assets
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("retrieval_cases.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lexical-only", action="store_true")
    args = parser.parse_args()
    report = evaluate(json.loads(args.cases.read_text()), ("lexical",) if args.lexical_only else ("lexical", "hybrid"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({mode: value["summary"] for mode, value in report["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
