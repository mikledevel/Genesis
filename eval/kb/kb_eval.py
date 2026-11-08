"""Measure how well the bot knowledge base (app/core/knowledge_base.py) finds the right passage - on YOUR documents.

    git clone --depth 1 https://github.com/basecamp/handbook /tmp/handbook
    python3 eval/kb/kb_eval.py --corpus /tmp/handbook --questions eval/kb/questions_basecamp.json \\
                                                                  --questions eval/kb/questions_basecamp_heldout.json
    python3 eval/kb/kb_eval.py ... --sweep            # the same questions at several relevance thresholds
    python3 eval/kb/kb_eval.py ... --chunker section  # the opt-in heading-aware chunker

Prints top-1 / top-3 recall for answerable questions and how many unanswerable questions correctly retrieve nothing.
To check a customer's documents, write a questions file in the same format (questions their newcomers really ask, with the file and
section heading where the answer lives, plus a few questions the documents do NOT answer). The retrieval gate in knowledge_base.py was
tuned on the Basecamp dev set (questions_basecamp.json) only; questions_basecamp_heldout.json was written afterwards and is for checking,\nnot for tuning. Run this on the customer's corpus before trusting the gate.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app.core.knowledge_base import MAX_CHUNKS_RETRIEVED, chunk_text, chunk_text_by_section, retrieve_relevant_chunks  # noqa: E402


def _sections(text, default):
    out, cur, buf = {}, default, []
    for line in text.split("\n"):
        if line.startswith("#"):
            out[cur] = "\n".join(buf)
            cur, buf = line.lstrip("# ").strip(), [line]
        else:
            buf.append(line)
    out[cur] = "\n".join(buf)
    return out


def load_chunks(corpus_dir, chunker=chunk_text):
    files = sorted(f for f in os.listdir(corpus_dir) if f.lower().endswith((".md", ".txt")))
    texts = {f: open(os.path.join(corpus_dir, f), encoding="utf-8-sig").read() for f in files}
    chunks = [{"content": c, "filename": f} for f in files for c in chunker(texts[f])]
    return chunks, {f: _sections(texts[f], f) for f in files}


def _is_hit(chunk, sections, filename, heading_part):
    if chunk["filename"] != filename:
        return False
    body = next((t for h, t in sections[filename].items() if heading_part.lower() in h.lower()), None)
    if body is None:
        raise ValueError(f"ground truth points at a section that does not exist: {filename} / {heading_part}")
    body, c = re.sub(r"\s+", " ", body), re.sub(r"\s+", " ", chunk["content"])
    return any(body[i:i + 60] in c for i in range(0, max(1, len(body) - 60), 15))


def evaluate(corpus_dir, questions, chunker=chunk_text, min_similarity=None):
    chunks, sections = load_chunks(corpus_dir, chunker)
    kw = {} if min_similarity is None else {"min_similarity": min_similarity}
    n = top1 = top3 = answerable_empty = 0
    misses = []
    for item in questions["answerable"]:
        got = retrieve_relevant_chunks(item["q"], chunks, **kw)
        answerable_empty += not got
        ok = [any(_is_hit(c, sections, f, h) for f, h in item["truth"]) for c in got]
        n += 1
        top1 += bool(ok and ok[0])
        top3 += any(ok)
        if not any(ok):
            misses.append((item["q"], [c["filename"] for c in got]))
    empty = [len(retrieve_relevant_chunks(q, chunks, **kw)) == 0 for q in questions["unanswerable"]]
    return {"chunks": len(chunks), "n": n, "top1": top1, "top3": top3, "misses": misses, "answerable_empty": answerable_empty,
            "unanswerable": len(empty), "unanswerable_dropped": sum(empty)}


def _report(name, r):
    print(f"[{name}] {r['chunks']} chunks; answerable questions: {r['n']}")
    print(f"  best chunk first : {r['top1']}/{r['n']} ({100 * r['top1'] // r['n']}%)")
    print(f"  in top {MAX_CHUNKS_RETRIEVED}          : {r['top3']}/{r['n']} ({100 * r['top3'] // r['n']}%)")
    print(f"  answerable questions that got NOTHING (gate said 'not covered'): {r['answerable_empty']}/{r['n']}")
    print(f"  unanswerable questions that correctly retrieve nothing: {r['unanswerable_dropped']}/{r['unanswerable']}")
    for q, files in r["misses"]:
        print(f"  missed: {q}  -> {files}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="folder with .md / .txt documents")
    ap.add_argument("--questions", required=True, action="append", help="JSON file: {answerable: [{q, truth}], unanswerable: [q]}; repeatable")
    ap.add_argument("--chunker", choices=["default", "section"], default="default")
    ap.add_argument("--min-similarity", type=float, default=None, help="override the relevance gate")
    ap.add_argument("--sweep", action="store_true", help="print a table over several gate values instead of the full report")
    a = ap.parse_args()
    chunker = chunk_text_by_section if a.chunker == "section" else chunk_text
    sets = [(os.path.basename(p).replace("questions_", "").replace(".json", ""), json.load(open(p, encoding="utf-8"))) for p in a.questions]
    if a.sweep:
        print("gate   " + "   ".join(f"{name:>34}" for name, _ in sets))
        print("       " + "   ".join(f"{'top3  not-covered(ans)  dropped(unans)':>34}" for _ in sets))
        for g in (0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.24, 0.28):
            cells = []
            for _, q in sets:
                r = evaluate(a.corpus, q, chunker, g)
                cells.append(f"{r['top3']:>2}/{r['n']:<2}   {r['answerable_empty']:>2}/{r['n']:<2}        {r['unanswerable_dropped']:>2}/{r['unanswerable']:<2}".rjust(34))
            print(f"{g:<6.2f} " + "   ".join(cells))
    else:
        for name, q in sets:
            _report(name, evaluate(a.corpus, q, chunker, a.min_similarity))
