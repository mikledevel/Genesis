"""Проверка закреплённого Basecamp Handbook и документации Genesis, без LLM-вызовов."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval.kb.kb_eval import evaluate
from app.core.knowledge_base import chunk_text


def run(corpus):
    commit = subprocess.check_output(['git', '-C', str(corpus), 'rev-parse', 'HEAD'], text=True).strip()
    expected = 'c9536061f26d0b5bd9f2f900d63e27e9f4deb4ed'
    if commit != expected:
        raise ValueError('Corpus revision differs from the pinned benchmark revision')
    dev = json.loads((ROOT / 'eval/kb/questions_basecamp.json').read_text())
    heldout = json.loads((ROOT / 'eval/kb/questions_basecamp_heldout.json').read_text())
    baseline = evaluate(corpus, dev, chunk_text, 0.16)
    # Выбираем порог только на dev: сохраняем >=90% найденных базовым поиском ответов.
    candidates = [(g, evaluate(corpus, dev, chunk_text, g)) for g in (0.16, 0.18, 0.20, 0.24)]
    eligible = [(g, r) for g, r in candidates if r['top3'] >= baseline['top3'] * 0.9]
    selected, _ = max(eligible, key=lambda item: (item[1]['unanswerable_dropped'], item[1]['top3'], -item[0]))
    report = {'source': 'https://github.com/basecamp/handbook', 'commit': commit,
              'corpus_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(corpus.glob('*.md'))},
              'selection_rule': 'Maximize dev abstentions while retaining at least 90% of baseline dev top3 hits; tie-break by recall, then lower gate.',
              'selected_gate': selected, 'default_gate_unchanged': 0.16,
              'dev_candidates': [{'gate': g, 'metrics': r} for g, r in candidates],
              'heldout': {'baseline': evaluate(corpus, heldout, chunk_text, 0.16),
                          'selected': evaluate(corpus, heldout, chunk_text, selected)}}
    own = json.loads((ROOT / 'eval/kb/questions_genesis.json').read_text())
    names = ('LLM_IMPROVEMENTS.md', 'RENDER_DEPLOYMENT.md', 'BACKEND_HARDENING.md')
    with tempfile.TemporaryDirectory(prefix='genesis-docs-eval-') as work:
        for name in names:
            shutil.copy2(ROOT / 'docs' / name, Path(work) / name)
        report['genesis_docs'] = {str(g): evaluate(work, own, chunk_text, g) for g in (0.16, selected)}
    report['genesis_docs_sha256'] = {name: hashlib.sha256((ROOT / 'docs' / name).read_bytes()).hexdigest()
                                    for name in names}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('selected_gate', 'heldout', 'genesis_docs')}, ensure_ascii=False, indent=2))
