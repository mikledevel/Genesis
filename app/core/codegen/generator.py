"""Экспорт пайплайна с тем же trainer, который работает на сервере."""
import os
import re
import uuid
from pathlib import Path


class CodeGenerator:
    def __init__(self, db=None):
        self.db = db

    def generate_from_project(self, project_name):
        return self.generate_from_description(project_name)

    def generate_from_description(self, description):
        target = self._detect_target(description)
        return self._script(target)

    def _detect_target(self, description):
        match = re.search(r"(?:target|predict|таргет|столбец|колонку)\s+([\w-]+)", description, re.I)
        return match.group(1) if match else None

    def _script(self, target):
        return """# Запускайте из распакованного /api/local-training/bundle после установки requirements-local.txt.
import argparse
import json
import pickle
from pathlib import Path
from app.core.engine.trainer import DatasetTrainer
from train_local import read_table, infer_task

parser = argparse.ArgumentParser(description='Genesis AI: verified training pipeline')
parser.add_argument('data', help='CSV, JSON, XLSX or Parquet dataset')
parser.add_argument('--target', default=TARGET_DEFAULT, required=TARGET_REQUIRED)
parser.add_argument('--task', default='auto', choices=['auto','binary_classification','multiclass_classification','regression'])
parser.add_argument('--budget', default='balanced', choices=['fast','balanced','thorough'])
parser.add_argument('--out', default='model-output')
args = parser.parse_args()
df = read_table(args.data)
if args.target not in df.columns:
    parser.error('Target column does not exist')
task = infer_task(df[args.target].dropna()) if args.task == 'auto' else args.task
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
trainer = DatasetTrainer(model_dir=str(out / 'training'))
result = trainer.train(df, args.target, task, model_type='auto', budget=args.budget)
with (out / 'model.pkl').open('wb') as handle:
    pickle.dump(result.model, handle)
(out / 'metrics.json').write_text(json.dumps(result.metrics, ensure_ascii=False, default=str), encoding='utf-8')
print(json.dumps({'model': str(out / 'model.pkl'), 'metrics': result.metrics}, ensure_ascii=False, default=str))
""".replace('TARGET_DEFAULT', repr(target)).replace('TARGET_REQUIRED', repr(target is None))

    def save_code(self, code, filename=None, owner_user_id=None):
        prefix = f'user_{owner_user_id}_' if owner_user_id else 'anon_'
        name = filename or prefix + 'pipeline_' + uuid.uuid4().hex + '.py'
        if Path(name).name != name:
            raise ValueError('A filename without directories is required')
        destination = Path('generated') / name
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(code, encoding='utf-8')
        from app.db.database import GenesisDB
        GenesisDB().persist_artifact(destination, owner_user_id or 'system')
        return str(destination)
