"""Единая граница предсказаний; пользовательские модели требуют sandbox."""
import json
import tempfile
from pathlib import Path
import shutil


def predict_model(registry, model_id, features):
    from app.core.tracing import span
    record = registry.get_record(model_id)
    with span("model_inference", model_id=model_id, model_version=record.get("version", 1)):
        import numpy as np
        prediction = _predict_model(registry, model_id, features, record)
        array = np.asarray(prediction)
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError("The model returned a non-finite prediction")
        return prediction


def _predict_model(registry, model_id, features, record):
    import numpy as np
    from app.config import settings
    meta = record.get("metadata") or {}
    untrusted = meta.get("trained_on") == "user_machine" or meta.get("built_via") in {"ModelBuilder", "local_training", "AICodeGenerator"}
    if not untrusted:
        return registry.get_model(model_id).predict(features)
    from app.core.codegen.docker_sandbox import DockerSandbox, docker_available
    if not docker_available():
        raise RuntimeError("An isolated model runner is required for this model")
    with tempfile.TemporaryDirectory(prefix="genesis-inference-") as directory:
        work = Path(directory)
        from app.core.model_isolation import copy_model_runtime
        copy_model_runtime(work)
        if getattr(registry,"_document",None):
            registry._document.db.restore_artifact(record["model_path"])
        shutil.copyfile(record["model_path"], work / "model.pkl")
        data = features.to_json(orient="split", date_format="iso") if hasattr(features, "to_json") else json.dumps(np.asarray(features).tolist())
        (work / "input.json").write_text(data)
        code = '''import json, pickle
import pandas as pd
import numpy as np
with open('input.json') as f: value=json.load(f)
x=pd.DataFrame(value['data'],columns=value['columns'],index=value.get('index')) if isinstance(value,dict) else np.asarray(value)
with open('model.pkl','rb') as f: model=pickle.load(f)
print('PREDICTION_JSON:'+json.dumps(model.predict(x).tolist()))
print('PIPELINE_SUCCESS')
'''
        result = DockerSandbox().execute(code, directory)
        if not result.get("success"):
            raise RuntimeError("Isolated prediction failed")
        for line in result.get("stdout", "").splitlines():
            if line.startswith("PREDICTION_JSON:"):
                return np.asarray(json.loads(line.partition(":")[2]))
        raise RuntimeError("Isolated runner returned no prediction")
