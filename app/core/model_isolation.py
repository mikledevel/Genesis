"""Проверка внешнего pickle вне процесса API; production требует Docker."""
import json
from pathlib import Path
import shutil
import tempfile


def evaluate_model_file(path, plan, features=None, model_type="custom", target=None):
    if target is not None and plan.target is None:
        from dataclasses import replace
        from app.core.engine.verification import infer_task
        plan = replace(plan,target=target,task=infer_task(plan.train[target]))
    from app.config import settings
    from app.core.codegen.docker_sandbox import DockerSandbox, docker_available
    if not docker_available():
        if not settings.debug:
            raise RuntimeError("An isolated Docker runner is required to import or check external models")
        return _evaluate_development(path, plan, features, model_type)
    with tempfile.TemporaryDirectory(prefix="genesis-model-check-") as directory:
        work = Path(directory)
        shutil.copyfile(path, work / "model.pkl")
        plan.train.to_csv(work / "train.csv",index=False)
        plan.holdout.to_csv(work / "holdout.csv",index=False)
        config = {"target":plan.target,"task":plan.task,"n_rows":plan.n_rows,"checks":plan.checks,"dropped":plan.dropped,
                  "features":features,"model_type":model_type}
        (work / "config.json").write_text(json.dumps(config,default=str))
        copy_model_runtime(work)
        script = '''import json
import pandas as pd
from app.core.engine.verification import DataPlan
from app.core.model_isolation import _evaluate_development
from app.core.safe_pickle import UnsafePickleError
with open('config.json') as f: c=json.load(f)
p=DataPlan(pd.read_csv('train.csv'),pd.read_csv('holdout.csv'),c['target'],c['task'],c['n_rows'],c['checks'],c['dropped'])
try:
    result=_evaluate_development('model.pkl',p,c['features'],c['model_type'])
except UnsafePickleError as error:
    result={'_error':'unsafe_pickle','message':str(error)[:1000]}
except ValueError as error:
    result={'_error':'invalid_model','message':str(error)[:1000]}
print('MODEL_CHECK_JSON:'+json.dumps(result,default=str))
print('PIPELINE_SUCCESS')
'''
        response = DockerSandbox().execute(script,directory)
        if not response.get("success"):
            raise ValueError("The isolated model check failed")
        for line in response.get("stdout", "").splitlines():
            if line.startswith("MODEL_CHECK_JSON:"):
                result = json.loads(line.partition(":")[2])
                if result.get("_error") == "unsafe_pickle":
                    from app.core.safe_pickle import UnsafePickleError
                    raise UnsafePickleError(str(result.get("message", "Model file refused"))[:1000])
                if result.get("_error"):
                    raise ValueError(str(result.get("message", "Invalid model"))[:1000])
                return result
        raise ValueError("No isolated model check result")


def _evaluate_development(path,plan,features,model_type):
    import pandas as pd
    from app.core.safe_pickle import safe_load
    from app.core.engine.verification import evaluate_holdout, assess
    with open(path,"rb") as handle:
        model = safe_load(handle)
    if not hasattr(model,"predict"):
        raise ValueError("The file does not contain a model with a predict() method")
    inner = model
    for _ in range(5):
        if hasattr(inner,"feature_names_in_") or not hasattr(inner,"estimator"):
            break
        inner = inner.estimator
    if features is None:
        names = getattr(inner,"feature_names_in_",None)
        features = [str(c) for c in names] if names is not None else [str(c) for c in plan.train.columns if c != plan.target]
    metrics,error,contract = evaluate_holdout(model,plan,plan.target,features)
    if error:
        raise ValueError("The platform could not score this model on the hidden test set. " + error)
    metrics["trust"] = assess(metrics,plan,{})
    numeric = [c for c in features if c in plan.train.columns and pd.api.types.is_numeric_dtype(plan.train[c])]
    estimator = inner.steps[-1][1] if getattr(inner,"steps",None) else inner
    return {"metrics":metrics,"contract":contract,"features":features,"target":plan.target,"task":plan.task,
            "model_type":type(estimator).__name__ if model_type == "custom" else model_type,
            "input_schema":{"numeric":numeric,"categorical":[c for c in features if c not in numeric],"datetime":[],"text":[]}}


def copy_model_runtime(work):
    root = Path(__file__).resolve().parents[2]
    for relative in ("app/__init__.py","app/core/__init__.py","app/core/safe_pickle.py","app/core/engine/__init__.py",
                     "app/core/engine/trainer.py","app/core/engine/transformers.py","app/core/engine/forecaster.py",
                     "app/core/engine/verification.py","app/core/engine/predict_utils.py","app/core/model_isolation.py"):
        target = work / relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root / relative,target)


def untrusted_record(record):
    meta = record.get("metadata") or {}
    return meta.get("trained_on") == "user_machine" or meta.get("built_via") in {"ModelBuilder","local_training","AICodeGenerator"}


def improve_isolated(registry,model_id,data,target,task,**options):
    from types import SimpleNamespace
    from app.core.codegen.docker_sandbox import DockerSandbox, docker_available
    if not docker_available():
        raise RuntimeError("An isolated Docker runner is required to improve this external model")
    record = registry.get_record(model_id)
    if getattr(registry,"_document",None):
        registry._document.db.restore_artifact(record["model_path"])
    with tempfile.TemporaryDirectory(prefix="genesis-model-improve-") as directory:
        work = Path(directory)
        copy_model_runtime(work)
        source = Path(__file__).with_name("model_improver.py")
        shutil.copyfile(source,work / "app/core/model_improver.py")
        shutil.copyfile(record["model_path"],work / "model.pkl")
        data.to_csv(work / "data.csv",index=False)
        (work / "options.json").write_text(json.dumps({"target":target,"task":task,**options}))
        script = '''import json,pickle
import pandas as pd
from pathlib import Path
from app.core.safe_pickle import safe_load
from app.core.model_improver import improve_model
with open('options.json') as f: options=json.load(f)
with open('model.pkl','rb') as f: model=safe_load(f)
result=improve_model(model,pd.read_csv('data.csv'),**options)
if result.improved:
    Path('models').mkdir(exist_ok=True)
    with open('models/improved.pkl','wb') as f: pickle.dump(result.model,f)
print('IMPROVE_JSON:'+json.dumps({'improved':result.improved,'report':result.report},default=str))
print('PIPELINE_SUCCESS')
'''
        runner = DockerSandbox()
        runner.TIMEOUT_SECONDS = 600
        response = runner.execute(script,directory)
        if not response.get("success"):
            raise RuntimeError("Isolated model improvement failed")
        for line in response.get("stdout","").splitlines():
            if line.startswith("IMPROVE_JSON:"):
                result = json.loads(line.partition(":")[2])
                destination = None
                if result["improved"]:
                    import uuid
                    output = work / "models/improved.pkl"
                    if output.is_symlink() or work.resolve() not in output.resolve().parents:
                        raise ValueError("Unsafe isolated model output")
                    destination = Path("models") / ("improved_"+uuid.uuid4().hex+".pkl")
                    destination.parent.mkdir(exist_ok=True)
                    shutil.copyfile(work / "models/improved.pkl",destination)
                return SimpleNamespace(improved=result["improved"],report=result["report"],model=None,model_path=destination)
        raise RuntimeError("Missing isolated improvement result")
