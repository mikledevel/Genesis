import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.real_docker,
              pytest.mark.skipif(os.environ.get("GENESIS_TEST_DOCKER") != "1",reason="Set GENESIS_TEST_DOCKER=1 for the real container checks")]


def test_real_container_is_unprivileged_readonly_and_has_no_network(tmp_path):
    from app.core.codegen.docker_sandbox import DockerSandbox, docker_available
    assert docker_available()
    response = DockerSandbox().execute('''import os,socket
assert os.getuid()!=0
try:
    open('/rootfs-write-test','w').write('no')
except OSError:
    pass
else:
    raise AssertionError('Writable root filesystem')
try:
    socket.create_connection(('1.1.1.1',80),timeout=1)
except OSError:
    pass
else:
    raise AssertionError('Unexpected network access')
print('PIPELINE_SUCCESS')
''',str(tmp_path))
    assert response["success"],response


def test_external_model_prediction_and_verification_use_real_container(tmp_path,monkeypatch):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from app.core.engine.registry import ModelRegistry
    from app.core.inference import predict_model
    from app.core.model_isolation import evaluate_model_file
    from app.core.engine.verification import DataPlan
    from app.config import settings
    monkeypatch.chdir(tmp_path)
    frame=pd.DataFrame({"x":np.arange(40),"target":[0]*20+[1]*20})
    model=Pipeline([('scale',StandardScaler()),('model',LogisticRegression())]).fit(frame[['x']],frame.target)
    registry=ModelRegistry()
    mid=registry.register(model,{},'binary_classification','logistic',['x'],'target',metadata={'author':'external-owner','trained_on':'user_machine'})
    monkeypatch.setattr(settings,'debug',False)
    with pytest.raises(RuntimeError,match='isolated'):
        registry.get_model(mid)
    prediction=predict_model(registry,mid,pd.DataFrame({'x':[1,38]}))
    assert prediction.tolist()==[0,1]
    plan=DataPlan(frame.iloc[:30],frame.iloc[30:],'target','binary_classification',40)
    verified=evaluate_model_file(registry.get_record(mid)['model_path'],plan)
    assert verified['features']==['x']
    assert verified['metrics']['n_test']==10


def test_real_skill_container_uses_guarded_host_bridge(monkeypatch):
    from app.core.codegen.docker_sandbox import DockerSandboxNetworked
    from app.core.skills.net_guard import SafeFetcher
    requests = []
    def fetch(self, url, *args, **kwargs):
        requests.append(url)
        return {"status": 200, "body": "isolated test"}
    monkeypatch.setattr(SafeFetcher, "fetch", fetch)
    result = DockerSandboxNetworked().execute_with_params(
        'def run(params):\n    return fetch("https://example.com/test")', {}, ["example.com"])
    assert result["success"], result
    assert requests == ["https://example.com/test"]


def test_container_preserves_refusal_for_hostile_pickle(tmp_path):
    import pickle
    from app.core.model_isolation import evaluate_model_file
    from app.core.engine.verification import DataPlan
    from app.core.safe_pickle import UnsafePickleError
    class Evil:
        def __reduce__(self):
            return os.system, ('touch pwned_by_pickle',)
    file = tmp_path / 'hostile.pkl'
    file.write_bytes(pickle.dumps(Evil()))
    frame = pd.DataFrame({'x': range(30), 'target': [0, 1] * 15})
    plan = DataPlan(frame.iloc[:20], frame.iloc[20:], 'target', 'binary_classification', 30)
    with pytest.raises(UnsafePickleError):
        evaluate_model_file(str(file), plan)
    assert not (tmp_path / 'pwned_by_pickle').exists()


def test_container_preserves_incompatible_features_error(tmp_path):
    import pickle
    from sklearn.linear_model import LogisticRegression
    from app.core.model_isolation import evaluate_model_file
    from app.core.engine.verification import DataPlan
    model = LogisticRegression().fit(np.arange(90).reshape(30, 3), [0, 1] * 15)
    file = tmp_path / 'incompatible.pkl'
    file.write_bytes(pickle.dumps(model))
    frame = pd.DataFrame({'x': range(30), 'target': [0, 1] * 15})
    plan = DataPlan(frame.iloc[:20], frame.iloc[20:], 'target', 'binary_classification', 30)
    with pytest.raises(ValueError, match='could not score'):
        evaluate_model_file(str(file), plan)
