"""Registry safety under concurrent writers, atomic index writes, and no orphan pkl copies."""
import json, os, sys, tempfile, threading, multiprocessing as mp
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.engine.registry import ModelRegistry


def _register_many(storage, n, tag):
    reg = ModelRegistry(storage_dir=storage)
    for i in range(n):
        reg.register({"x": i}, {"acc": 0.5}, "regression", "linear", ["a"], "y", metadata={"tag": tag})


def test_threads_with_separate_registry_objects_lose_nothing():
    d = tempfile.mkdtemp()
    ts = [threading.Thread(target=_register_many, args=(d, 15, f"t{i}")) for i in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(ModelRegistry(storage_dir=d).list_models(limit=1000)) == 60


def test_separate_processes_lose_nothing():
    d = tempfile.mkdtemp()
    ps = [mp.get_context("spawn").Process(target=_register_many, args=(d, 10, f"p{i}")) for i in range(3)]
    [p.start() for p in ps]; [p.join(60) for p in ps]
    assert all(p.exitcode == 0 for p in ps)
    assert len(ModelRegistry(storage_dir=d).list_models(limit=1000)) == 30


def test_reader_sees_registration_made_by_another_instance():
    d = tempfile.mkdtemp()
    a, b = ModelRegistry(storage_dir=d), ModelRegistry(storage_dir=d)
    mid = b.register({"x": 1}, {}, "regression", "linear", ["a"], "y")
    assert a.get_record(mid)["model_id"] == mid          # `a` never registered it itself


def test_index_is_valid_json_and_no_temp_file_left():
    d = tempfile.mkdtemp()
    _register_many(d, 3, "x")
    json.load(open(os.path.join(d, "index.json"), encoding="utf-8"))
    assert not os.path.exists(os.path.join(d, "index.json.tmp"))


def test_agent_train_leaves_exactly_one_pkl(tmp_path, monkeypatch):
    from app.core.agent import GenesisAgent
    from app.core.engine.trainer import DatasetTrainer
    from app.db.database import GenesisDB
    monkeypatch.chdir(tmp_path)
    os.makedirs("datasets")
    rng = np.random.RandomState(0)
    x = rng.randn(300)
    pd.DataFrame({"x": x, "y": (x + rng.randn(300) * 0.5 > 0).astype(int)}).to_csv("datasets/d.csv", index=False)
    tdir = tmp_path / "trained"
    agent = GenesisAgent(trainer=DatasetTrainer(model_dir=str(tdir)),
                         registry=ModelRegistry(storage_dir=str(tmp_path / "reg")),
                         db=GenesisDB(db_path=str(tmp_path / "t.db")))
    agent.current_user_id = "u1"
    r = agent._do_train("train", {"path": "datasets/d.csv", "target_column": "y", "model_type": "logistic"})
    assert r.success, r.message
    assert list(tdir.glob("*.pkl")) == []                              # trainer's temp copy removed
    assert len(list((tmp_path / "reg" / "models").glob("*.pkl"))) == 1
    assert os.path.exists(r.data["model_path"])                        # path returned to the UI is the real one
