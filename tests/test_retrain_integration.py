"""
Integration test for the full "retrain an existing model" flow through GenesisAgent -
real training (DatasetTrainer), real registry, isolated tempdirs throughout so this never
touches the real ./registry or ./models.
"""
import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from app.core.agent import GenesisAgent
from app.core.engine.registry import ModelRegistry
from app.core.engine.trainer import DatasetTrainer


def _make_dataset(path, n=100, seed=0):
    import numpy as np
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "age": rng.randint(20, 65, n),
        "salary": rng.randint(30000, 120000, n),
        "churn": rng.randint(0, 2, n),
    })
    df.to_csv(path, index=False)


def test_full_retrain_flow_creates_new_version_with_lineage():
    workdir = tempfile.mkdtemp()
    try:
        os.chdir(workdir)
        os.makedirs("datasets", exist_ok=True)
        ds1 = "datasets/churn_v1.csv"
        ds2 = "datasets/churn_v2.csv"
        _make_dataset(ds1, seed=0)
        _make_dataset(ds2, seed=1)  # a "new/updated" dataset

        registry = ModelRegistry(storage_dir=os.path.join(workdir, "registry"))
        trainer = DatasetTrainer(model_dir=os.path.join(workdir, "models"))
        agent = GenesisAgent(registry=registry, trainer=trainer)
        agent.current_user_id = "test_user"

        # 1. Initial training via the normal chat-tool path
        train_resp = agent._do_train(f"train {ds1}", {"target": "churn"})
        assert train_resp.success
        original_id = train_resp.data["model_id"]
        original_record = registry.get_record(original_id)
        assert original_record["version"] == 1
        assert original_record["family_id"] == original_id

        # 2. Retrain on the new dataset
        retrain_resp = agent._do_retrain(original_id, ds2, {})
        assert retrain_resp.success, retrain_resp.message
        new_id = retrain_resp.data["model_id"]
        assert new_id != original_id

        new_record = registry.get_record(new_id)
        assert new_record["version"] == 2
        assert new_record["family_id"] == original_id
        assert new_record["parent_model_id"] == original_id
        # same task/target carried over from the original
        assert new_record["task"] == original_record["task"]
        assert new_record["target"] == original_record["target"]

        # 3. History shows both versions, newest first, and the OLD version is still
        # fully intact and usable - nothing was overwritten or deleted.
        history = registry.get_family_history(original_id)
        assert len(history) == 2
        assert history[0]["model_id"] == new_id
        assert history[1]["model_id"] == original_id
        old_model_still_works = registry.get_model(original_id)
        assert old_model_still_works is not None

        # 4. The new version's registered features match what the retrained model
        # actually expects (the same feature-column-integrity check that matters for the
        # original training path applies here too).
        import numpy as np
        new_model = registry.get_model(new_id)
        row = np.zeros((1, len(new_record["features"])))
        new_model.predict(row)  # must not raise a shape-mismatch error
    finally:
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shutil.rmtree(workdir, ignore_errors=True)


def test_retrain_nonexistent_model_fails_cleanly():
    workdir = tempfile.mkdtemp()
    try:
        registry = ModelRegistry(storage_dir=os.path.join(workdir, "registry"))
        from app.db.database import GenesisDB
        db = GenesisDB(db_path=os.path.join(workdir, "test.db"))
        agent = GenesisAgent(registry=registry, db=db)
        resp = agent._do_retrain("does_not_exist", "some.csv", {})
        assert not resp.success
        assert "не найдена" in resp.message.lower() or "not found" in resp.message.lower()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_retrain_missing_dataset_file_fails_cleanly():
    workdir = tempfile.mkdtemp()
    try:
        os.chdir(workdir)
        os.makedirs("datasets", exist_ok=True)
        _make_dataset("datasets/churn.csv")
        registry = ModelRegistry(storage_dir=os.path.join(workdir, "registry"))
        trainer = DatasetTrainer(model_dir=os.path.join(workdir, "models"))
        agent = GenesisAgent(registry=registry, trainer=trainer)
        agent.current_user_id = "test_user"
        train_resp = agent._do_train("train datasets/churn.csv", {"target": "churn"})
        model_id = train_resp.data["model_id"]

        resp = agent._do_retrain(model_id, "datasets/does_not_exist.csv", {})
        assert not resp.success
        # original model must be completely untouched by a failed retrain attempt
        assert registry.get_record(model_id)["version"] == 1
    finally:
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shutil.rmtree(workdir, ignore_errors=True)


def test_retrain_can_override_model_type():
    workdir = tempfile.mkdtemp()
    try:
        os.chdir(workdir)
        os.makedirs("datasets", exist_ok=True)
        _make_dataset("datasets/churn.csv")
        registry = ModelRegistry(storage_dir=os.path.join(workdir, "registry"))
        trainer = DatasetTrainer(model_dir=os.path.join(workdir, "models"))
        agent = GenesisAgent(registry=registry, trainer=trainer)
        agent.current_user_id = "test_user"
        train_resp = agent._do_train("train datasets/churn.csv", {"target": "churn", "model_type": "random_forest"})
        model_id = train_resp.data["model_id"]
        assert registry.get_record(model_id)["model_type"] == "random_forest"

        retrain_resp = agent._do_retrain(model_id, "datasets/churn.csv", {"model_type": "logistic"})
        assert retrain_resp.success
        new_record = registry.get_record(retrain_resp.data["model_id"])
        assert new_record["model_type"] == "logistic"
    finally:
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shutil.rmtree(workdir, ignore_errors=True)
