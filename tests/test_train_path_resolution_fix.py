"""
Regression test for a real bug found via live QA testing (real Groq key, real server):
_do_train never consulted args["path"] - the LLM's own extracted path argument - unlike
analyze_dataset and retrain_model, which both already preferred it. It relied entirely on
a raw-text regex re-scan, and when that failed (e.g. the LLM omitted a path because its
few-shot example never demonstrated supplying one), the fallback error message showed an
unrelated hardcoded example filename ("datasets/churn_demo.csv") instead of anything
actionable.

Fixed by: (1) preferring args["path"] like the other two handlers already do, (2) adding a
train_model few-shot example that demonstrates extracting path into args, and (3) making the
"file not found" error show the actual path that was tried.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import numpy as np
from app.core.agent import GenesisAgent
from app.core.engine.registry import ModelRegistry
from app.core.engine.trainer import DatasetTrainer
from app.db.database import GenesisDB


def _make_agent(workdir):
    registry = ModelRegistry(storage_dir=os.path.join(workdir, "registry"))
    trainer = DatasetTrainer(model_dir=os.path.join(workdir, "models"))
    db = GenesisDB(db_path=os.path.join(workdir, "test.db"))
    agent = GenesisAgent(registry=registry, trainer=trainer, db=db)
    agent.current_user_id = "test_user"
    return agent


def _make_dataset(path, n=50, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({"age": rng.randint(20, 65, n), "churn": rng.randint(0, 2, n)})
    df.to_csv(path, index=False)


class TestTrainModelPathResolution:
    def test_explicit_path_arg_is_used_directly(self):
        """The fix: when the LLM supplies args['path'] (as the updated few-shot example now
        teaches it to), _do_train must use it directly rather than re-deriving from raw text."""
        workdir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(workdir, "datasets"))
            ds = os.path.join(workdir, "datasets", "churn.csv")
            _make_dataset(ds)
            agent = _make_agent(workdir)
            agent.ALLOWED_DIRS = [os.path.join(workdir, "datasets"), os.path.join(workdir, "models")]
            resp = agent._do_train("some message", {"path": ds, "target_column": "churn"})
            assert resp.success, resp.message
        finally:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)

    def test_falls_back_to_regex_extraction_when_no_path_arg(self):
        """Backward compatible: if the LLM genuinely doesn't supply a path (e.g. 'train
        xgboost on churn' with no filename in the message at all, or a conversation where the
        dataset was already analyzed earlier), the raw-text regex fallback must still work."""
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            os.makedirs("datasets")
            _make_dataset("datasets/churn.csv")
            agent = _make_agent(workdir)
            resp = agent._do_train("train a model on datasets/churn.csv to predict churn", {"target_column": "churn"})
            assert resp.success, resp.message
        finally:
            os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)

    def test_missing_file_error_shows_the_actual_path_tried(self):
        """This is the exact UX bug found live: the old message was a generic hardcoded
        example ('Загрузите датасет: datasets/churn_demo.csv') with no relation to what was
        actually attempted. Now it must name the real path that failed."""
        workdir = tempfile.mkdtemp()
        try:
            agent = _make_agent(workdir)
            resp = agent._do_train("train a model", {"path": "datasets/this_does_not_exist.csv"})
            assert not resp.success
            assert "this_does_not_exist.csv" in resp.message
            assert "churn_demo" not in resp.message  # the old unrelated example must be gone
        finally:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)

    def test_no_path_at_all_gives_actionable_message_not_a_fake_example(self):
        workdir = tempfile.mkdtemp()
        try:
            agent = _make_agent(workdir)
            resp = agent._do_train("train a model on churn", {})
            assert not resp.success
            assert "churn_demo" not in resp.message
        finally:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)

    def test_path_outside_allowed_dirs_is_rejected_with_clear_reason(self):
        workdir = tempfile.mkdtemp()
        try:
            outside = tempfile.mkdtemp()
            _make_dataset(os.path.join(outside, "sneaky.csv"))
            agent = _make_agent(workdir)
            resp = agent._do_train("train a model", {"path": os.path.join(outside, "sneaky.csv")})
            assert not resp.success
            assert "запрещ" in resp.message.lower() or "denied" in resp.message.lower()
        finally:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)
            shutil.rmtree(outside, ignore_errors=True)
