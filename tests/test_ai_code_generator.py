"""
Tests for the AICodeGenerator ("our models build user models") hardening:
  - shape validation catches a script that runs cleanly but declares a wrong feature list
  - the fix loop actually retries and recovers from both execution errors and validation errors
  - the catalog-driven default model resolution
  - a full realistic build() -> real subprocess execution -> real model file -> ModelRegistry
    registration chain, with only the Groq call itself mocked (no live GROQ_API_KEY here -
    see the honesty note in the final summary about what could/couldn't be tested live)

No Docker daemon is available in this environment, so these naturally exercise the real
subprocess execution fallback path (see docker_sandbox.docker_available()).
"""
import os, sys, tempfile, shutil, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from unittest.mock import patch
from app.core.codegen.ai_generator import AICodeGenerator

VALID_PIPELINE_SCRIPT = '''
import pandas as pd, numpy as np, pickle, json, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("{dataset_path}")
target = "churn"
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
os.makedirs("models", exist_ok=True)
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("METRICS_JSON:" + json.dumps({{"accuracy": acc, "task": "binary_classification", "model_type": "random_forest", "target": target, "features": list(X_train.columns)}}))
print("PIPELINE_SUCCESS")
'''

# Same script, but the declared "features" list is WRONG (missing a column) - the model was
# actually trained on 2 columns but the script claims only 1. This is exactly the failure
# mode the shape-validation step exists to catch.
BROKEN_FEATURES_SCRIPT = VALID_PIPELINE_SCRIPT.replace(
    '"features": list(X_train.columns)', '"features": [list(X_train.columns)[0]]')


@pytest.fixture
def demo_csv():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "churn.csv")
    with open(path, "w") as f:
        f.write("age,salary,churn\n")
        for i in range(30):
            f.write(f"{20+i},{40000+i*1000},{i % 2}\n")
    yield path
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def workdir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)
    # build() also writes to ./models and ./generated in the cwd - clean those test artifacts up
    for p in ("models/best_model.pkl",):
        if os.path.exists(p):
            os.remove(p)


class TestCatalogIntegration:
    def test_default_model_pulled_from_catalog_not_hardcoded(self):
        gen = AICodeGenerator()
        from app.core.models_catalog import MODEL_CATALOG
        assert gen.MODEL in MODEL_CATALOG
        assert "coding" in MODEL_CATALOG[gen.MODEL].categories

    def test_explicit_model_override_still_works(self):
        gen = AICodeGenerator(model="some-other-model-id")
        assert gen.MODEL == "some-other-model-id"


class TestShapeValidation:
    def test_valid_script_passes_validation(self, demo_csv, workdir):
        gen = AICodeGenerator()
        code = VALID_PIPELINE_SCRIPT.format(dataset_path=demo_csv)
        result = gen.execute(code, workdir)
        assert result["success"] is True
        metrics, error = gen._validate_saved_model(workdir, result["stdout"])
        assert error is None
        assert metrics["features"] == ["age", "salary"]

    def test_broken_features_list_is_caught_not_silently_registered(self, demo_csv, workdir):
        """This is the exact bug class found and fixed in trainer.py, now guarded against for
        LLM-generated pipelines too: the script exits 0 and prints PIPELINE_SUCCESS, but its
        declared feature list doesn't match what the model actually expects."""
        gen = AICodeGenerator()
        code = BROKEN_FEATURES_SCRIPT.format(dataset_path=demo_csv)
        result = gen.execute(code, workdir)
        assert result["success"] is True  # the script itself ran fine
        metrics, error = gen._validate_saved_model(workdir, result["stdout"])
        assert error is not None  # but validation catches the real problem
        assert "features" in error.lower()

    def test_missing_metrics_json_line_is_caught(self, workdir):
        gen = AICodeGenerator()
        code = 'print("PIPELINE_SUCCESS")'
        result = gen.execute(code, workdir)
        metrics, error = gen._validate_saved_model(workdir, result["stdout"])
        assert error is not None
        assert "METRICS_JSON" in error


class TestBuildLoop:
    def test_build_end_to_end_with_mocked_groq_real_execution(self, demo_csv):
        """Full realistic chain: generate() is mocked (no live Groq key), but execute() runs
        the REAL subprocess, produces a REAL pickled model, and _validate_saved_model performs
        a REAL predict() shape check - only the LLM call itself is faked."""
        gen = AICodeGenerator()
        script = VALID_PIPELINE_SCRIPT.format(dataset_path=demo_csv)
        with patch.object(gen, "generate", return_value=script):
            result = gen.build("predict churn", dataset_path=demo_csv)
        assert result["success"] is True
        assert result["saved_model_path"] is not None
        assert os.path.exists(result["saved_model_path"])
        assert result["metrics"]["features"] == ["age", "salary"]

        # And the saved model is genuinely usable - register it and predict, same as
        # ModelBuilder.register() + a real /api/models/predict call would.
        from app.core.engine.registry import ModelRegistry
        import numpy as np
        reg_dir = tempfile.mkdtemp()
        try:
            reg = ModelRegistry(storage_dir=reg_dir)
            with open(result["saved_model_path"], "rb") as f:
                model = pickle.load(f)
            mid = reg.register(model=model, metrics=result["metrics"], task="binary_classification",
                               model_type="random_forest", features=result["metrics"]["features"], target="churn")
            loaded = reg.get_model(mid)
            pred = loaded.predict(np.zeros((1, 2)))
            assert pred is not None
        finally:
            shutil.rmtree(reg_dir, ignore_errors=True)
            if os.path.exists("models/best_model.pkl"):
                os.remove("models/best_model.pkl")
            gen_files = os.path.dirname(result.get("saved_path", "")) if result.get("saved_path") else None

    def test_self_correction_recovers_from_a_real_execution_error(self, demo_csv):
        """First attempt has a real Python bug (NameError) - the fix loop should call fix()
        and succeed on the second attempt."""
        broken_script = 'import pandas as pd\ndf = pd.read_csv("{dataset_path}")\nprint(undefined_variable)'.format(dataset_path=demo_csv)
        fixed_script = VALID_PIPELINE_SCRIPT.format(dataset_path=demo_csv)
        gen = AICodeGenerator()
        with patch.object(gen, "generate", return_value=broken_script), \
             patch.object(gen, "fix", return_value=fixed_script) as mock_fix:
            result = gen.build("predict churn", dataset_path=demo_csv, max_iterations=2)
        assert result["success"] is True
        assert mock_fix.call_count == 1  # fixed on the first retry
        assert result["iterations"] == 2  # attempt 1 (failed) + attempt 2 (succeeded)
        if os.path.exists("models/best_model.pkl"):
            os.remove("models/best_model.pkl")

    def test_self_correction_triggers_on_validation_error_not_just_execution_error(self, demo_csv):
        """Critical case: the FIRST script runs with exit code 0 (no execution error at all)
        but has a broken features list - the fix loop must still trigger, driven by the
        validation error rather than a traceback."""
        broken_script = BROKEN_FEATURES_SCRIPT.format(dataset_path=demo_csv)
        fixed_script = VALID_PIPELINE_SCRIPT.format(dataset_path=demo_csv)
        gen = AICodeGenerator()
        with patch.object(gen, "generate", return_value=broken_script), \
             patch.object(gen, "fix", return_value=fixed_script) as mock_fix:
            result = gen.build("predict churn", dataset_path=demo_csv, max_iterations=2)
        assert result["success"] is True
        assert mock_fix.call_count == 1
        # the error text handed to fix() must mention the real problem, not a generic traceback
        fix_call_payload = mock_fix.call_args[0][1]
        assert "features" in fix_call_payload.lower()
        if os.path.exists("models/best_model.pkl"):
            os.remove("models/best_model.pkl")

    def test_exhausting_all_retries_reports_failure_honestly(self, demo_csv):
        always_broken = 'raise RuntimeError("always fails")'
        gen = AICodeGenerator()
        with patch.object(gen, "generate", return_value=always_broken), \
             patch.object(gen, "fix", return_value=always_broken):
            result = gen.build("predict churn", dataset_path=demo_csv, max_iterations=1)
        assert result["success"] is False
        assert result["saved_model_path"] is None
        assert result["metrics"] == {}

    def test_groq_generation_failure_reported_clearly(self, demo_csv):
        gen = AICodeGenerator()
        with patch.object(gen, "generate", return_value=None):
            result = gen.build("predict churn", dataset_path=demo_csv)
        assert result["success"] is False
        assert result["code"] is None
        assert "GROQ_API_KEY" in result["error"] or "generation failed" in result["error"].lower()
