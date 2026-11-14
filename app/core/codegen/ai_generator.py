"""
AICodeGenerator — generates real, runnable ML pipeline code via Groq, actually EXECUTES
it, and self-corrects on real errors (unlike the template-based CodeGenerator, which never
runs its own output).

Flow: description (+ optional dataset/target) -> Groq writes a complete Python script
   -> run it in a subprocess sandbox -> if it errors, feed the real traceback back to
   Groq and ask for a fix -> retry up to MAX_ITERATIONS -> return the final code + whether
   it actually ran successfully + captured stdout (metrics, etc).

SAFETY NOTE (dev-only, by design - see architecture discussion): this sandbox is a plain
subprocess with a timeout, isolated to its own temp working directory. That is enough to
catch accidental bugs during local development, but it is NOT strong isolation - it does
not block network access, does not cap CPU/memory, and should never be exposed to
untrusted/external users as-is. That hardening (Docker/gVisor, resource limits, no
network) is a separate, later piece of work once this core loop is proven to work.
"""
import os
import sys
import json
import subprocess
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Optional

from app.core import progress as _progress


def pd_is_numeric(series) -> bool:
    import pandas as pd
    return bool(pd.api.types.is_numeric_dtype(series)) and not pd.api.types.is_bool_dtype(series)


class AICodeGenerator:
    MAX_ITERATIONS = 3
    EXEC_TIMEOUT = 45  # seconds

    GENERATE_PROMPT = """You are a senior ML engineer writing a COMPLETE, RUNNABLE Python script
for an AutoML platform (Genesis AI). Given a plain-language task description and info about
the dataset, write a single self-contained Python script that:
- loads the training data from the file named in the dataset info (when it says "train.csv", read
  exactly that file from the current working directory - there is NO other data file in the sandbox)
- profiles it briefly (prints shape, dtypes, missing values)
- builds ONE scikit-learn Pipeline: a ColumnTransformer that imputes missing values and
  encodes categoricals / scales numerics INSIDE the pipeline, followed by the estimator. Infer the
  preprocessing from the actual columns you're told about - never assume column names that
  weren't given
- uses cross-validation (or a validation split carved out of the training file) to compare 2-3
  appropriate sklearn/xgboost models for the detected task (classification or regression - infer
  it from the target column's nature), then refits the best one on the whole training file
- prints the validation metrics
- prints a line starting with "METRICS_JSON:" followed by a compact JSON object with the
  computed metrics (e.g. {"accuracy": 0.91, "f1": 0.88} for classification, or
  {"rmse": 123.4, "r2": 0.85} for regression), the detected task type ("task"), the
  model_type used ("model_type", e.g. "xgboost"), the target column name ("target"),
  and the list of feature column names ("features")
- CRITICAL: "features" MUST be computed programmatically from the training dataframe - e.g.
  `list(X_train.columns)` - never hand-typed: the ORIGINAL column names fed into the Pipeline,
  in order, excluding the target.
- CRITICAL CONTRACT, checked by the platform on a HIDDEN TEST SET you never see: the saved
  object's predict() must accept a pandas DataFrame holding exactly those original feature
  columns (raw strings, dates, missing values) and return the ORIGINAL target labels (for
  classification the original class values, never integer codes). Do NOT preprocess outside the Pipeline.
- the metrics you print are informational only - the platform measures the official ones itself on
  the hidden test set, and a claimed number that you did not really achieve is detected and reported
- saves the fitted Pipeline to "models/best_model.pkl" with pickle, using only standard
  scikit-learn / xgboost classes (no custom classes, lambdas or helper functions inside it -
  they are rejected when the file is loaded)
- prints "PIPELINE_SUCCESS" as the very last line if it completes without error

Respond with ONLY the raw Python code, no markdown fences, no explanation before or after.
The code must run with: pandas, numpy, scikit-learn, xgboost (already installed).
If the dataset info contains "platform_notes", they describe what the platform already did to the
data (columns removed, hidden test set) - respect them."""

    FIX_PROMPT = """The Python script below failed when actually executed, or the platform's check of the
model it saved failed. Fix it. Respond with ONLY the corrected, complete raw Python code (same rules as
before: no markdown fences; read only "train.csv" if that is the file given; one Pipeline whose predict()
accepts a DataFrame of the ORIGINAL feature columns and returns the ORIGINAL labels; "features" computed from
the training dataframe's columns, not hand-typed; a METRICS_JSON: line before PIPELINE_SUCCESS). Keep everything
that wasn't the cause of the error unchanged - fix only what the traceback/validation error points to."""

    def __init__(self, db=None, model: Optional[str] = None, user_id: Optional[str] = None):
        self.db = db
        self.user_id = user_id
        self.groq_key = os.environ.get("GROQ_API_KEY", "")
        self._sandbox_backend = None  # resolved lazily, see _get_sandbox_backend()
        # Single source of truth for which Groq model writes this code: pull the catalog's
        # best "coding"-category model rather than hardcoding an id here that could silently
        # drift out of date with models_catalog.py (see agent.py MODEL_MAP / the model
        # selector UI, which both already read from the same catalog).
        self.MODEL = model or self._default_code_model()

    def _log_usage(self, call_site: str, completion=None, success: bool = True, error: str = None):
        """Best-effort internal Groq usage logging - see GenesisDB.log_groq_usage docstring
        for why this exists. Never lets a logging failure break the actual codegen call."""
        try:
            if self.db is None:
                from app.db.database import GenesisDB
                self.db = GenesisDB()
            usage = getattr(completion, "usage", None) if completion else None
            self.db.log_groq_usage(
                self.user_id, call_site, self.MODEL,
                prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
                total_tokens=getattr(usage, "total_tokens", None) if usage else None,
                success=success, error=error)
        except Exception:
            pass  # logging must never break the actual call

    @staticmethod
    def _default_code_model() -> str:
        try:
            from app.core.models_catalog import MODEL_CATALOG
            candidates = [s for s in MODEL_CATALOG.values() if "coding" in getattr(s, "categories", [])]
            if candidates:
                return max(candidates, key=lambda s: s.intelligence).id
        except Exception:
            pass
        return "openai/gpt-oss-120b"  # last-resort fallback if the catalog is ever unavailable


    def _get_sandbox_backend(self) -> str:
        """Docker gives real isolation (no network, memory/CPU caps, non-root) and is
        required for anyone other than the developer running this locally. Falls back to
        the weaker subprocess sandbox only when Docker genuinely isn't available AND the
        app is running with debug=True, so local single-user development keeps working
        without a Docker install. In production (debug=False) this raises instead of
        silently executing untrusted, model-generated code with no isolation - that
        combination (multi-tenant + no sandbox) is an arbitrary-code-execution hole, not
        a degraded feature."""
        if self._sandbox_backend is not None:
            return self._sandbox_backend
        from app.core.codegen.docker_sandbox import docker_available, ensure_image_built
        if docker_available() and ensure_image_built():
            self._sandbox_backend = "docker"
        else:
            from app.config import settings
            if not settings.debug:
                raise RuntimeError(
                    "Docker sandbox is unavailable and debug=False. Refusing to execute "
                    "AI-generated code without isolation in a non-dev environment. Install "
                    "and start Docker, or set debug=True only for trusted local development."
                )
            print("[AICodeGenerator] Docker not available - falling back to dev-only subprocess "
                  "sandbox (no network/resource isolation) because debug=True. Install Docker "
                  "Desktop before serving this to anyone other than yourself.")
            self._sandbox_backend = "subprocess"
        return self._sandbox_backend

    def _groq_kwargs(self) -> Dict:
        return {"reasoning_effort": "low"} if self.MODEL.startswith("openai/gpt-oss") else {}

    def _call_groq_code(self, system: str, user: str, max_tokens: int = 2000,
                        call_site: str = "codegen") -> Optional[str]:
        if not self.groq_key:
            return None
        try:
            from app.core.groq_client import call_llm
            completion = call_llm(
                self.groq_key,
                model=self.MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2,
                max_tokens=max_tokens,
                heavy=True,  # generates a full training script - can legitimately take a while
                groq_only_kwargs=self._groq_kwargs(),
            )
            self._log_usage(call_site, completion, success=True)
            code = completion.choices[0].message.content.strip()
            # strip markdown fences if the model added them anyway
            if code.startswith("```"):
                code = code.split("\n", 1)[1] if "\n" in code else code
                if code.endswith("```"):
                    code = code.rsplit("```", 1)[0]
            return code.strip()
        except Exception as e:
            print(f"[AICodeGenerator] error: {e}")
            self._log_usage(call_site, success=False, error=str(e))
            return None

    def _describe_dataset(self, dataset_path: str, display_path: Optional[str] = None) -> str:
        """Give the model real column names/types instead of letting it guess - this is
        what fixes the old generator's blind 'last column is the target' assumption."""
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path, nrows=200)
            return json.dumps({
                "path": display_path or dataset_path,
                "columns": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "n_rows_sampled": len(df),
                "sample_row": df.iloc[0].to_dict() if len(df) else {},
            }, default=str)
        except Exception as e:
            return json.dumps({"path": dataset_path, "error": f"could not read dataset: {e}"})

    def generate(self, description: str, dataset_path: Optional[str] = None, target_column: Optional[str] = None,
                 notes: Optional[List[str]] = None, display_path: Optional[str] = None) -> Optional[str]:
        context = {"task_description": description}
        if dataset_path and os.path.exists(dataset_path):
            context["dataset"] = self._describe_dataset(dataset_path, display_path)
        if target_column:
            context["target_column"] = target_column
        if notes:
            context["platform_notes"] = notes
        return self._call_groq_code(self.GENERATE_PROMPT, json.dumps(context, ensure_ascii=False),
                                    call_site="codegen_generate")

    def execute(self, code: str, workdir: str) -> Dict:
        """Run the generated script, using Docker isolation when available (see
        _get_sandbox_backend), falling back to a plain subprocess sandbox otherwise
        (dev-only isolation - see module docstring)."""
        if self._get_sandbox_backend() == "docker":
            from app.core.codegen.docker_sandbox import DockerSandbox
            return DockerSandbox().execute(code, workdir)
        return self._execute_subprocess(code, workdir)

    def _execute_subprocess(self, code: str, workdir: str) -> Dict:
        """Dev-grade fallback sandbox - see module docstring for why this alone isn't
        enough once other users are involved."""
        os.makedirs(os.path.join(workdir, "models"), exist_ok=True)
        script_path = os.path.join(workdir, "pipeline.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=self.EXEC_TIMEOUT,
            )
            success = proc.returncode == 0 and "PIPELINE_SUCCESS" in proc.stdout
            return {"success": success, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:], "returncode": proc.returncode}
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": f"Execution exceeded {self.EXEC_TIMEOUT}s timeout - likely an infinite loop or a dataset too large for this sandbox.", "returncode": None}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e), "returncode": None}

    def fix(self, code: str, error_output: str) -> Optional[str]:
        payload = json.dumps({"code": code, "error": error_output}, ensure_ascii=False)
        return self._call_groq_code(self.FIX_PROMPT, payload, call_site="codegen_fix")

    def _validate_saved_model(self, workdir: str, stdout: str, verification=None) -> tuple:
        """A script that exits 0 and prints PIPELINE_SUCCESS can still have declared a
        'features' list that doesn't match what the saved model actually expects - the same
        feature/column mismatch bug class already found and fixed in the built-in trainer
        (see engine/trainer.py). Since this code is LLM-generated we can't just trust its
        self-reported METRICS_JSON; verify it by actually loading the pickled model and
        calling .predict() with a vector shaped to the declared feature count.
        Returns (metrics_info, error) - error is None if validation passed."""
        metrics_info = {}
        found = False
        for line in stdout.splitlines():
            if line.startswith("METRICS_JSON:"):
                found = True
                try:
                    metrics_info = json.loads(line[len("METRICS_JSON:"):].strip())
                except json.JSONDecodeError:
                    return {}, "METRICS_JSON: line was present but not valid JSON."
                break
        if not found:
            return {}, "No METRICS_JSON: line found in stdout - required before PIPELINE_SUCCESS."

        features = metrics_info.get("features")
        if not features or not isinstance(features, list):
            return metrics_info, "METRICS_JSON's 'features' field is missing or not a list."

        model_path = os.path.join(workdir, "models", "best_model.pkl")
        if not os.path.realpath(model_path).startswith(os.path.realpath(workdir) + os.sep) or os.path.islink(model_path):
            return metrics_info, "The sandbox returned an unsafe model path"
        if not os.path.exists(model_path):
            return metrics_info, "models/best_model.pkl was not found - the script must save the trained model there."

        from app.config import settings
        from app.core.model_isolation import evaluate_model_file
        if verification is not None:
            try:
                checked = evaluate_model_file(model_path, verification, features, metrics_info.get("model_type","custom"), metrics_info.get("target"))
                from app.core.engine.verification import assess
                measured = checked["metrics"]
                self_reported = {k:v for k,v in metrics_info.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}
                measured["trust"] = assess(measured,verification,self_reported)
                measured["self_reported"] = self_reported
                measured["contract"] = checked["contract"]
                measured["data_checks"] = verification.checks
                self._last_verification = checked
            except Exception as error:
                return metrics_info, "Platform check of declared features failed: " + str(error)
        elif not settings.debug:
            return metrics_info, "A hidden test set is required before an external model can be accepted"
        else:
            try:
                import numpy as np
                from app.core.safe_pickle import safe_load
                with open(model_path,"rb") as handle:
                    model = safe_load(handle)
                model.predict(np.zeros((1,len(features))))
            except Exception as error:
                return metrics_info, "The saved model was rejected as unsafe or incompatible: " + str(error)
        return metrics_info, None

    def build(self, description: str, dataset_path: Optional[str] = None, target_column: Optional[str] = None,
              max_iterations: Optional[int] = None, keep_columns: Optional[List[str]] = None) -> Dict:
        """Full generate -> execute -> validate -> fix loop. A script that runs cleanly but
        produces a model whose declared features don't actually match what it was trained on
        is NOT treated as success - see _validate_saved_model - so this doesn't just check
        that the script exited 0, it checks the artifact it produced is actually usable.
        Returns the final code, whether it actually ran AND validated successfully, the
        execution log for transparency, and where it was saved."""
        max_iterations = self.MAX_ITERATIONS if max_iterations is None else max_iterations
        workdir = tempfile.mkdtemp(prefix="genesis_codegen_")

        # The platform - not the LLM's script - prepares the data: surprising duplicates / ID columns / probable
        # leakage are removed, and a HIDDEN TEST SET is split off. The sandbox only ever receives train.csv.
        plan, notes = None, []
        if dataset_path and os.path.exists(dataset_path):
            _progress.step("data", "active", "hiding a test set the AI never sees")
            try:
                from app.core.engine.verification import load_table, prepare_data
                plan = prepare_data(load_table(dataset_path), target_column, keep_columns=keep_columns)
                plan.train.to_csv(os.path.join(workdir, "train.csv"), index=False)
                notes = [c["message"] for c in plan.checks]
                if plan.verifiable:
                    notes.append(f"{len(plan.holdout)} of {plan.n_rows} rows are held out by the platform as a hidden "
                                 "test set; train.csv is the training set only.")
                _progress.step("data", "done", f"{len(plan.train)} rows to learn from, {len(plan.holdout)} hidden for verification")
            except Exception as e:
                print(f"[AICodeGenerator] could not prepare data for verification: {e}")
                plan = None
                _progress.step("data", "error", "could not prepare the data - falling back to unverified mode")
        verification = plan if (plan is not None and plan.verifiable) else None
        code = self.generate(description, os.path.join(workdir, "train.csv") if plan is not None else dataset_path,
                             target_column, notes=notes, display_path="train.csv" if plan is not None else None)
        if not code:
            shutil.rmtree(workdir, ignore_errors=True)
            return {"code": None, "success": False, "log": [], "error": "Groq generation failed - check GROQ_API_KEY"}

        log = []
        success = False
        metrics_info = {}
        self._last_verification = None
        try:
            for i in range(max_iterations + 1):
                result = self.execute(code, workdir)
                validation_error = None
                if result["success"]:
                    if verification is not None:
                        _progress.step("verify", "active", "scoring the saved model on the hidden test set")
                    self._last_verification = None
                    metrics_info, validation_error = self._validate_saved_model(workdir, result["stdout"], verification)
                    if verification is not None:
                        _progress.step("verify", "error" if validation_error else "done",
                                       (validation_error or "")[:120] if validation_error else
                                       f"accuracy/R² measured by the platform on {len(verification.holdout)} unseen rows")
                step_ok = result["success"] and not validation_error
                log.append({"iteration": i, "success": step_ok,
                            "stdout_tail": result["stdout"][-600:], "stderr_tail": result["stderr"][-600:],
                            "validation_error": validation_error})
                if step_ok:
                    success = True
                    break
                if i < max_iterations:
                    error_for_fix = validation_error or (result["stderr"] or result["stdout"])
                    fixed = self.fix(code, error_for_fix)
                    if fixed:
                        code = fixed

            saved_path = None
            saved_model_path = None
            if success:
                os.makedirs("generated", exist_ok=True)
                saved_path = f"generated/user_{self.user_id}_pipeline_{__import__('uuid').uuid4().hex}.py"
                with open(saved_path, "w", encoding="utf-8") as f:
                    f.write(code)
                # bring any trained model file along so it's not lost with the temp dir
                models_dir = os.path.join(workdir, "models")
                if os.path.isdir(models_dir) and os.listdir(models_dir):
                    model_destination = os.path.join("models", str(self.user_id or "system"), __import__("uuid").uuid4().hex)
                    os.makedirs(model_destination, exist_ok=True)
                    for fn in os.listdir(models_dir):
                        source = os.path.join(models_dir,fn)
                        if os.path.islink(source) or not os.path.realpath(source).startswith(os.path.realpath(workdir)+os.sep) or not os.path.isfile(source):
                            raise ValueError("The sandbox returned an unsafe output path")
                        dest = os.path.join(model_destination, fn)
                        shutil.copy2(os.path.join(models_dir, fn), dest)
                        if fn == "best_model.pkl":
                            saved_model_path = dest
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        ver = getattr(self, "_last_verification", None) if success else None
        merged = {}
        if success:
            merged = dict(metrics_info)                      # keeps the script's own keys ("features", "target", ...)
            if ver:
                merged.update({k: v for k, v in ver["metrics"].items() if k not in ("task",)})
        return {"code": code, "success": success, "log": log, "iterations": len(log),
                "saved_path": saved_path if success else None,
                "saved_model_path": saved_model_path if success else None,
                "metrics": merged,
                "verified": bool(ver), "verification": ver}
