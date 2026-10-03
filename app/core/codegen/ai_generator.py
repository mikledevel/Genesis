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


class AICodeGenerator:
    MAX_ITERATIONS = 3
    EXEC_TIMEOUT = 45  # seconds

    GENERATE_PROMPT = """You are a senior ML engineer writing a COMPLETE, RUNNABLE Python script
for an AutoML platform (Genesis AI). Given a plain-language task description and info about
the dataset, write a single self-contained Python script that:
- loads the CSV at the exact path given
- profiles it briefly (prints shape, dtypes, missing values)
- preprocesses (handles missing values, encodes categoricals) - infer this from the actual
  columns you're told about, don't assume specific column names that weren't given
- splits train/test, trains 2-3 appropriate sklearn/xgboost models for the detected task
  (classification or regression - infer from the target column's nature)
- evaluates and prints metrics
- prints a line starting with "METRICS_JSON:" followed by a compact JSON object with the
  computed metrics (e.g. {"accuracy": 0.91, "f1": 0.88} for classification, or
  {"rmse": 123.4, "r2": 0.85} for regression), the detected task type ("task"), the
  model_type used ("model_type", e.g. "xgboost"), the target column name ("target"),
  and the list of feature column names ("features")
- CRITICAL: "features" in that JSON line MUST be computed programmatically from the actual
  training dataframe - e.g. `list(X_train.columns)` - never hand-typed. The saved model is
  only usable if this list exactly matches, in order, what the model was actually fit on
  (this is checked automatically after your script runs, and a mismatch fails the build).
- saves the best model to "models/" with pickle, filename "best_model.pkl"
- prints "PIPELINE_SUCCESS" as the very last line if it completes without error

Respond with ONLY the raw Python code, no markdown fences, no explanation before or after.
The code must run with: pandas, numpy, scikit-learn, xgboost (already installed)."""

    FIX_PROMPT = """The Python script below failed when actually executed. Fix it.
Respond with ONLY the corrected, complete raw Python code (same rules as before: no markdown
fences, prints a METRICS_JSON: line before PIPELINE_SUCCESS on success, with "features"
computed from the actual training dataframe's columns, not hand-typed). Keep everything that
wasn't the cause of the error unchanged - fix only what the traceback/validation error points to."""

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

    def _describe_dataset(self, dataset_path: str) -> str:
        """Give the model real column names/types instead of letting it guess - this is
        what fixes the old generator's blind 'last column is the target' assumption."""
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path, nrows=200)
            return json.dumps({
                "path": dataset_path,
                "columns": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "n_rows_sampled": len(df),
                "sample_row": df.iloc[0].to_dict() if len(df) else {},
            }, default=str)
        except Exception as e:
            return json.dumps({"path": dataset_path, "error": f"could not read dataset: {e}"})

    def generate(self, description: str, dataset_path: Optional[str] = None, target_column: Optional[str] = None) -> Optional[str]:
        context = {"task_description": description}
        if dataset_path and os.path.exists(dataset_path):
            context["dataset"] = self._describe_dataset(dataset_path)
        if target_column:
            context["target_column"] = target_column
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

    def _validate_saved_model(self, workdir: str, stdout: str) -> tuple:
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
        if not os.path.exists(model_path):
            return metrics_info, "models/best_model.pkl was not found - the script must save the trained model there."

        try:
            import pickle
            import numpy as np
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            model.predict(np.zeros((1, len(features))))
        except Exception as e:
            return metrics_info, (
                f"The saved model does not accept input shaped to the declared 'features' list "
                f"(length {len(features)}): {e}. 'features' in METRICS_JSON must exactly match, "
                f"in order, the columns the model was actually trained on - compute it as "
                f"list(X_train.columns) rather than typing it by hand."
            )
        return metrics_info, None

    def build(self, description: str, dataset_path: Optional[str] = None, target_column: Optional[str] = None,
              max_iterations: Optional[int] = None) -> Dict:
        """Full generate -> execute -> validate -> fix loop. A script that runs cleanly but
        produces a model whose declared features don't actually match what it was trained on
        is NOT treated as success - see _validate_saved_model - so this doesn't just check
        that the script exited 0, it checks the artifact it produced is actually usable.
        Returns the final code, whether it actually ran AND validated successfully, the
        execution log for transparency, and where it was saved."""
        max_iterations = self.MAX_ITERATIONS if max_iterations is None else max_iterations
        code = self.generate(description, dataset_path, target_column)
        if not code:
            return {"code": None, "success": False, "log": [], "error": "Groq generation failed - check GROQ_API_KEY"}

        workdir = tempfile.mkdtemp(prefix="genesis_codegen_")
        log = []
        success = False
        metrics_info = {}
        try:
            for i in range(max_iterations + 1):
                result = self.execute(code, workdir)
                validation_error = None
                if result["success"]:
                    metrics_info, validation_error = self._validate_saved_model(workdir, result["stdout"])
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
                saved_path = f"generated/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
                with open(saved_path, "w", encoding="utf-8") as f:
                    f.write(code)
                # bring any trained model file along so it's not lost with the temp dir
                models_dir = os.path.join(workdir, "models")
                if os.path.isdir(models_dir) and os.listdir(models_dir):
                    os.makedirs("models", exist_ok=True)
                    for fn in os.listdir(models_dir):
                        dest = os.path.join("models", fn)
                        shutil.copy2(os.path.join(models_dir, fn), dest)
                        if fn == "best_model.pkl":
                            saved_model_path = dest
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        return {"code": code, "success": success, "log": log, "iterations": len(log),
                "saved_path": saved_path if success else None,
                "saved_model_path": saved_model_path if success else None,
                "metrics": metrics_info if success else {}}
