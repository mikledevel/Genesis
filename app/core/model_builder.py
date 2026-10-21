"""
ModelBuilder - lets a regular (non-technical) user describe an ML model in plain
language and get back a real, tested, trained model registered in ModelRegistry,
ready to publish to the Agent Store (free) or Marketplace (paid).

This is the ML-model counterpart to BotBuilder: same generate -> test -> self-correct
shape, but the "test" here is genuinely running the generated pipeline against real
data (via AICodeGenerator + DockerSandbox) rather than asking test questions of a chat
persona.
"""
import pickle
from typing import Dict, Optional

from app.core.codegen.ai_generator import AICodeGenerator


class ModelBuilder:
    def build(self, description: str, dataset_path: str, target_column: Optional[str] = None,
              user_id: Optional[str] = None, keep_columns: Optional[list] = None) -> Dict:
        """Generate, execute, and self-correct a full ML pipeline for the given dataset.
        Returns everything needed for a preview screen: whether it worked, the metrics,
        the code, and (if successful) the path to the trained model file ready to register."""
        gen_kwargs = {"keep_columns": keep_columns} if keep_columns else {}
        result = AICodeGenerator(user_id=user_id).build(description, dataset_path=dataset_path, target_column=target_column, **gen_kwargs)
        if not result.get("code"):
            return {"success": False, "error": result.get("error", "Generation failed")}
        return {
            "success": result["success"],
            "code": result["code"],
            "metrics": result.get("metrics", {}),
            "log": result["log"],
            "iterations": result["iterations"],
            "saved_path": result.get("saved_path"),
            "saved_model_path": result.get("saved_model_path"),
            "verified": result.get("verified", False),
            "verification": result.get("verification"),
        }

    def register(self, saved_model_path: str, metrics: Dict, author_user_id: str, verification: Optional[Dict] = None,
                 registry=None) -> Optional[str]:
        """Load the pickled model produced by build() and register it in ModelRegistry.
        Returns the new model_id, or None if the model file couldn't be loaded."""
        from app.core.engine.registry import ModelRegistry
        try:
            # This file was written by LLM-generated code inside the sandbox, so it is untrusted:
            # load it with the restricted unpickler, never a plain pickle.load on the host.
            from app.core.safe_pickle import safe_load
            with open(saved_model_path, "rb") as f:
                model = safe_load(f)
        except Exception as e:
            print(f"[ModelBuilder] could not load model file: {e}")
            return None
        reg = registry or ModelRegistry()
        metadata = {"author": author_user_id, "built_via": "ModelBuilder"}
        v = verification
        if v:
            # Measured by the platform on a hidden test set. If the model accepts the ORIGINAL columns it is
            # registered in the raw-input format, so predict paths can take raw strings / dates / NaN.
            metadata.update({"verified": True, "trust": v["metrics"].get("trust"), "data_checks": v["metrics"].get("data_checks", [])})
            if v["contract"] == "raw_dataframe":
                from app.core.engine.predict_utils import RAW_FORMAT
                metadata.update({"input_format": RAW_FORMAT, "input_schema": v["input_schema"]})
        return reg.register(
            model=model,
            metrics={k: val for k, val in metrics.items() if isinstance(val, (int, float)) and not isinstance(val, bool)},
            task=(v or {}).get("task") or metrics.get("task", "binary_classification"),
            model_type=(v or {}).get("model_type") or metrics.get("model_type", "unknown"),
            features=(v or {}).get("features") or metrics.get("features", []),
            target=(v or {}).get("target") or metrics.get("target", ""),
            metadata=metadata,
        )
