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
              user_id: Optional[str] = None) -> Dict:
        """Generate, execute, and self-correct a full ML pipeline for the given dataset.
        Returns everything needed for a preview screen: whether it worked, the metrics,
        the code, and (if successful) the path to the trained model file ready to register."""
        result = AICodeGenerator(user_id=user_id).build(description, dataset_path=dataset_path, target_column=target_column)
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
        }

    def register(self, saved_model_path: str, metrics: Dict, author_user_id: str) -> Optional[str]:
        """Load the pickled model produced by build() and register it in ModelRegistry.
        Returns the new model_id, or None if the model file couldn't be loaded."""
        from app.core.engine.registry import ModelRegistry
        try:
            with open(saved_model_path, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            print(f"[ModelBuilder] could not load model file: {e}")
            return None
        reg = ModelRegistry()
        return reg.register(
            model=model,
            metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            task=metrics.get("task", "binary_classification"),
            model_type=metrics.get("model_type", "unknown"),
            features=metrics.get("features", []),
            target=metrics.get("target", ""),
            metadata={"author": author_user_id, "built_via": "ModelBuilder"},
        )
