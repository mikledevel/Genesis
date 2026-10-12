"""Tests for the "support bot + ML model" scenario: demo data, per-user model listing,
linking/unlinking a model on a bot, draft bots, and the confidence in bot predictions.
unittest-style so they run under pytest or `python -m unittest`. No sklearn needed."""
import os, sys, tempfile, types, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import seed_churn_demo as demo
from app.core.engine.registry import ModelRegistry


class DemoDataTests(unittest.TestCase):
    def test_shape_columns_and_determinism(self):
        a, b = demo.generate_churn_dataset(500, seed=1), demo.generate_churn_dataset(500, seed=1)
        self.assertEqual(list(a.columns), demo.FEATURES + [demo.TARGET])
        self.assertEqual(len(a), 500)
        self.assertTrue(a.equals(b))
        self.assertFalse(a.equals(demo.generate_churn_dataset(500, seed=2)))

    def test_realistic_rate_and_signal(self):
        d = demo.generate_churn_dataset(3000, seed=42)
        self.assertTrue(0.10 < d.churn.mean() < 0.40)
        self.assertTrue(set(d.churn.unique()) <= {0, 1})
        long_gap = d[d.days_since_last_order > 60].churn.mean()
        short_gap = d[d.days_since_last_order <= 60].churn.mean()
        self.assertGreater(long_gap, short_gap + 0.15)        # the pattern a model can learn
        self.assertTrue(d.select_dtypes(exclude=[np.number]).empty)  # all numeric: predict tool needs that

    def test_large_enough_to_avoid_small_dataset_warning(self):
        self.assertGreaterEqual(len(demo.generate_churn_dataset()), 1000)   # default size, well above the 100-row warning


class InTempDir(unittest.TestCase):
    def setUp(self):
        self._cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()


class RegistryAuthorTests(InTempDir):
    def test_lists_only_that_users_models_newest_first(self):
        reg = ModelRegistry(storage_dir="./reg")
        m1 = reg.register({"x": 1}, {"acc": .9}, "binary_classification", "xgboost", ["a"], "churn", metadata={"author": "u1"})
        m2 = reg.register({"x": 2}, {"acc": .8}, "regression", "linear", ["b"], "price", metadata={"author": "u2"})
        m3 = reg.register({"x": 3}, {"acc": .7}, "binary_classification", "logistic", ["c"], "churn", metadata={"author": "u1"})
        reg.register({"x": 4}, {}, "regression", "linear", ["d"], "legacy")          # no author at all
        ids = [m["model_id"] for m in reg.list_models_by_author("u1")]
        self.assertEqual(set(ids), {m1, m3})
        self.assertNotIn(m2, ids)
        self.assertEqual(reg.list_models_by_author("nobody"), [])
        self.assertEqual(len(reg.list_models_by_author("u1", limit=1)), 1)


class BotLinkTests(InTempDir):
    def _bot(self, publish):
        from app.core.bot_builder import BotBuilder
        return BotBuilder(user_id="owner").publish(
            {"name": "T", "system_prompt": "x", "description": "d", "greeting": "hi"},
            author="owner", publish=publish)

    def test_draft_is_unpublished_and_hidden_from_store(self):
        from app.core.agent_store import AgentStore
        draft, live = self._bot(False), self._bot(True)
        self.assertFalse(draft["published"])
        store = AgentStore()
        self.assertTrue(store.get_agent(live["id"])["published"])
        self.assertNotIn(draft["id"], [a["id"] for a in store.list_agents()])

    def test_link_unlink_and_ownership(self):
        from app.core.agent_store import AgentStore
        bot, store = self._bot(False), AgentStore()
        self.assertEqual(store.update_agent(bot["id"], {"linked_model_id": "m1"}, "owner")["linked_model_id"], "m1")
        self.assertIsNone(store.update_agent(bot["id"], {"linked_model_id": "m2"}, "intruder"))
        self.assertEqual(store.get_agent(bot["id"])["linked_model_id"], "m1")          # unchanged
        self.assertIsNone(store.update_agent(bot["id"], {"linked_model_id": None}, "owner")["linked_model_id"])


try:
    import sklearn  # noqa: F401
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


@unittest.skipUnless(HAVE_SKLEARN, "scikit-learn not installed")
class PredictConfidenceTests(InTempDir):
    """Uses REAL fitted sklearn models: registry.get_model now loads through the restricted
    unpickler, which (correctly) refuses classes it doesn't know, such as test stubs."""

    FEATS = ["score", "other"]

    def _fit(self, model):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 2)) * 20 + 50
        y = (X[:, 0] > 50).astype(int)
        return model.fit(X, y)

    def _run(self, model, features):
        from app.core.agent import GenesisAgent
        from app.core.agent_store import AgentStore
        from app.core.bot_builder import BotBuilder
        reg = ModelRegistry(storage_dir="./reg")
        mid = reg.register(model, {}, "binary_classification", "sk", self.FEATS, "churn", metadata={"author": "owner"})
        bot = BotBuilder(user_id="owner").publish({"name": "T", "system_prompt": "x"}, author="owner", publish=False)
        AgentStore().update_agent(bot["id"], {"linked_model_id": mid}, "owner")
        return GenesisAgent._do_bot_predict(types.SimpleNamespace(registry=reg), bot["id"], {"features": features})

    def test_includes_confidence_for_classifier(self):
        from sklearn.linear_model import LogisticRegression
        model = self._fit(LogisticRegression(max_iter=500))
        r = self._run(model, {"score": 95, "other": 50})
        expected = round(float(np.max(model.predict_proba([[95, 50]])[0])), 4)
        self.assertTrue(r.success)
        self.assertEqual(r.data["prediction"], 1)
        self.assertEqual(r.data["confidence"], expected)
        self.assertIn(f"{round(expected * 100)}%", r.message)
        self.assertEqual(self._run(model, {"score": 5, "other": 50}).data["prediction"], 0)

    def test_model_without_predict_proba_still_works(self):
        from sklearn.linear_model import LinearRegression      # regressors have no predict_proba
        r = self._run(self._fit(LinearRegression()), {"score": 60, "other": 50})
        self.assertTrue(r.success)
        self.assertIsNone(r.data["confidence"])
        self.assertNotIn("confidence", r.message)

    def test_missing_feature_is_asked_for(self):
        from sklearn.linear_model import LogisticRegression
        r = self._run(self._fit(LogisticRegression(max_iter=500)), {"score": 1})
        self.assertFalse(r.success)
        self.assertIn("other", r.message)


if __name__ == "__main__":
    unittest.main()
