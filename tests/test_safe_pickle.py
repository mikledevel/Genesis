"""Tests for app/core/safe_pickle.py and its use at the untrusted-model entry points.
unittest-style (runs under pytest too). sklearn-dependent tests are skipped if sklearn is
missing; xgboost/lightgbm tests only run where those are installed (they are in requirements.txt,
so they DO run on a normal project install - please run them there)."""
import io, os, pickle, sys, tempfile, types, unittest, warnings
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from app.core.safe_pickle import safe_load, safe_loads, UnsafePickleError

warnings.filterwarnings("ignore")

try:
    import sklearn  # noqa: F401
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False
try:
    import xgboost  # noqa: F401
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False


class _Exploit:
    """Pickles to 'call os.system(cmd)' - the textbook malicious model file."""
    def __init__(self, fn, *args):
        self.fn, self.args = fn, args

    def __reduce__(self):
        return (self.fn, self.args)


def exploit_bytes(fn, *args) -> bytes:
    return pickle.dumps(_Exploit(fn, *args))


class MaliciousPayloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.marker = os.path.join(self.tmp.name, "PWNED")

    def tearDown(self):
        self.tmp.cleanup()

    def _assert_blocked_and_not_executed(self, payload: bytes):
        with self.assertRaises(UnsafePickleError):
            safe_loads(payload)
        self.assertFalse(os.path.exists(self.marker), "payload EXECUTED despite being blocked")

    def test_os_system_blocked(self):
        self._assert_blocked_and_not_executed(exploit_bytes(os.system, f"touch {self.marker}"))

    def test_subprocess_blocked(self):
        import subprocess
        self._assert_blocked_and_not_executed(exploit_bytes(subprocess.call, ["touch", self.marker]))

    def test_eval_exec_blocked(self):
        code = f"open({self.marker!r}, 'w').close()"
        self._assert_blocked_and_not_executed(exploit_bytes(eval, code))
        self._assert_blocked_and_not_executed(exploit_bytes(exec, code))

    def test_getattr_and_import_gadgets_blocked(self):
        self._assert_blocked_and_not_executed(exploit_bytes(getattr, "abc", "upper"))
        self._assert_blocked_and_not_executed(exploit_bytes(__import__, "os"))

    def test_open_blocked(self):
        self._assert_blocked_and_not_executed(exploit_bytes(open, self.marker, "w"))

    def test_numpy_runstring_gadget_blocked(self):
        # numpy ships a function that exec()s a string - denied by module AND by "classes only"
        payload = (b"cnumpy.testing._private.utils\nrunstring\n(S\"open(%r,'w').close()\"\n(dtR." % self.marker.encode())
        self._assert_blocked_and_not_executed(payload)

    def test_function_from_allowed_library_blocked(self):
        # an allowed library does not make its FUNCTIONS allowed (only classes + a tiny list)
        self._assert_blocked_and_not_executed(exploit_bytes(np.load, self.marker))

    def test_error_names_the_blocked_global(self):
        with self.assertRaises(UnsafePickleError) as cm:
            safe_loads(exploit_bytes(os.system, "echo hi"))
        self.assertIn("system", str(cm.exception))

    def test_safe_load_file_object_variant(self):
        with self.assertRaises(UnsafePickleError):
            safe_load(io.BytesIO(exploit_bytes(os.system, f"touch {self.marker}")))
        self.assertFalse(os.path.exists(self.marker))


class BenignDataTests(unittest.TestCase):
    def test_plain_python_and_numpy_roundtrip(self):
        obj = {"a": [1, 2.5, "x", (3, 4), {5, 6}, frozenset({7}), None, True, b"by", 2 + 3j],
               "arr": np.arange(12, dtype=np.float64).reshape(3, 4), "i": np.int64(5), "f": np.float32(1.5)}
        out = safe_loads(pickle.dumps(obj))
        self.assertEqual(out["a"], obj["a"])
        self.assertTrue(np.array_equal(out["arr"], obj["arr"]))
        self.assertEqual(out["i"], 5)

    def test_structured_and_object_arrays(self):
        s = np.zeros(3, dtype=[("a", "<i4"), ("b", "<f8")])
        self.assertTrue(np.array_equal(safe_loads(pickle.dumps(s)), s))
        o = np.array(["x", "y"], dtype=object)
        self.assertTrue(np.array_equal(safe_loads(pickle.dumps(o)), o))


@unittest.skipUnless(HAVE_SKLEARN, "scikit-learn not installed")
class RealModelTests(unittest.TestCase):
    """Real fitted estimators must survive the restricted loader with IDENTICAL predictions -
    otherwise the security fix would silently break prediction for legitimate models."""

    @classmethod
    def setUpClass(cls):
        import seed_churn_demo as demo
        df = demo.generate_churn_dataset(400, seed=3)
        cls.X, cls.y, cls.reg_y = df[demo.FEATURES], df[demo.TARGET], df["avg_order_value"]

    def _roundtrip(self, model, y=None):
        model.fit(self.X, self.y if y is None else y)
        loaded = safe_loads(pickle.dumps(model))
        self.assertTrue(np.array_equal(loaded.predict(self.X[:60]), model.predict(self.X[:60])))

    def test_every_model_type_the_platform_trains(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                                      HistGradientBoostingClassifier, HistGradientBoostingRegressor)
        self._roundtrip(Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=500))]))
        self._roundtrip(Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]), self.reg_y)
        self._roundtrip(RandomForestClassifier(n_estimators=15, random_state=0))
        self._roundtrip(RandomForestRegressor(n_estimators=15, random_state=0), self.reg_y)
        self._roundtrip(HistGradientBoostingClassifier(random_state=0))
        self._roundtrip(HistGradientBoostingRegressor(random_state=0), self.reg_y)

    def test_other_common_estimators_in_generated_pipelines(self):
        from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.naive_bayes import GaussianNB
        for m in (GradientBoostingClassifier(n_estimators=10), ExtraTreesClassifier(n_estimators=10),
                  AdaBoostClassifier(), KNeighborsClassifier(), KNeighborsClassifier(algorithm="ball_tree"),
                  SVC(probability=True), DecisionTreeClassifier(), GaussianNB()):
            with self.subTest(model=type(m).__name__):
                self._roundtrip(m)

    def test_function_transformer_and_column_transformer(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        Xn = self.X.values
        for pipe in (make_pipeline(FunctionTransformer(np.log1p), LogisticRegression(max_iter=300)),
                     make_pipeline(ColumnTransformer([("s", StandardScaler(), [0, 1]), ("p", "passthrough", [2, 3, 4, 5])]),
                                   LogisticRegression(max_iter=300))):
            pipe.fit(Xn, self.y)
            loaded = safe_loads(pickle.dumps(pipe))
            self.assertTrue(np.array_equal(loaded.predict(Xn[:50]), pipe.predict(Xn[:50])))

    def test_platform_label_decoding_wrapper(self):
        from sklearn.linear_model import LogisticRegression
        from app.core.engine.trainer import _LabelDecodingModel
        m = _LabelDecodingModel(LogisticRegression(max_iter=300).fit(self.X, self.y), ["stay", "leave"])
        loaded = safe_loads(pickle.dumps(m))
        self.assertEqual(list(loaded.predict(self.X[:20])), list(m.predict(self.X[:20])))

    def test_model_trained_by_the_real_trainer_loads(self):
        import seed_churn_demo as demo
        from app.core.engine.trainer import DatasetTrainer
        with tempfile.TemporaryDirectory() as d:
            res = DatasetTrainer(model_dir=d).train(self.X.assign(**{demo.TARGET: self.y}), demo.TARGET,
                                                   "binary_classification", model_type="auto")
            with open(res.model_path, "rb") as f:
                model = safe_load(f)
            self.assertEqual(len(model.predict(self.X[:5])), 5)

    def test_sklearn_gadget_function_still_blocked(self):
        from sklearn.utils import check_array
        with self.assertRaises(UnsafePickleError):
            safe_loads(exploit_bytes(check_array, [[1]]))


@unittest.skipUnless(HAVE_XGB, "xgboost not installed here - runs on a normal project install")
class XGBoostTests(unittest.TestCase):
    def test_xgboost_models_load_with_identical_predictions(self):
        from xgboost import XGBClassifier, XGBRegressor
        import seed_churn_demo as demo
        df = demo.generate_churn_dataset(400, seed=3)
        X, y = df[demo.FEATURES], df[demo.TARGET]
        for m, target in ((XGBClassifier(n_estimators=20, verbosity=0), y),
                          (XGBRegressor(n_estimators=20, verbosity=0), df["avg_order_value"])):
            m.fit(X, target)
            loaded = safe_loads(pickle.dumps(m))
            self.assertTrue(np.allclose(loaded.predict(X[:50]), m.predict(X[:50])))


class EntryPointTests(unittest.TestCase):
    """The three places that load untrusted model files must refuse a malicious one."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.marker = os.path.join(self.tmp.name, "PWNED")
        self.evil = exploit_bytes(os.system, f"touch {self.marker}")
        self._cwd = os.getcwd()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self.tmp.cleanup()

    def test_sandbox_output_validation_refuses_malicious_pickle(self):
        from app.core.codegen.ai_generator import AICodeGenerator
        os.makedirs("work/models")
        with open("work/models/best_model.pkl", "wb") as f:
            f.write(self.evil)
        stdout = 'METRICS_JSON: {"features": ["a", "b"]}\nPIPELINE_SUCCESS'
        info, err = AICodeGenerator._validate_saved_model(types.SimpleNamespace(), "work", stdout)
        self.assertIsNotNone(err)
        self.assertIn("unsafe", err)                    # tells the code-fixing loop the real reason
        self.assertFalse(os.path.exists(self.marker))

    def test_model_builder_register_refuses_malicious_pickle(self):
        from app.core.model_builder import ModelBuilder
        with open("evil.pkl", "wb") as f:
            f.write(self.evil)
        self.assertIsNone(ModelBuilder().register("evil.pkl", {"task": "x"}, "u1"))
        self.assertFalse(os.path.exists(self.marker))

    def test_registry_get_model_refuses_tampered_file(self):
        from app.core.engine.registry import ModelRegistry
        reg = ModelRegistry(storage_dir="./reg")
        mid = reg.register({"ok": 1}, {}, "regression", "stub", ["a"], "t", metadata={"author": "u1"})
        self.assertEqual(reg.get_model(mid), {"ok": 1})                 # legit model still loads
        with open(reg.get_record(mid)["model_path"], "wb") as f:       # attacker swaps the file
            f.write(self.evil)
        with self.assertRaises(UnsafePickleError):
            reg.get_model(mid)
        self.assertFalse(os.path.exists(self.marker))


if __name__ == "__main__":
    unittest.main()
