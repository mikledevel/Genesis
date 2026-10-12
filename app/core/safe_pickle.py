"""Safer loading of pickled ML models.

THE PROBLEM
    pickle.load() on a file runs whatever code the file asks it to. Model files here come
    from places an attacker can influence: LLM-generated pipelines executed in the Docker
    sandbox (a poisoned dataset or a hostile "describe your model" prompt can make that code
    write ANY bytes to models/best_model.pkl), and, via the marketplace, other users' models.
    The sandbox has no network, but the HOST then unpickles the result - outside the sandbox.

WHAT THIS DOES
    safe_load()/safe_loads() use an Unpickler whose find_class() only resolves:
      * a short explicit list of harmless builtins / numpy reconstruction helpers,
      * CLASSES (never plain functions) from the ML libraries a model legitimately needs
        (sklearn, xgboost, lightgbm, numpy, scipy.sparse), minus a deny-list of modules that
        can run code or touch files (numpy.f2py, numpy.testing, sklearn.externals, ...).
    So the classic payloads - os.system, subprocess, eval/exec, getattr tricks - are refused
    BEFORE anything runs, with an error naming the blocked global.

WHAT THIS DOES NOT DO (be honest about it)
    It is defense in depth, not a proof of safety: Python's own docs say unpickling untrusted
    data can never be made fully safe, and a future gadget inside an allowed class could still
    be abused. Treat it as raising the bar a lot, and keep the real boundary (the sandbox) too.
    Do not use this to load files from sources you have no relationship with.

If a legitimate model is refused, the error names the blocked global; add it to the
allowlists below only after checking it is a pure data/reconstruction helper.
"""
import io
import pickle
from typing import Any, BinaryIO

__all__ = ["UnsafePickleError", "safe_load", "safe_loads"]


class UnsafePickleError(pickle.UnpicklingError):
    """The pickle references something outside the allowlist; nothing was executed."""


# Exact (module, name) pairs that are always allowed.
_SAFE_BUILTINS = {"set", "frozenset", "dict", "list", "tuple", "int", "float", "str", "bytes",
                  "bytearray", "complex", "bool", "slice", "range", "object"}
_ALLOWED_EXACT = (
    {("builtins", n) for n in _SAFE_BUILTINS}
    | {("collections", "OrderedDict"), ("collections", "defaultdict"), ("collections", "Counter"),
       ("copyreg", "_reconstructor"), ("_codecs", "encode"),
       # numpy 1.x and 2.x spellings of the array/scalar reconstruction helpers
       ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
       ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
       ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
       ("numpy.random._pickle", "__randomstate_ctor"), ("numpy.random._pickle", "__generator_ctor"),
       ("numpy.random._pickle", "__bit_generator_ctor"),
       # Cython-generated helpers that rebuild ONE compiled class from its saved state; they only
       # create an instance of an (already allowlisted) class, no code from the pickle runs.
       ("numpy.random.bit_generator", "__pyx_unpickle_SeedSequence"),
       ("sklearn.neighbors._kd_tree", "newObj"), ("sklearn.neighbors._ball_tree", "newObj"),
       ("sklearn.metrics._dist_metrics", "newObj"),
       # the platform's own label-decoding wrapper (engine/trainer.py)
       ("app.core.engine.trainer", "_LabelDecodingModel")}
    # pure element-wise numpy functions that FunctionTransformer-based pipelines commonly use
    | {("numpy", n) for n in ("log", "log1p", "exp", "expm1", "sqrt", "abs", "absolute", "square",
                              "sign", "negative", "reciprocal", "isnan", "isfinite", "clip",
                              "maximum", "minimum", "round", "floor", "ceil", "sin", "cos", "tan")}
)

# Libraries whose CLASSES may be instantiated while loading a model.
_ALLOWED_CLASS_PREFIXES = ("sklearn.", "xgboost.", "lightgbm.", "numpy.", "scipy.sparse.")
# "_loss" is how scikit-learn's compiled loss classes (used by GradientBoosting / HistGradientBoosting)
# name their module inside a pickle.
_ALLOWED_CLASS_MODULES = {"numpy", "sklearn", "xgboost", "lightgbm", "_loss"}

# Modules never allowed even if they sit under an allowed prefix (code execution / file / shell).
_DENY_PREFIXES = ("numpy.f2py", "numpy.testing", "numpy.distutils", "numpy.ctypeslib",
                  "numpy.lib.npyio", "numpy.lib._npyio", "numpy.__config__", "numpy.show",
                  "sklearn.externals", "sklearn.utils._testing", "sklearn.datasets",
                  "xgboost.testing", "xgboost.dask", "xgboost.spark", "xgboost.tracker")


def _under(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _ALLOWED_EXACT:
            return super().find_class(module, name)
        if any(_under(module, p) for p in _DENY_PREFIXES):
            raise UnsafePickleError(f"Blocked unpickling of '{module}.{name}' (denied module).")
        if module in _ALLOWED_CLASS_MODULES or module.startswith(_ALLOWED_CLASS_PREFIXES):
            obj = super().find_class(module, name)
            if isinstance(obj, type):
                return obj
            raise UnsafePickleError(
                f"Blocked unpickling of '{module}.{name}': only classes are allowed from "
                f"'{module.split('.')[0]}', not functions or other callables.")
        raise UnsafePickleError(f"Blocked unpickling of '{module}.{name}' (module not allowed).")


def safe_loads(data: bytes) -> Any:
    """Like pickle.loads, but restricted to the allowlist above."""
    return _RestrictedUnpickler(io.BytesIO(data)).load()


def safe_load(file: BinaryIO) -> Any:
    """Like pickle.load, but restricted to the allowlist above."""
    return _RestrictedUnpickler(file).load()
