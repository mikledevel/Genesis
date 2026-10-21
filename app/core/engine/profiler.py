import re
import pandas as pd, numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

class DatasetProfile:
    def __init__(self):
        self.file_name = self.suggested_target = self.suggested_task = self.creation_time = ""
        self.file_size_mb = self.task_confidence = 0.0
        self.rows = self.columns = 0
        self.column_info: Dict = {}
        self.missing_summary: Dict = {}
        self.feature_columns: List = []
        self.warnings: List = []
        self.target_candidates: List = []
        self.needs_target_confirmation = False

class DatasetProfiler:
    UNIQUE_THRESHOLD_CLASSIFICATION = 0.05
    UNIQUE_THRESHOLD_CATEGORICAL = 0.10
    MAX_CLASSIFICATION_CLASSES = 100

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.profile = DatasetProfile()

    @staticmethod
    def _is_numeric(s): return pd.api.types.is_numeric_dtype(s.dtype)

    def load_data(self, path_str: str) -> pd.DataFrame:
        p = Path(path_str)
        if not p.exists(): raise FileNotFoundError(f"Not found: {path_str}")
        self.profile.file_name = p.name
        self.profile.file_size_mb = p.stat().st_size/(1024*1024)
        l = {".csv": pd.read_csv, ".json": pd.read_json, ".parquet": pd.read_parquet,
             ".xlsx": pd.read_excel}      # chat uploads may be Excel files (needs openpyxl)
        s = p.suffix.lower()
        if s not in l: raise ValueError(f"Format: {s}")
        self.df = l[s](path_str)
        self.profile.rows, self.profile.columns = self.df.shape
        self.profile.creation_time = datetime.now().isoformat()
        return self.df

    def analyze_columns(self):
        if self.df is None: raise ValueError("Not loaded")
        for c in self.df.columns:
            s = self.df[c]
            info = {"dtype":str(s.dtype),"total":len(s),"missing":int(s.isna().sum()),"missing_pct":round(s.isna().mean()*100,2),"unique":int(s.nunique()),"unique_pct":round(s.nunique()/len(s)*100,2)}
            if self._is_numeric(s):
                info.update({"mean":round(s.mean(),4) if not s.isna().all() else None,"std":round(s.std(),4) if not s.isna().all() else None,"min":round(s.min(),4) if not s.isna().all() else None,"max":round(s.max(),4) if not s.isna().all() else None})
                info["suggested_type"] = "binary" if info["unique"]==2 else ("categorical_numeric" if info["unique_pct"]<self.UNIQUE_THRESHOLD_CATEGORICAL*100 else "continuous")
            else:
                info["suggested_type"] = "categorical"
                info["top_values"] = {str(k):v for k,v in s.value_counts().head(10).to_dict().items()}
            self.profile.column_info[c] = info
        return self.profile.column_info

    def analyze_missing(self):
        if self.df is None: raise ValueError("Not loaded")
        t,m = self.df.size, self.df.isna().sum().sum()
        self.profile.missing_summary = {"total_cells":int(t),"total_missing":int(m),"missing_pct":round(m/t*100,2),"columns_with_missing":int((self.df.isna().sum()>0).sum())}
        return self.profile.missing_summary

    # Target ranking. The old rule ("the last numeric column") was right only when the file happened to
    # end with the answer; on 12 real-dataset layouts it picked the right column 4 times. Now every
    # column is scored on its NAME (target/label/churn/price/цель/отток...), its SHAPE (a binary or
    # low-cardinality column is a likelier label), its POSITION (a weak tie-breaker) and is penalised
    # if it looks like an ID, a date, free text, a constant or mostly empty.
    _STRONG = re.compile(r"^(target|label|labels|class|y|outcome|result|response|churn|churned|default|fraud|survived|"
                         r"survival|diagnosis|species|status|price|cost|salary|revenue|sales|score|rating|"
                         r"цель|целевая|таргет|метка|класс|отток|результат|цена|стоимость|зарплата|выручка|продажи|оценка)$", re.I)
    _CONTAINS = re.compile(r"(target|label|churn|default|fraud|surviv|outcome|price|salary|revenue|diagnos|"
                           r"цель|таргет|отток|цена|зарплат)", re.I)
    _FEATURE_LIKE = re.compile(r"(^|[_\s-])(id|idx|index|uuid|guid|name|email|phone|url|date|time|timestamp|zip|postal|"
                               r"lat|lon|lng)([_\s-]|$)|^unnamed", re.I)

    def rank_targets(self, limit: int = 4) -> List[Dict]:
        from app.core.engine.transformers import detect_datetime_columns, detect_text_columns, looks_like_id
        if self.df is None: raise ValueError("Not loaded")
        df, cols = self.df, list(self.df.columns)
        dates = set(detect_datetime_columns(df, [c for c in cols if not self._is_numeric(df[c])]))
        texts = set(detect_text_columns(df, [c for c in cols if not self._is_numeric(df[c]) and c not in dates]))
        out = []
        for i, c in enumerate(cols):
            s = df[c]; u = s.nunique(dropna=True); reasons = []; score = 0.0
            if c in dates: continue                                    # a date is a feature, not an answer
            if c in texts: continue
            if u <= 1: continue                                        # constant
            if s.isna().mean() > 0.5: continue
            if looks_like_id(s, c): continue
            if self._STRONG.match(str(c).strip()): score += 3.0; reasons.append("the name says it is the answer")
            elif self._CONTAINS.search(str(c)): score += 1.5; reasons.append("the name hints it is the answer")
            if self._FEATURE_LIKE.search(str(c)): score -= 2.0
            if i == len(cols) - 1: score += 1.0; reasons.append("it is the last column")
            elif i == 0: score += 0.3
            if u == 2: score += 1.0; reasons.append("it has exactly two values (a yes/no label)")
            elif u <= 20 and (not self._is_numeric(s) or s.nunique() / len(s) < 0.05): score += 0.6; reasons.append(f"it has only {u} distinct values")
            task = self._task_for(s)[0]
            out.append({"column": c, "task": task, "score": round(score, 2), "reason": "; ".join(reasons) or "plain numeric column"})
        out.sort(key=lambda d: -d["score"])
        return out[:limit]

    def _task_for(self, s):
        u, up = s.nunique(), s.nunique() / len(s)
        if self._is_numeric(s):
            if u <= 2: return "binary_classification", 0.95
            if up < self.UNIQUE_THRESHOLD_CLASSIFICATION: return "multiclass_classification", min(0.9, 1.0 - up)
            return "regression", min(0.95, up)
        if u <= 2: return "binary_classification", 0.9
        if u <= self.MAX_CLASSIFICATION_CLASSES: return "multiclass_classification", min(0.85, 1.0 - up)
        return "regression", 0.5

    def suggest_task(self, target_col: Optional[str] = None):
        if self.df is None: raise ValueError("Not loaded")
        self.profile.target_candidates = self.rank_targets()
        top = self.profile.target_candidates
        if target_col is None:
            target_col = top[0]["column"] if top else (self.df.select_dtypes(include=[np.number]).columns.tolist() or list(self.df.columns))[-1]
            # Ask the user instead of guessing when nothing stands out: weak winner or a near tie.
            margin = (top[0]["score"] - top[1]["score"]) if len(top) > 1 else 9
            self.profile.needs_target_confirmation = (not top) or top[0]["score"] < 2.0 or margin < 1.0
        else:
            self.profile.needs_target_confirmation = False
        task, conf = self._task_for(self.df[target_col])
        self.profile.suggested_task = task; self.profile.suggested_target = target_col
        self.profile.feature_columns = [c for c in self.df.columns if c!=target_col]
        self.profile.task_confidence = round(conf,2)
        return task, target_col, conf

    def get_summary(self):
        return {"file_info":{"name":self.profile.file_name,"size_mb":round(self.profile.file_size_mb,2),"rows":self.profile.rows,"columns":self.profile.columns},"missing_summary":self.profile.missing_summary,"columns":{c:{"dtype":i["dtype"],"missing_pct":i["missing_pct"],"unique":i["unique"],"suggested_type":i["suggested_type"]} for c,i in self.profile.column_info.items()},"suggested_ml_task":{"task":self.profile.suggested_task,"target_column":self.profile.suggested_target,"confidence":self.profile.task_confidence,"needs_confirmation":self.profile.needs_target_confirmation,"candidates":self.profile.target_candidates},"warnings":self.profile.warnings}

    def run_full_profile(self, path_str: str, target_col: Optional[str] = None):
        self.load_data(path_str); self.analyze_columns(); self.analyze_missing(); self.suggest_task(target_col)
        return self.profile
