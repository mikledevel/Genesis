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
        l = {".csv": pd.read_csv, ".json": pd.read_json, ".parquet": pd.read_parquet}
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

    def suggest_task(self, target_col: Optional[str] = None):
        if self.df is None: raise ValueError("Not loaded")
        if target_col is None:
            nc = self.df.select_dtypes(include=[np.number]).columns.tolist()
            target_col = nc[-1] if nc else self.df.columns[-1]
        s = self.df[target_col]; u, up = s.nunique(), s.nunique()/len(s)
        if self._is_numeric(s):
            if u<=2: task, conf = "binary_classification", 0.95
            elif up<self.UNIQUE_THRESHOLD_CLASSIFICATION: task, conf = "multiclass_classification", min(0.9,1.0-up)
            else: task, conf = "regression", min(0.95,up)
        else:
            if u<=self.MAX_CLASSIFICATION_CLASSES: task, conf = "multiclass_classification", min(0.85,1.0-up)
            else: task, conf = "regression", 0.5
        self.profile.suggested_task = task; self.profile.suggested_target = target_col
        self.profile.feature_columns = [c for c in self.df.columns if c!=target_col]
        self.profile.task_confidence = round(conf,2)
        return task, target_col, conf

    def get_summary(self):
        return {"file_info":{"name":self.profile.file_name,"size_mb":round(self.profile.file_size_mb,2),"rows":self.profile.rows,"columns":self.profile.columns},"missing_summary":self.profile.missing_summary,"columns":{c:{"dtype":i["dtype"],"missing_pct":i["missing_pct"],"unique":i["unique"],"suggested_type":i["suggested_type"]} for c,i in self.profile.column_info.items()},"suggested_ml_task":{"task":self.profile.suggested_task,"target_column":self.profile.suggested_target,"confidence":self.profile.task_confidence},"warnings":self.profile.warnings}

    def run_full_profile(self, path_str: str, target_col: Optional[str] = None):
        self.load_data(path_str); self.analyze_columns(); self.analyze_missing(); self.suggest_task(target_col)
        return self.profile
