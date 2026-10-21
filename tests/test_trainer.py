import pytest, pandas as pd, numpy as np, tempfile, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.engine.trainer import DatasetTrainer

class TestTrainer:
    @pytest.fixture
    def df(self): return pd.DataFrame({"age":[25,30,35,40,45,28,33,38,43,48],"salary":[50,60,75,90,110,55,70,85,100,120],"churn":[0,1,0,0,1,0,1,0,0,1]})
    @pytest.fixture
    def t(self): return DatasetTrainer(model_dir=tempfile.mkdtemp())
    def test_prep(self,t,df): X,y=t.prepare_data(df,"churn"); assert len(X)==len(y)
    def test_clf(self,t,df): r=t.train(df,"churn","binary_classification","xgboost"); assert r.metrics["accuracy_test"]>0
    def test_reg(self,t,df): r=t.train(df,"salary","regression","xgboost"); assert "r2_test" in r.metrics
    def test_sum(self,t,df): t.train(df,"churn","binary_classification","xgboost"); s=t.get_summary(); assert "model_path" in s
    def test_saved(self,t,df): r=t.train(df,"churn","binary_classification","xgboost"); assert os.path.exists(r.model_path)

