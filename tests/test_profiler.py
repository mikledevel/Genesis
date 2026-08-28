import pytest, pandas as pd, numpy as np, tempfile, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.engine.profiler import DatasetProfiler

class TestProfiler:
    @pytest.fixture
    def csv(self):
        df = pd.DataFrame({"age":[25,30,35,40,45,28,33,38,43,48],"salary":[50,60,75,90,110,55,70,85,100,120],"churn":[0,1,0,0,1,0,1,0,0,1],"dept":["IT","HR","IT","Sales","HR","IT","Sales","IT","HR","Sales"]})
        with tempfile.NamedTemporaryFile(mode="w",suffix=".csv",delete=False) as f: df.to_csv(f,index=False); p=f.name
        yield p; os.unlink(p)
    @pytest.fixture
    def p(self): return DatasetProfiler()
    def test_load(self,p,csv): df=p.load_data(csv); assert len(df)==10
    def test_not_found(self,p):
        with pytest.raises(FileNotFoundError): p.load_data("x.csv")
    def test_cols(self,p,csv): p.load_data(csv); i=p.analyze_columns(); assert i["churn"]["suggested_type"]=="binary"
    def test_clf(self,p,csv): p.load_data(csv); p.analyze_columns(); t,_,c=p.suggest_task("churn"); assert t=="binary_classification"
    def test_reg(self,p,csv): p.load_data(csv); p.analyze_columns(); t,_,_=p.suggest_task("salary"); assert t=="regression"
    def test_sum(self,p,csv): p.run_full_profile(csv,"churn"); s=p.get_summary(); assert s["suggested_ml_task"]["task"]=="binary_classification"
