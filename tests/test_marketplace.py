import pytest, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.marketplace import Marketplace

class TestMarketplace:
    @pytest.fixture
    def mp(self):
        d = tempfile.mkdtemp()
        return Marketplace(storage_dir=d)

    def test_publish(self, mp):
        lid = mp.publish(
            author="alice",
            model_id="binary_xgboost_001",
            title="Churn Predictor",
            description="Predicts customer churn with 95% accuracy",
            task="binary_classification",
            model_type="xgboost",
            metrics={"accuracy": 0.95},
            price_per_call=0.05,
        )
        assert lid in mp.listings
        assert mp.listings[lid]["author"] == "alice"

    def test_search_by_task(self, mp):
        mp.publish("alice", "m1", "Model 1", "", "binary_classification", "xgboost", {"accuracy": 0.9})
        mp.publish("bob", "m2", "Model 2", "", "regression", "xgboost", {"r2": 0.85})
        results = mp.search(task="binary_classification")
        assert len(results) == 1

    def test_search_by_price(self, mp):
        mp.publish("alice", "m1", "Cheap", "", "binary_classification", "xgboost", {}, price_per_call=0.01)
        mp.publish("bob", "m2", "Expensive", "", "binary_classification", "xgboost", {}, price_per_call=1.0)
        results = mp.search(max_price=0.05)
        assert len(results) == 1

    def test_use_model(self, mp):
        lid = mp.publish("alice", "m1", "Test", "", "binary_classification", "xgboost", {}, price_per_call=0.10)
        txn = mp.use_model(lid, "user123")
        assert txn["price"] == 0.10
        assert txn["platform_fee"] == 0.025
        assert txn["author_earning"] == pytest.approx(0.075)

    def test_add_review(self, mp):
        lid = mp.publish("alice", "m1", "Test", "", "binary_classification", "xgboost", {})
        mp.add_review(lid, "user1", 5, "Great!")
        mp.add_review(lid, "user2", 3, "Okay")
        assert mp.listings[lid]["rating"] == 4.0
        assert len(mp.listings[lid]["reviews"]) == 2

    def test_author_earnings(self, mp):
        lid = mp.publish("alice", "m1", "Test", "", "binary_classification", "xgboost", {}, price_per_call=0.10)
        mp.use_model(lid, "user1")
        mp.use_model(lid, "user2")
        earnings = mp.get_author_earnings("alice")
        assert earnings["total_calls"] == 2
        assert earnings["total_earnings"] == 0.15

    def test_platform_stats(self, mp):
        mp.publish("alice", "m1", "T1", "", "binary_classification", "xgboost", {})
        mp.publish("bob", "m2", "T2", "", "regression", "xgboost", {})
        stats = mp.get_platform_stats()
        assert stats["total_listings"] == 2
        assert "binary_classification" in stats["by_task"]

    def test_deactivate(self, mp):
        lid = mp.publish("alice", "m1", "Test", "", "binary_classification", "xgboost", {})
        mp.deactivate_listing(lid)
        results = mp.search()
        assert len(results) == 0

