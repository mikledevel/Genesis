import csv
import io
from unittest.mock import patch

import pandas as pd
import pytest
import app.main
from fastapi.testclient import TestClient
from app.core.auth import create_token
from app.db.database import GenesisDB


def example_result():
    return {"model_type": "forecast_mean_level", "horizon": 2, "target": "sales", "date_col": "date",
            "metrics": {"trust": {"level": "low", "reasons": ["Limited history"]}, "n_points": 100,
                        "n_origins": 5, "horizon": 2, "mae_test": 1.2345, "chosen": "mean_level",
                        "coverage_80_test": 0.2, "baseline_mae_test": 2.5},
            "info": {"freq": "D", "n": 100, "gaps": 0, "duplicates": 0},
            "history": pd.Series([8., 9.], index=pd.date_range("2026-11-01", periods=2)),
            "forecast": pd.DataFrame({"date": pd.date_range("2026-11-03", periods=2),
                                      "forecast": [10.1234, 11.5678], "lower": [8., 9.], "upper": [12., 13.]})}


@pytest.fixture
def exported_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from app.core.forecast_reports import create_snapshot
    db = GenesisDB()
    owner = db.create_user("owner@example.test", "unused")["id"]
    other = db.create_user("other@example.test", "unused")["id"]
    jid = db.enqueue_job(owner, "forecast", {"path": "old.csv"})
    db.claim_job()
    snapshot = create_snapshot(example_result(), "model-one", 3, "sum")
    db.finish_job(jid, result={"success": True, "data": {"report_snapshot": snapshot}})
    return TestClient(app.main.app), db, owner, other, jid, snapshot


@pytest.mark.parametrize("lang,warning", [("en", "not calibrated"), ("ru", "не откалиброван")])
def test_report_is_owned_offline_and_does_not_retrain(exported_job, lang, warning):
    client, db, owner, other, jid, snapshot = exported_job
    headers = {"Authorization": "Bearer " + create_token(owner)}
    with patch("app.core.engine.forecaster.Forecaster.run", side_effect=AssertionError("must not retrain")):
        result = client.get(f"/api/jobs/{jid}/report?lang={lang}", headers=headers)
        assert result.status_code == 200, result.text
        assert warning in result.text and "10.1" in result.text
        assert "<script" not in result.text and 'src="http' not in result.text
        assert result.headers["cache-control"] == "no-store"
        assert result.headers["content-disposition"].startswith("attachment;")
        assert client.get(f"/api/jobs/{jid}/report?lang={lang}", headers=headers).content == result.content
    assert db.get_job(owner, jid)["result"]["data"]["report_snapshot"] == snapshot
    assert client.get(f"/api/jobs/{jid}/report", headers={"Authorization": "Bearer " + create_token(other)}).status_code == 404
    assert client.get(f"/api/jobs/{jid}/report").status_code == 401


def test_csv_retains_snapshot_values_and_rejects_invalid_options(exported_job):
    client, _, owner, _, jid, _ = exported_job
    headers = {"Authorization": "Bearer " + create_token(owner)}
    result = client.get(f"/api/jobs/{jid}/report?format=csv", headers=headers)
    assert result.status_code == 200
    rows = list(csv.DictReader(io.StringIO(result.content.decode("utf-8-sig"))))
    assert float(rows[0]["forecast"]) == 10.1234
    assert client.get(f"/api/jobs/{jid}/report?format=exe", headers=headers).status_code == 422
    assert client.get(f"/api/jobs/{jid}/report?lang=xx", headers=headers).status_code == 422


def test_legacy_and_pending_jobs_do_not_get_recomputed(exported_job):
    client, db, owner, _, jid, _ = exported_job
    headers = {"Authorization": "Bearer " + create_token(owner)}
    with db.write_transaction() as conn:
        conn.execute("UPDATE background_jobs SET result_json='{}' WHERE id=?", (jid,))
    assert client.get(f"/api/jobs/{jid}/report", headers=headers).status_code == 409
    with db.write_transaction() as conn:
        conn.execute("UPDATE background_jobs SET status='pending' WHERE id=?", (jid,))
    assert client.get(f"/api/jobs/{jid}/report", headers=headers).status_code == 409


def test_snapshot_escapes_text_and_rejects_invalid_numeric_series():
    from app.core.forecast_reports import create_snapshot, render_snapshot
    raw = example_result()
    raw["target"] = "<script>alert(1)</script>"
    snapshot = create_snapshot(raw, "model-one", 1, "sum")
    document = render_snapshot(snapshot, "en", "html")
    assert "<script>" not in document and "&lt;script&gt;" in document
    snapshot["forecast"]["future"]["values"][0] = float("nan")
    with pytest.raises(ValueError):
        render_snapshot(snapshot, "en", "html")


def test_upload_and_preview_excel_style_csv_without_spending_quota(exported_job):
    client, db, owner, other, _, _ = exported_job
    headers = {"Authorization": "Bearer " + create_token(owner)}
    content = "Дата;Продажи\n" + "\n".join(f"2026-01-{i:02d};{i * 10}" for i in range(1, 31))
    result = client.post("/api/agent/upload", headers=headers,
                         files={"file": ("sales.csv", content.encode("cp1251"), "text/csv")})
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["columns"] == ["Дата", "Продажи"]
    body = {"dataset_id": data["dataset_id"], "date_column": "Дата", "target_column": "Продажи", "horizon": 7}
    with patch("app.core.engine.forecaster.Forecaster.run", side_effect=AssertionError("preview must not train")):
        preview = client.post("/api/forecasts/preview", headers=headers, json=body)
    assert preview.status_code == 200, preview.text
    assert preview.json()["points"] == 30 and preview.json()["eligible"]
    assert db.count_feature_use_this_month(owner, "train") == 0
    assert client.post("/api/forecasts/preview", headers=headers, json={**body, "target_column": "Дата"}).status_code == 422
    assert client.post("/api/forecasts/preview", headers={"Authorization": "Bearer " + create_token(other)}, json=body).status_code == 404
