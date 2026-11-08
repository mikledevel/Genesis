"""A client-supplied file path is only used when it is inside an allowed folder (app/core/paths.py)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app.main  # noqa: E402,F401  (must be imported before chdir: it mounts ./static)
from fastapi.testclient import TestClient  # noqa: E402

from app.core.paths import is_allowed_path  # noqa: E402

_IP = 0


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for d in ("datasets", "models", "static", "generated", "datasets_backup", "static_private", "other"):
        os.makedirs(d)
        open(os.path.join(d, "f.csv"), "w").write("a,b\n1,2\n")
    return tmp_path


def test_files_inside_the_allowed_folders_are_accepted(work):
    for p in ("datasets/f.csv", "./models/f.csv", "static/f.csv", "generated/f.csv", "datasets", str(work / "datasets" / "f.csv")):
        assert is_allowed_path(p), p


def test_a_sibling_folder_that_merely_starts_with_the_same_letters_is_refused(work):
    # the old check was realpath.startswith(realpath("./datasets")), which also accepted these
    for p in ("datasets_backup/f.csv", "static_private/f.csv", str(work / "datasets_backup" / "f.csv")):
        assert not is_allowed_path(p), p


def test_traversal_absolute_paths_symlinks_and_junk_are_refused(work):
    assert not is_allowed_path("datasets/../other/f.csv")
    assert not is_allowed_path("other/f.csv") and not is_allowed_path("/etc/passwd")
    assert not is_allowed_path("") and not is_allowed_path(None) and not is_allowed_path("datasets/f.csv\x00.png")
    os.symlink(str(work / "other" / "f.csv"), str(work / "datasets" / "link.csv"))          # a link inside datasets/ that points outside
    assert not is_allowed_path("datasets/link.csv")


def test_the_agents_own_validator_uses_the_same_rule(work):
    from app.core.agent import GenesisAgent
    agent = GenesisAgent.__new__(GenesisAgent)
    assert agent._validate_path("datasets/f.csv") and not agent._validate_path("datasets_backup/f.csv")


@pytest.mark.parametrize("route", ["/api/model-builder/suggest", "/api/model-builder/generate"])
def test_the_model_builder_routes_refuse_a_file_outside_the_allowed_folders(work, route):
    from app.main import app
    from app import deps
    deps._agents.clear()
    c = TestClient(app)
    global _IP
    _IP += 1
    c.headers["x-forwarded-for"] = f"10.55.{_IP // 250}.{_IP % 250}"
    r = c.post("/api/auth/register", json={"email": f"pv{_IP}@t.test", "password": "secret12", "name": "T"})
    assert r.status_code == 200, r.text
    from app.db.database import GenesisDB
    GenesisDB().set_user_plan(r.json()["user"]["id"], "pro")
    h = {"Authorization": "Bearer " + r.json()["token"]}
    for bad in ("other/f.csv", "datasets_backup/f.csv"):
        resp = c.post(route, json={"dataset_path": bad, "description": "d"}, headers=h)
        assert resp.status_code == 403 and "inside the datasets folder" in resp.json()["detail"], (bad, resp.status_code, resp.text)
