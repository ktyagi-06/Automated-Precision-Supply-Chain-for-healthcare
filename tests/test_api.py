import dataclasses

import pytest
from fastapi.testclient import TestClient

import main
from config import Settings


@pytest.fixture(scope="module")
def client(monkeypatch_module=None):
    main_settings = dataclasses.replace(Settings.from_env(), autostart=False, api_key="secret", retrain_every_days=0)
    orig = Settings.from_env
    Settings.from_env = staticmethod(lambda: main_settings)
    try:
        with TestClient(main.app) as c:
            yield c
    finally:
        Settings.from_env = orig


def test_health_and_status(client):
    assert client.get("/api/health").json()["status"] == "ok"
    s = client.get("/api/status").json()
    assert s["live_mode"] == "simulate" and s["model_version"] != "none"


def test_snapshot_is_complete_and_consistent(client):
    s = client.get("/api/snapshot?days=7").json()
    assert {"status", "summary", "alerts", "reorders", "orders", "events"} <= set(s)
    assert len(s["reorders"]) == 10
    assert s["summary"]["total_inventory_units"] == sum(r["current_stock"] for r in s["reorders"])


def test_forecast_endpoint_and_validation(client):
    f = client.get("/api/forecast/P001?days=5").json()
    assert len(f["daily_forecast"]) == 5 and all(d["predicted_units"] >= 0 for d in f["daily_forecast"])
    assert client.get("/api/forecast/NOPE").status_code == 404
    assert client.get("/api/forecast/P001?days=0").status_code == 422
    assert client.get("/api/forecast/P001?days=61").status_code == 422


def test_order_flow(client):
    r = client.post("/api/orders", json={"product_id": "P004", "qty": 25})
    assert r.status_code == 201 and r.json()["status"] == "in_transit"
    assert any(o["id"] == r.json()["id"] for o in client.get("/api/orders").json())
    assert client.post("/api/orders", json={"product_id": "P004", "qty": 0}).status_code == 422
    assert client.post("/api/orders", json={"product_id": "NOPE", "qty": 5}).status_code == 404


def test_ingest_requires_api_key(client):
    body = {"product_id": "P001", "units": 2}
    assert client.post("/api/ingest/sale", json=body).status_code == 401
    ok = client.post("/api/ingest/sale", json=body, headers={"X-API-Key": "secret"})
    assert ok.status_code == 202 and ok.json()["source"] == "ingest"


def test_model_info_reports_baseline_and_versions(client):
    m = client.get("/api/model-info").json()
    assert m["best_model"] in {r["model"] for r in m["model_comparison"]}
    assert m["baseline"]["MAE_7d"] > 0 and "scikit-learn" in m["library_versions"]
