from __future__ import annotations

import os

# Must be set before app.core.config is imported.
_TEST_SECRET = "test-jwt-secret-key-32-chars-min"
os.environ.setdefault("JWT_SECRET", _TEST_SECRET)
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:5173")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SEED_ADMIN_EMAIL", "admin@example.com")
os.environ.setdefault("SEED_ADMIN_PASSWORD", "test-admin-password")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "100")
os.environ.setdefault("RATE_LIMIT_USER_PER_MINUTE", "100")
os.environ.setdefault("RATE_LIMIT_LOGIN_PER_MINUTE", "10")
os.environ.setdefault("MQTT_ENABLED", "false")

import pytest
from fastapi.testclient import TestClient

from app.core.config import reset_settings_cache
from app.core.rate_limit import limiter
from app.db.session import reset_engine
from app.main import create_app
from app.services.mqtt_consumer import reset_runtime_status


@pytest.fixture
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    db_path = tmp_path / "phase1.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("JWT_SECRET", _TEST_SECRET)
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:5173")
    monkeypatch.setenv("SEED_ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("SEED_ADMIN_PASSWORD", "test-admin-password")
    monkeypatch.setenv("RATE_LIMIT_LOGIN_PER_MINUTE", "10")
    monkeypatch.setenv("MQTT_ENABLED", "false")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    reset_settings_cache()
    reset_engine()
    limiter.reset()
    reset_runtime_status()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    reset_engine()
    reset_settings_cache()
    limiter.reset()
    reset_runtime_status()
