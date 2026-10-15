import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_cors_config_rejects_wildcard():
    with pytest.raises(ValidationError):
        Settings(
            jwt_secret="x" * 32,
            allowed_origins="*",
            database_url="sqlite:///:memory:",
        )


def test_cors_config_rejects_wildcard_in_list():
    with pytest.raises(ValidationError):
        Settings(
            jwt_secret="x" * 32,
            allowed_origins="http://localhost:5173, *",
            database_url="sqlite:///:memory:",
        )


def test_cors_allows_configured_origin(client):
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "*" not in (response.headers.get("access-control-allow-origin") or "")


def test_cors_rejects_unknown_origin(client):
    response = client.options(
        "/health",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.headers.get("access-control-allow-origin") not in {
        "*",
        "https://evil.example",
    }
