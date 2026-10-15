from app.core.rate_limit import SlidingWindowLimiter


def test_login_rate_limit_returns_429(client):
    payload = {"email": "admin@example.com", "password": "wrong-password"}
    statuses = []
    last_limited = None
    for _ in range(12):
        response = client.post("/api/v1/auth/login", json=payload)
        statuses.append(response.status_code)
        if response.status_code == 429:
            last_limited = response
            break
    assert 429 in statuses
    assert last_limited is not None
    assert last_limited.headers["X-RateLimit-Limit"] == "10"
    assert last_limited.headers["X-RateLimit-Remaining"] == "0"
    assert "X-RateLimit-Reset" in last_limited.headers


def test_limiter_key_exhausts_then_resets_window_unit():
    limiter = SlidingWindowLimiter()
    last_ok = None
    for _ in range(3):
        last_ok = limiter.hit("unit:ip", limit=3, window_seconds=60)
        assert last_ok.allowed
    denied = limiter.hit("unit:ip", limit=3, window_seconds=60)
    assert not denied.allowed
    assert denied.remaining == 0
    assert last_ok is not None
    assert last_ok.remaining == 0


def test_login_rate_limit_ignores_spoofed_forwarded_for(client):
    payload = {"email": "admin@example.com", "password": "wrong-password"}
    statuses = []
    last_limited = None
    for i in range(12):
        response = client.post(
            "/api/v1/auth/login",
            json=payload,
            headers={"X-Forwarded-For": f"203.0.113.{i}"},
        )
        statuses.append(response.status_code)
        if response.status_code == 429:
            last_limited = response
            break
    assert 429 in statuses
    assert last_limited is not None
    assert last_limited.headers["X-RateLimit-Limit"] == "10"
