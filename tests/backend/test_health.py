def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "X-RateLimit-Limit" in response.headers
    assert "X-RateLimit-Remaining" in response.headers
    assert "X-RateLimit-Reset" in response.headers
