def test_protected_route_without_token_is_401(client):
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401


def test_buildings_without_token_is_401(client):
    response = client.get("/api/v1/buildings")
    assert response.status_code == 401


def test_login_seed_user_and_me(client):
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "test-admin-password"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["role"] == "admin"
    token = body["access_token"]
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "admin@example.com"
    assert me.json()["role"] == "admin"


def test_placeholder_lists_are_empty(client):
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "test-admin-password"},
    )
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    for path in (
        "/api/v1/buildings",
        "/api/v1/readings",
        "/api/v1/models",
        "/api/v1/forecasts",
        "/api/v1/anomalies",
    ):
        response = client.get(path, headers=headers)
        assert response.status_code == 200
        assert response.json() == []
