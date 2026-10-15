---
inclusion: fileMatch
fileMatchPattern: ['backend/**/*.py']
---

# Backend security and contracts

## Auth

- Protect all routes except `GET /health` and `POST /api/v1/auth/login`.
- JWT access tokens; `JWT_SECRET` (and DB URLs) from env only — never hardcode.
- Roles: `viewer` (read), `admin` (simulator start/stop, ingest). Return 401 / 403.

## CORS

- Allow only `ALLOWED_ORIGINS` (comma-separated env). No wildcard `*` with credentials.
- Set methods, headers, and `allow_credentials=True` explicitly.

## Rate limiting

- IP-based global limit (default 100/min, env-configurable).
- User-based limit after auth. Stricter cap on `/auth/login` (e.g. 10/min/IP).
- On 429, send `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.

## Data origin

MQTT and ingest must set `data_origin`. Allowed: `historical` | `simulated_iot` | `real_iot`. Drop payloads missing origin. Historical CSV ingest must not publish MQTT.

```python
# BAD
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True)

# GOOD
origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, ...)
```
