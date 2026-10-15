"""Write Mosquitto 2 password file from MQTT_* env vars. Never hardcode credentials."""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PASSWD_PATH = Path(__file__).resolve().parent / "config" / "passwd"


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def mosquitto_hash(password: str, iterations: int = 101) -> str:
    salt = secrets.token_bytes(12)
    digest = hashlib.pbkdf2_hmac("sha512", password.encode("utf-8"), salt, iterations, dklen=64)
    return f"$7${iterations}${_b64(salt)}${_b64(digest)}"


def main() -> None:
    values = _parse_env_file(REPO_ROOT / ".env")
    username = os.environ.get("MQTT_USERNAME") or values.get("MQTT_USERNAME")
    password = os.environ.get("MQTT_PASSWORD") or values.get("MQTT_PASSWORD")
    if not username or not password:
        raise SystemExit("MQTT_USERNAME and MQTT_PASSWORD must be set in the environment or .env")
    line = f"{username}:{mosquitto_hash(password)}\n"
    PASSWD_PATH.write_text(line, encoding="ascii")
    print(f"Wrote {PASSWD_PATH} for user {username!r}")


if __name__ == "__main__":
    main()
