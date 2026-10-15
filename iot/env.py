"""Load MQTT settings from the environment / .env. Credentials never live in source."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_env_file(path: Path | None = None) -> dict[str, str]:
    env_path = path or (REPO_ROOT / ".env")
    values: dict[str, str] = {}
    if not env_path.is_file():
        return values
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_env(path: Path | None = None) -> None:
    for key, value in parse_env_file(path).items():
        os.environ.setdefault(key, value)


def mqtt_settings(*, env_path: Path | None = None) -> dict[str, Any]:
    load_env(env_path)
    prefix = os.environ.get("MQTT_TOPIC_PREFIX") or os.environ.get("MQTT_TOPIC_ROOT") or "smartbuilding"
    port_raw = os.environ.get("MQTT_PORT", "1883")
    try:
        port = int(port_raw)
    except ValueError:
        port = 1883
    return {
        "host": os.environ.get("MQTT_HOST", "localhost"),
        "port": port,
        "username": os.environ.get("MQTT_USERNAME") or None,
        "password": os.environ.get("MQTT_PASSWORD") or None,
        "topic_root": prefix.strip().strip("/") or "smartbuilding",
    }
