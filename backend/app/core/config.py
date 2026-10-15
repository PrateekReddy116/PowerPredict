from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """All secrets and tunables come from the environment / .env — never from source defaults that look like real passwords."""

    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str = Field(default="sqlite:///./powerpredict-dev.sqlite3")
    jwt_secret: str
    jwt_expire_minutes: int = 60
    allowed_origins: str
    rate_limit_per_minute: int = 100
    rate_limit_user_per_minute: int = 100
    rate_limit_login_per_minute: int = 10
    # Only honor X-Forwarded-For when the process sits behind a trusted proxy.
    trust_x_forwarded_for: bool = False
    seed_admin_email: str | None = None
    seed_admin_password: str | None = None
    mqtt_enabled: bool = True
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_username: str | None = None
    mqtt_password: str | None = None
    mqtt_topic_prefix: str = "smartbuilding"
    mqtt_client_id: str = "powerpredict-api"
    artifacts_dir: str = Field(default=str(_REPO_ROOT / "artifacts"))
    anomaly_warning_deviation_pct: float = 25.0
    anomaly_critical_deviation_pct: float = 50.0
    anomaly_warning_score_cutoff: float | None = None
    anomaly_critical_score_cutoff: float | None = None
    anomaly_flag_if_outlier: bool = True
    anomaly_expected_floor_kwh: float = 1.0

    @field_validator("jwt_secret")
    @classmethod
    def jwt_secret_strength(cls, value: str) -> str:
        if len(value) < 32:
            raise ValueError("JWT_SECRET must be at least 32 characters")
        return value

    @field_validator("allowed_origins")
    @classmethod
    def reject_wildcard_origins(cls, value: str) -> str:
        origins = [part.strip() for part in value.split(",") if part.strip()]
        if not origins:
            raise ValueError("ALLOWED_ORIGINS must list at least one origin")
        if any(origin == "*" for origin in origins):
            raise ValueError("ALLOWED_ORIGINS must not contain wildcard '*'")
        return value

    @property
    def allowed_origins_list(self) -> list[str]:
        return [part.strip() for part in self.allowed_origins.split(",") if part.strip()]

    @property
    def mqtt_topic_root(self) -> str:
        return (self.mqtt_topic_prefix or "smartbuilding").strip().strip("/") or "smartbuilding"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
