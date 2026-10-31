"""Runtime settings, read from environment variables / .env (prefix PROMPTSHIELD_)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PROMPTSHIELD_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./promptshield.db"
    api_keys: str = "dev-key"  # comma-separated
    ner_backend: str = "onnx"  # onnx | torch | none
    ner_model_file: str = "model.int8.onnx"
    ner_threads: int = 0  # ONNX Runtime intra-op threads per worker; 0 = all cores
    llm_provider: str = "mock"  # mock | ollama | openai_compatible
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2"
    llm_api_key: str = ""
    rate_limit_per_minute: int = 120
    max_text_chars: int = 50_000
    max_upload_mb: int = 10

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


settings = Settings()
