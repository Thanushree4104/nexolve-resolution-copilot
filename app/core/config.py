from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "nexolve-resolution-copilot"
    log_level: str = "INFO"
    llm_provider: str = "mock"  # mock | ollama | hosted (used later)
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    llm_timeout_s: float = 30.0
    llm_cache_dir: str = ".cache/llm"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()