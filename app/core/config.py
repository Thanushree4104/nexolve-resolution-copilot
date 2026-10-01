from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "nexolve-resolution-copilot"
    log_level: str = "INFO"
    llm_provider: str = "mock"  # mock | ollama | hosted (used later)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()