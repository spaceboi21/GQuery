from pathlib import Path
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="XIA_",
        env_file_encoding="utf-8",
    )

    # Data & indexing
    data_path: Path = Path.home() / "Documents" / "Excel"
    db_path: Path = Path("xia_knowledge.db")
    large_file_threshold_mb: int = 100
    sandbox_timeout_sec: int = 180
    sandbox_memory_mb: int = 512
    max_sample_rows: int = 5
    fts_result_limit: int = 20
    kg_hop_limit: int = 2

    # LLM
    llm_provider: str = "openai"        # openai | anthropic | ollama
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = ""              # Azure or Ollama override
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096

    # App server
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    @field_validator("data_path", mode="before")
    @classmethod
    def resolve_data_path(cls, v: str | Path) -> Path:
        p = Path(v).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @model_validator(mode="after")
    def resolve_db_path(self) -> "Settings":
        if not self.db_path.is_absolute():
            self.db_path = (self.data_path / self.db_path).resolve()
        return self

    def safe_path(self, raw: str | Path) -> Path:
        """Resolve and assert path is under data_path (directory traversal guard)."""
        resolved = Path(raw).resolve()
        try:
            resolved.relative_to(self.data_path)
        except ValueError:
            raise PermissionError(
                f"Path '{resolved}' is outside the approved data directory '{self.data_path}'"
            )
        return resolved


settings = Settings()
