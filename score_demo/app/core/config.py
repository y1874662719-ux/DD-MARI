from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "Essay Scoring Service"
    api_prefix: str = "/api/v1"
    workspace_dir: Path = Path(__file__).resolve().parents[2]
    rule_path: Path = Field(
        default=Path("data/rules/final_rule.json"),
        validation_alias=AliasChoices("ESSAY_RULE_PATH", "RULE_PATH"),
    )

    llm_provider: str = Field(
        default="auto",
        validation_alias=AliasChoices("LLM_PROVIDER", "ESSAY_LLM_PROVIDER", "BA_AGENT_LLM_PROVIDER"),
    )
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "ESSAY_LLM_API_KEY", "BA_AGENT_LLM_API_KEY"),
    )
    llm_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_BASE_URL", "ESSAY_LLM_BASE_URL", "BA_AGENT_LLM_BASE_URL"),
    )
    llm_model: str = Field(
        default="gpt-5.4-mini",
        validation_alias=AliasChoices("LLM_MODEL", "ESSAY_LLM_MODEL", "BA_AGENT_LLM_MODEL"),
    )

    geval_proxy_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GEVAL_PROXY_API_KEY", "ESSAY_GEVAL_PROXY_API_KEY", "BA_AGENT_GEVAL_PROXY_API_KEY"),
    )
    geval_proxy_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("GEVAL_PROXY_BASE_URL", "ESSAY_GEVAL_PROXY_BASE_URL", "BA_AGENT_GEVAL_PROXY_BASE_URL"),
    )
    geval_proxy_model: str = Field(
        default="gpt-5.4-mini",
        validation_alias=AliasChoices("GEVAL_PROXY_MODEL", "ESSAY_GEVAL_PROXY_MODEL", "BA_AGENT_GEVAL_PROXY_MODEL"),
    )

    deepseek_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "ESSAY_DEEPSEEK_API_KEY", "BA_AGENT_DEEPSEEK_API_KEY"),
    )
    deepseek_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("DEEPSEEK_BASE_URL", "ESSAY_DEEPSEEK_BASE_URL", "BA_AGENT_DEEPSEEK_BASE_URL"),
    )
    deepseek_model: str = Field(
        default="deepseek-chat",
        validation_alias=AliasChoices("DEEPSEEK_MODEL", "ESSAY_DEEPSEEK_MODEL", "BA_AGENT_DEEPSEEK_MODEL"),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def resolved_rule_path(self) -> Path:
        if self.rule_path.is_absolute():
            return self.rule_path
        return self.workspace_dir / self.rule_path


settings = Settings()
