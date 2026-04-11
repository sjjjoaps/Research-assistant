"""
配置管理模块
从 .env 文件读取所有配置，暴露统一的 Settings 对象
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


# 项目根目录
ROOT_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM 配置
    model_name: str = Field(default="qwen3.5-flash", alias="MODEL_NAME")
    base_url: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1", alias="BASE_URL")
    api_key: str = Field(default="", alias="API_KEY")
    embedding_model_name: str = Field(default="text-embedding-v3", alias="EMBEDDING_MODEL_NAME")
    embedding_dim: int = 1024  # text-embedding-v3 固定维度

    # LLM 稳定性
    llm_timeout_seconds: int = Field(default=30, alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")

    # 应用配置
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="127.0.0.1", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    # 实体抽取配置
    enable_entity_extraction: bool = Field(default=False, alias="ENABLE_ENTITY_EXTRACTION")
    entity_extraction_max_chunks: int = Field(default=20, alias="ENTITY_EXTRACTION_MAX_CHUNKS")

    # 存储路径
    data_dir: Path = Field(default=ROOT_DIR / "data", alias="DATA_DIR")
    sqlite_path: Path = Field(default=ROOT_DIR / "data" / "metadata.db", alias="SQLITE_PATH")
    faiss_index_dir: Path = Field(default=ROOT_DIR / "data" / "faiss", alias="FAISS_INDEX_DIR")

    # Neo4j 配置
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field(default="neo4j", alias="NEO4J_PASSWORD")


# 全局单例
settings = Settings()
