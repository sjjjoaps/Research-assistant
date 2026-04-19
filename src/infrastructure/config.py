"""
配置管理模块（基础设施层）
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


ROOT_DIR = Path(__file__).parent.parent.parent


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
    embedding_dim: int = 1024

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
    relation_faiss_index_dir: Path = Field(
        default=ROOT_DIR / "data" / "relation_faiss",
        alias="RELATION_FAISS_INDEX_DIR",
    )

    # Neo4j 配置
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field(default="neo4j", alias="NEO4J_PASSWORD")

    # Token 费用估算
    price_per_1k_prompt: float = Field(default=0.04, alias="PRICE_PER_1K_PROMPT")
    price_per_1k_completion: float = Field(default=0.12, alias="PRICE_PER_1K_COMPLETION")

    # Reranker（P1-Step 2）
    reranker_enabled: bool = Field(default=False, alias="RERANKER_ENABLED")
    reranker_api_url: str = Field(default="", alias="RERANKER_API_URL")
    reranker_api_key: str = Field(default="", alias="RERANKER_API_KEY")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL")
    reranker_top_n: int = Field(default=5, alias="RERANKER_TOP_N")
    reranker_timeout: int = Field(default=10, alias="RERANKER_TIMEOUT")
    reranker_max_candidates: int = Field(default=50, alias="RERANKER_MAX_CANDIDATES")

    # ToolCallLimiter（P5-Step 1.5）
    tool_call_max_per_window: int = Field(default=5, alias="TOOL_CALL_MAX_PER_WINDOW")
    tool_call_window_seconds: float = Field(default=60.0, alias="TOOL_CALL_WINDOW_SECONDS")


settings = Settings()
