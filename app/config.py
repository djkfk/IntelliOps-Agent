"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from pathlib import Path
from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict

# 始终从项目根目录加载 .env，避免从其他工作目录启动 uvicorn 时读不到配置
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "IntelliOps Agent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # Milvus 配置（默认端口与仓库内 vector-database.yml 宿主机映射 29530:19530 一致）
    milvus_host: str = "localhost"
    milvus_port: int = 29530
    milvus_timeout: int = 10000  # 毫秒

    @property
    def milvus_uri(self) -> str:
        """pymilvus.MilvusClient / langchain_milvus 必须用 uri；仅 host+port 不会覆盖默认 localhost:19530。"""
        return f"http://{self.milvus_host}:{self.milvus_port}"

    # RAG 配置
    rag_top_k: int = 3
    rag_recall_k: int = 20  # 混合检索召回数量（重排前）
    rag_rrf_k: int = 60  # RRF 融合常数
    rag_rerank_enabled: bool = True
    rag_rerank_model: str = "gte-rerank-v2"
    rag_model: str = "qwen-max"  # 使用快速响应模型，不带扩展思考

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # 对话上下文管理（超过阈值后摘要压缩，保留近几轮原文）
    context_enabled: bool = True
    context_max_tokens: int = 24000
    context_keep_messages: int = 8

    # MCP 服务配置
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            }
        }


# 全局配置实例
config = Settings()
