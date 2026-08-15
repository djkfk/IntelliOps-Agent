"""知识检索工具 - 从向量数据库中检索相关信息（混合检索 + 重排）"""

import threading
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.services.hybrid_retriever_service import hybrid_retriever_service
from app.services.rerank_service import rerank_service

# 协程内可读；工具若在线程池执行则同时写入全局，供请求协程读取
_last_rag_result: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "last_rag_result", default=None
)
_last_rag_global: Optional[Dict[str, Any]] = None
_last_rag_lock = threading.Lock()
# idle | searching | done —— 供流式接口轮询，不依赖 LangChain callbacks
_rag_phase: str = "idle"


def clear_last_rag_result() -> None:
    """清空当前协程与全局中的 RAG 命中摘要"""
    global _last_rag_global, _rag_phase
    _last_rag_result.set(None)
    with _last_rag_lock:
        _last_rag_global = None
        _rag_phase = "idle"


def get_last_rag_result() -> Optional[Dict[str, Any]]:
    """获取最近一次 RAG 命中摘要（优先协程上下文，其次全局）"""
    ctx_val = _last_rag_result.get()
    if ctx_val is not None:
        return ctx_val
    with _last_rag_lock:
        return _last_rag_global


def get_rag_phase() -> str:
    """获取 RAG 检索阶段：idle / searching / done"""
    with _last_rag_lock:
        return _rag_phase


def mark_rag_searching() -> None:
    """标记知识库检索已开始"""
    global _rag_phase
    with _last_rag_lock:
        _rag_phase = "searching"
    logger.info("RAG 阶段标记: searching")


def docs_to_sources(docs: List[Document]) -> List[Dict[str, str]]:
    """将文档列表转为前端可展示的来源摘要"""
    sources: List[Dict[str, str]] = []
    for doc in docs:
        metadata = doc.metadata or {}
        headers = []
        for key in ("h1", "h2", "h3"):
            if metadata.get(key):
                headers.append(str(metadata[key]))
        source = (
            metadata.get("_file_name")
            or metadata.get("_source")
            or metadata.get("source")
            or "未知来源"
        )
        content = doc.page_content or ""
        sources.append(
            {
                "source": str(source),
                "title": " > ".join(headers),
                "snippet": content[:120] + ("..." if len(content) > 120 else ""),
            }
        )
    return sources


def _set_rag_result(docs: List[Document], query: str) -> Dict[str, Any]:
    """写入并返回 RAG 命中摘要"""
    global _last_rag_global, _rag_phase
    sources = docs_to_sources(docs)
    hit = len(sources) > 0
    result: Dict[str, Any] = {
        "hit": hit,
        "count": len(sources),
        "query": query,
        "sources": sources,
        "message": (
            f"已从知识库命中 {len(sources)} 条" if hit else "知识库未命中相关内容"
        ),
    }
    _last_rag_result.set(result)
    with _last_rag_lock:
        _last_rag_global = result
        _rag_phase = "done"
    logger.info(f"RAG 阶段标记: done hit={hit} count={len(sources)}")
    return result


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题

    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。

    Args:
        query: 用户的问题或查询

    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    try:
        logger.info(f"知识检索工具被调用: query='{query}'")
        mark_rag_searching()

        # 1) 混合检索（向量 + 关键词 → RRF）
        candidates = hybrid_retriever_service.retrieve(query, recall_k=config.rag_recall_k)

        if not candidates:
            logger.warning("未检索到相关文档")
            _set_rag_result([], query)
            return "没有找到相关信息。", []

        # 2) DashScope 重排，失败时自动降级
        docs = rerank_service.rerank(query, candidates, top_n=config.rag_top_k)

        context = format_docs(docs)
        _set_rag_result(docs, query)

        logger.info(f"检索到 {len(docs)} 个相关文档（候选 {len(candidates)}）")
        return context, docs

    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        _set_rag_result([], query)
        return f"检索知识时发生错误: {str(e)}", []


def format_docs(docs: List[Document]) -> str:
    """
    格式化文档列表为上下文文本

    Args:
        docs: 文档列表

    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []

    for i, doc in enumerate(docs, 1):
        # 提取元数据
        metadata = doc.metadata
        source = metadata.get("_file_name", "未知来源")

        # 提取标题信息 (如果有)
        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])

        header_str = " > ".join(headers) if headers else ""

        # 构建格式化文本
        formatted = f"【参考资料 {i}】"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"
        formatted += f"\n内容:\n{doc.page_content}\n"

        formatted_parts.append(formatted)

    return "\n".join(formatted_parts)
