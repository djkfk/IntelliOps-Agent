"""文本重排服务 - 基于 DashScope TextReRank"""

from __future__ import annotations

from http import HTTPStatus
from typing import List

from dashscope import TextReRank
from langchain_core.documents import Document
from loguru import logger

from app.config import config


class RerankService:
    """使用 DashScope 对候选文档重排"""

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int | None = None,
    ) -> List[Document]:
        """
        对候选文档重排

        Args:
            query: 用户查询
            documents: 候选文档
            top_n: 返回数量，默认 rag_top_k

        Returns:
            重排后的文档列表；失败时原样截断返回（降级）
        """
        if not documents:
            return []

        limit = top_n or config.rag_top_k
        if not config.rag_rerank_enabled:
            return documents[:limit]

        if len(documents) <= 1:
            return documents[:limit]

        try:
            texts = [doc.page_content for doc in documents]
            resp = TextReRank.call(
                model=config.rag_rerank_model,
                query=query,
                documents=texts,
                top_n=min(limit, len(texts)),
                return_documents=False,
                api_key=config.dashscope_api_key or None,
            )

            if resp.status_code != HTTPStatus.OK:
                logger.warning(
                    f"DashScope 重排失败，降级使用融合结果: "
                    f"code={resp.status_code}, message={getattr(resp, 'message', '')}"
                )
                return documents[:limit]

            results = getattr(getattr(resp, "output", None), "results", None) or []
            reranked: List[Document] = []
            for item in results:
                # SDK 可能返回对象或 dict
                index = item["index"] if isinstance(item, dict) else getattr(item, "index", None)
                if index is None or not isinstance(index, int):
                    continue
                if 0 <= index < len(documents):
                    reranked.append(documents[index])

            if not reranked:
                logger.warning("DashScope 重排结果为空，降级使用融合结果")
                return documents[:limit]

            logger.info(f"重排完成: model={config.rag_rerank_model}, 返回 {len(reranked)} 条")
            return reranked[:limit]

        except Exception as e:
            logger.warning(f"重排异常，降级使用融合结果: {e}")
            return documents[:limit]


rerank_service = RerankService()
