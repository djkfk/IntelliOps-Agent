"""混合检索服务 - 向量召回 + 关键词召回 + RRF 融合"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Dict, List, Sequence

from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.vector_store_manager import vector_store_manager

# 运维问答中意义不大的停用词
_STOPWORDS = {
    "的",
    "了",
    "吗",
    "呢",
    "啊",
    "吧",
    "是",
    "有",
    "和",
    "与",
    "或",
    "在",
    "如何",
    "怎么",
    "怎样",
    "什么",
    "哪些",
    "一个",
    "一下",
    "处理",
    "方案",
    "问题",
    "告警",
    "please",
    "the",
    "a",
    "an",
    "how",
    "what",
    "is",
    "to",
}


class HybridRetrieverService:
    """应用层混合检索：向量 + 关键词，经 RRF 融合后输出候选集"""

    def retrieve(self, query: str, recall_k: int | None = None) -> List[Document]:
        """
        混合检索并返回融合后的候选文档

        Args:
            query: 用户查询
            recall_k: 召回数量，默认使用配置 rag_recall_k
        """
        if not query or not query.strip():
            return []

        top_k = recall_k or config.rag_recall_k
        terms = self._extract_terms(query)

        vector_docs = self._vector_search(query, top_k)
        keyword_docs = self._keyword_search(query, terms, top_k, fallback_docs=vector_docs)

        fused = self._rrf_fuse([vector_docs, keyword_docs], k=config.rag_rrf_k)
        result = fused[:top_k]

        logger.info(
            f"混合检索完成: query='{query}', terms={terms}, "
            f"vector={len(vector_docs)}, keyword={len(keyword_docs)}, fused={len(result)}"
        )
        return result

    def _vector_search(self, query: str, top_k: int) -> List[Document]:
        """向量相似度召回"""
        try:
            docs = vector_store_manager.similarity_search(query, k=top_k)
            return docs
        except Exception as e:
            logger.error(f"向量召回失败: {e}")
            return []

    def _keyword_search(
        self,
        query: str,
        terms: Sequence[str],
        top_k: int,
        fallback_docs: Sequence[Document] | None = None,
    ) -> List[Document]:
        """
        关键词召回：Milvus content LIKE + 标题/告警名加权排序

        若 LIKE 查询不可用，则退化为对向量召回结果做关键词重排。
        """
        if not terms:
            return []

        docs_by_key: Dict[str, Document] = {}
        scores: Dict[str, float] = defaultdict(float)

        # 1) Milvus LIKE 召回
        try:
            collection = milvus_manager.get_collection()
            for term in terms:
                expr = f'content like "%{self._escape_like(term)}%"'
                try:
                    rows = collection.query(
                        expr=expr,
                        output_fields=["id", "content", "metadata"],
                        limit=top_k,
                    )
                except Exception as e:
                    logger.debug(f"关键词 LIKE 查询失败 term={term}: {e}")
                    continue

                for row in rows:
                    doc = self._row_to_document(row)
                    key = self._doc_key(doc)
                    docs_by_key[key] = doc
                    scores[key] += self._term_score(term, doc, query)
        except Exception as e:
            logger.warning(f"关键词召回（Milvus）不可用，将仅基于向量候选打分: {e}")

        # 2) 若 LIKE 无结果，对向量候选做关键词打分，保证仍有关键词排序列表
        if not docs_by_key and fallback_docs:
            for doc in fallback_docs:
                key = self._doc_key(doc)
                docs_by_key[key] = doc
                for term in terms:
                    scores[key] += self._term_score(term, doc, query)

        ranked_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [docs_by_key[k] for k in ranked_keys[:top_k] if scores[k] > 0]

    def _term_score(self, term: str, doc: Document, query: str) -> float:
        """单关键词对文档的匹配分，告警名/标题命中加权"""
        content = doc.page_content or ""
        content_lower = content.lower()
        term_lower = term.lower()
        metadata = doc.metadata or {}

        score = 0.0
        if term_lower in content_lower:
            score += 1.0

        # 告警名常见写法：`HighCPUUsage` 或 **告警名**: `xxx`
        if re.search(rf"[`'\"]({re.escape(term)})[`'\"]", content, flags=re.IGNORECASE):
            score += 3.0
        if term.lower() in query.lower() and re.search(
            rf"告警名\s*[:：].*{re.escape(term)}", content, flags=re.IGNORECASE
        ):
            score += 2.0

        for header_key in ("h1", "h2", "h3"):
            header = str(metadata.get(header_key, "") or "")
            if term_lower and term_lower in header.lower():
                score += 1.5

        file_name = str(metadata.get("_file_name", "") or "")
        if term_lower and term_lower in file_name.lower():
            score += 1.0

        return score

    def _rrf_fuse(self, ranked_lists: Sequence[Sequence[Document]], k: int = 60) -> List[Document]:
        """Reciprocal Rank Fusion"""
        scores: Dict[str, float] = defaultdict(float)
        docs_by_key: Dict[str, Document] = {}

        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                key = self._doc_key(doc)
                docs_by_key[key] = doc
                scores[key] += 1.0 / (k + rank + 1)

        ordered = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [docs_by_key[key] for key in ordered]

    def _extract_terms(self, query: str) -> List[str]:
        """从查询中提取英文标识符与中文关键词"""
        terms: List[str] = []

        # 英文/数字标识（告警名、服务名、错误码等）
        for m in re.finditer(r"[A-Za-z][A-Za-z0-9_\-]{1,}", query):
            token = m.group(0)
            if token.lower() not in _STOPWORDS:
                terms.append(token)

        # 连续中文片段，再切成 2~4 字窗口以提升召回
        for m in re.finditer(r"[\u4e00-\u9fff]{2,}", query):
            zh = m.group(0)
            if zh in _STOPWORDS:
                continue
            terms.append(zh)
            if len(zh) > 4:
                for i in range(len(zh) - 1):
                    gram = zh[i : i + 2]
                    if gram not in _STOPWORDS:
                        terms.append(gram)

        # 去重保序
        seen = set()
        unique: List[str] = []
        for t in terms:
            key = t.lower()
            if key not in seen:
                seen.add(key)
                unique.append(t)
        return unique[:12]

    @staticmethod
    def _escape_like(value: str) -> str:
        """转义 Milvus LIKE 特殊字符"""
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_").replace('"', '\\"')

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """文档去重键"""
        meta_id = (doc.metadata or {}).get("pk") or (doc.metadata or {}).get("id")
        if meta_id:
            return str(meta_id)
        return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

    @staticmethod
    def _row_to_document(row: dict) -> Document:
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}
        # 保留 milvus id，便于融合去重
        metadata = {**metadata, "id": row.get("id")}
        return Document(page_content=row.get("content") or "", metadata=metadata)


hybrid_retriever_service = HybridRetrieverService()
