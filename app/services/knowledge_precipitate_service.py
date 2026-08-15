"""知识点沉淀服务 - 将 Chat 问答提炼为 FAQ 并入库"""

import re
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from app.core.llm_factory import llm_factory
from app.services.vector_index_service import vector_index_service

KNOWLEDGE_DIR = Path("./uploads/knowledge")

_DRAFT_SYSTEM_PROMPT = """你是运维知识库编辑助手。请将用户提供的「问答」提炼为可检索的 FAQ 知识点。

严格按以下 Markdown 模板输出，不要添加模板之外的说明文字：

# {{简洁标题}}

## 问题
{{规范化后的问题}}

## 答案
{{结构化、可操作的答案，保留关键步骤与结论}}

## 适用场景
{{2-4 条适用场景，每条一行，以 - 开头}}

## 来源
- session_id: {session_id}
- precipitated_at: {precipitated_at}

要求：
1. 标题简洁，不超过 30 字，不要加书名号
2. 答案基于给定内容整理，不要编造未出现的事实
3. 若原回答信息不足，在答案中如实说明不确定点
4. 全程使用中文
"""


class KnowledgePrecipitateService:
    """知识点沉淀服务"""

    def __init__(self) -> None:
        self.knowledge_dir = KNOWLEDGE_DIR
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"知识点沉淀服务初始化完成, dir={self.knowledge_dir}")

    async def generate_draft(
        self, session_id: str, question: str, answer: str
    ) -> dict[str, str]:
        """用 LLM 将问答提炼为 FAQ Markdown 草稿"""
        precipitated_at = datetime.now().isoformat(timespec="seconds")
        system_prompt = _DRAFT_SYSTEM_PROMPT.format(
            session_id=session_id or "unknown",
            precipitated_at=precipitated_at,
        )
        user_prompt = (
            f"会话 ID: {session_id}\n\n"
            f"【用户问题】\n{question.strip()}\n\n"
            f"【助手回答】\n{answer.strip()}\n"
        )

        try:
            llm = llm_factory.create_chat_model(temperature=0.2, streaming=False)
            response = await llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            content = (response.content if hasattr(response, "content") else str(response)).strip()
            if not content:
                raise RuntimeError("LLM 返回空内容")

            title = self._extract_title(content) or self._fallback_title(question)
            # 确保来源字段存在（LLM 偶发遗漏时补齐）
            content = self._ensure_source_section(content, session_id, precipitated_at)

            logger.info(f"知识点草稿生成成功: session={session_id}, title={title}")
            return {"title": title, "content": content}

        except Exception as e:
            logger.error(f"知识点草稿生成失败: {e}")
            raise

    def confirm_and_index(
        self, title: str, content: str, session_id: str | None = None
    ) -> dict:
        """人工确认后落盘并写入向量库"""
        title = title.strip()
        content = content.strip()
        if not title or not content:
            raise ValueError("标题和内容不能为空")

        safe_title = self._sanitize_filename(title)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{safe_title}.md"
        file_path = self.knowledge_dir / filename

        try:
            # 补齐一级标题（若用户编辑时删掉）
            if not content.lstrip().startswith("#"):
                content = f"# {title}\n\n{content}"

            file_path.write_text(content, encoding="utf-8")
            logger.info(f"知识点文件已写入: {file_path}")

            chunk_count = vector_index_service.index_single_file(str(file_path))
            logger.info(
                f"知识点入库成功: file={filename}, chunks={chunk_count}, session={session_id}"
            )

            return {
                "success": True,
                "filename": filename,
                "file_path": str(file_path.resolve().as_posix()),
                "chunk_count": chunk_count,
            }

        except Exception as e:
            logger.error(f"知识点确认入库失败: {e}")
            raise

    @staticmethod
    def _extract_title(content: str) -> str:
        """从 Markdown 提取一级标题"""
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return ""

    @staticmethod
    def _fallback_title(question: str) -> str:
        """问题截断作为后备标题"""
        q = re.sub(r"\s+", " ", question.strip())
        return q[:30] + ("..." if len(q) > 30 else "")

    @staticmethod
    def _ensure_source_section(
        content: str, session_id: str, precipitated_at: str
    ) -> str:
        """确保文档包含来源小节"""
        if "## 来源" in content:
            return content
        source = (
            f"\n\n## 来源\n"
            f"- session_id: {session_id or 'unknown'}\n"
            f"- precipitated_at: {precipitated_at}\n"
        )
        return content.rstrip() + source

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """规范化文件名片段"""
        sanitized = name.strip().replace(" ", "_")
        for char in ["\\", "/", ":", "*", "?", '"', "<", ">", "|", "#"]:
            sanitized = sanitized.replace(char, "_")
        sanitized = re.sub(r"_+", "_", sanitized).strip("._")
        return (sanitized[:50] or "knowledge").rstrip("_")


knowledge_precipitate_service = KnowledgePrecipitateService()
