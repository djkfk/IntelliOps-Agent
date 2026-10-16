"""运维 Wiki 服务

LLM 只生成修改提案，人工确认后才落盘、重建目录并写入向量库。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from app.config import config
from app.core.llm_factory import llm_factory

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 允许 LLM 创建或更新的目录。index.md / log.md / WIKI.md 只由程序维护。
ALLOWED_CATEGORIES = ("alerts", "services", "incidents", "topics")
ROOT_READABLE = {"index.md", "WIKI.md", "log.md"}
CATEGORY_LABELS = {
    "alerts": "告警",
    "services": "服务",
    "incidents": "诊断案例",
    "topics": "主题",
}
PAGE_TYPES = {
    "alerts": "alert",
    "services": "service",
    "incidents": "incident",
    "topics": "topic",
}
_SAFE_STEM = re.compile(r"^[\w.\-\u4e00-\u9fff]+$")
_SOURCE_LIMIT = 12000

WIKI_GUIDE = """# 运维 Wiki 写作规范

本文件约束如何维护 `wiki/` 下的页面。`index.md` 与 `log.md` 由程序生成，不要修改。

## 页面类型

只允许以下目录：

- `alerts/`：一种告警一页。记录判定条件、常见根因、排查步骤、处理方式和历史案例链接。
- `services/`：一个服务一页。记录职责、依赖、常见故障和关联告警。
- `incidents/`：一次诊断或故障一页。记录时间、现象、证据、根因和关联页面。
- `topics/`：不属于以上三类的可复用经验。

## 页面格式

第一行是一级标题。标题后紧跟摘要引用，格式固定：

> 类型: alert | 摘要: 一句话说明本页结论

类型取值：`alert`、`service`、`incident`、`topic`，与所在目录对应。

正文用二级标题组织，例如：结论、排查步骤、处理方式、历史案例。

文末保留 `## 来源`。每条结论都要能追溯到来源，写成列表项。不要删除已有来源。

## 修改原则

1. 只合并、补充或标注过时，不删除已有事实。过时内容注明「已过时」并写明原因。
2. 不编造材料中没有的命令、阈值、时间或根因。不确定就写明不确定。
3. 同一告警、同一服务尽量更新已有页面，不要另起一页重复主题。
4. 案例页用相对链接指向告警页和服务页，例如 `[CPU 使用率过高](../alerts/HighCPUUsage.md)`。
5. 全文使用中文。告警名、服务名、指标名保留原文。
"""

_SELECT_SYSTEM = """你是运维 Wiki 编辑。根据给定材料和现有目录，选出需要创建或更新的页面。

约束：
1. 最多 {max_pages} 个页面，按重要性排序。
2. path 只能是 alerts/、services/、incidents/、topics/ 下的一层 .md 文件，例如 alerts/HighCPUUsage.md。
3. 已有同主题页面必须用 update，不要新建重复页。
4. 文件名只用字母、数字、下划线、连字符或中文，不要空格。
5. source_type 为 incident 时：必须包含一篇新的 incidents/日期_简述.md，以及对应的 alerts/告警名.md（已有则 update）。
6. 不要选择 index.md、log.md、WIKI.md。
"""

_SELECTION_JSON_HINT = (
    "请以 json 格式输出，不要附加说明。"
    '示例：{"pages":[{"path":"alerts/Example.md","action":"create","reason":""}]}'
)

_REWRITE_SYSTEM = """你是运维 Wiki 编辑。请输出这一页修改后的完整 Markdown，不要代码围栏，不要额外解释。

必须遵守：
1. 第一行是 `# 标题`，下一行是 `> 类型: {page_type} | 摘要: 一句话`。
2. 保留并合并已有事实，不删除已有结论；过时内容标注「已过时」。
3. 只使用给定材料中的事实，不编造命令、阈值或根因。
4. 文末保留 `## 来源`，已有来源条目原样保留，并追加本次来源。
5. 全文中文，专有名词保留原文。
"""


class WikiPathError(ValueError):
    """Wiki 路径不在白名单内或包含非法字符。"""


class WikiConflictError(Exception):
    """提案生成后页面已被其他人修改。"""

    def __init__(self, paths: list[str]):
        self.paths = paths
        super().__init__("页面已被修改，请重新生成")


class PlannedPage(BaseModel):
    """LLM 选出的待改页面。"""

    path: str = Field(description="相对路径，如 alerts/HighCPUUsage.md")
    action: str = Field(description="create 或 update")
    reason: str = Field(description="修改原因，一句话")


class WikiPageSelection(BaseModel):
    """页面选择结果。"""

    pages: list[PlannedPage] = Field(default_factory=list, description="待创建或更新的页面")


class WikiService:
    """运维 Wiki：提案、确认写入、目录重建。"""

    def __init__(self) -> None:
        raw_dir = Path(config.wiki_dir)
        if not raw_dir.is_absolute():
            raw_dir = _PROJECT_ROOT / raw_dir
        self.wiki_dir = raw_dir.resolve()
        self._write_lock = asyncio.Lock()
        self.ensure_layout()
        logger.info(f"运维 Wiki 服务初始化完成, dir={self.wiki_dir}")

    def ensure_layout(self) -> None:
        """补齐目录、写作规范和空目录页。"""
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        for category in ALLOWED_CATEGORIES:
            (self.wiki_dir / category).mkdir(parents=True, exist_ok=True)

        guide = self.wiki_dir / "WIKI.md"
        if not guide.exists():
            guide.write_text(WIKI_GUIDE, encoding="utf-8")

        log_file = self.wiki_dir / "log.md"
        if not log_file.exists():
            log_file.write_text("# Wiki 变更日志\n", encoding="utf-8")

        index_file = self.wiki_dir / "index.md"
        if not index_file.exists():
            index_file.write_text(self.build_index(), encoding="utf-8")

    def read_page(self, path: str) -> str:
        """读取白名单内的页面。根目录只允许 index.md、WIKI.md、log.md。"""
        _, full = self._resolve(path, allow_root=True)
        if not full.is_file():
            raise FileNotFoundError(path)
        return full.read_text(encoding="utf-8")

    def contains_source(self, source: str) -> bool:
        """判断向量库里的文件路径是否落在 Wiki 目录下。"""
        if not source:
            return False
        try:
            Path(source).resolve().relative_to(self.wiki_dir)
            return True
        except (ValueError, OSError):
            return False

    async def propose(
        self,
        source_type: str,
        session_id: str,
        question: str | None,
        answer: str,
    ) -> dict:
        """生成修改提案，不落盘。

        先让 LLM 从目录里选页，再逐页生成修改后的全文，并带上原内容的 hash。
        """
        self.ensure_layout()
        question = (question or "").strip()
        answer = (answer or "").strip()
        if not answer:
            raise WikiPathError("内容不能为空")

        index_text = self.read_page("index.md")
        guide = self.read_page("WIKI.md")
        selected = await self._select_pages(
            source_type, session_id, question, answer, index_text, guide
        )
        intents = self._normalize_intents(selected, source_type, question, answer)
        if not intents:
            raise RuntimeError("没有可更新的 Wiki 页面")

        changes = []
        for rel, reason in intents:
            _, full = self._resolve(rel, allow_root=False)
            before = full.read_text(encoding="utf-8") if full.is_file() else ""
            action = "update" if before else "create"
            after = await self._rewrite_page(
                rel=rel,
                action=action,
                reason=reason,
                before=before,
                source_type=source_type,
                session_id=session_id,
                question=question,
                answer=answer,
                guide=guide,
            )
            changes.append(
                {
                    "path": rel,
                    "action": action,
                    "reason": reason,
                    "before": before,
                    "after": after,
                    "base_hash": self._hash(before),
                }
            )

        logger.info(
            f"Wiki 提案已生成: source={source_type}, session={session_id}, "
            f"pages={[item['path'] for item in changes]}"
        )
        return {"changes": changes}

    async def confirm(self, changes: list[dict], source_type: str, session_id: str | None) -> dict:
        """确认写入。hash 不一致则整批拒绝，避免覆盖并发修改。"""
        if len(changes) > config.wiki_max_pages_per_proposal:
            raise WikiPathError("一次确认的页面数量超出限制")

        async with self._write_lock:
            with self._interprocess_lock():
                return await self._confirm_locked(changes, source_type, session_id)

    async def _confirm_locked(
        self, changes: list[dict], source_type: str, session_id: str | None
    ) -> dict:
        """在写锁内校验 hash、落盘并入库。"""
        prepared: list[tuple[Path, str, str, str]] = []
        seen: set[str] = set()
        conflicts: list[str] = []

        for change in changes:
            rel, full = self._resolve(str(change.get("path") or ""), allow_root=False)
            content = str(change.get("content") or "").strip()
            if not content:
                raise WikiPathError(f"页面内容不能为空: {rel}")
            if len(content) > config.wiki_max_page_chars:
                raise WikiPathError(f"页面内容超过 {config.wiki_max_page_chars} 字: {rel}")
            if rel in seen:
                raise WikiPathError(f"重复的页面路径: {rel}")
            seen.add(rel)

            current = full.read_text(encoding="utf-8") if full.is_file() else ""
            base_hash = str(change.get("base_hash") or "")
            if self._hash(current) != base_hash:
                conflicts.append(rel)
            action = "update" if current else "create"
            body = content if content.endswith("\n") else content + "\n"
            prepared.append((full, rel, body, action))

        if conflicts:
            logger.warning(f"Wiki 确认冲突: {conflicts}")
            raise WikiConflictError(conflicts)

        pages = []
        for full, rel, content, action in prepared:
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")
            chunk_count, index_error = self._index_page(full)
            pages.append(
                {
                    "path": rel,
                    "action": action,
                    "chunk_count": chunk_count,
                    "index_error": index_error,
                }
            )

        index_text = self.build_index()
        (self.wiki_dir / "index.md").write_text(index_text, encoding="utf-8")
        self._append_log(source_type, session_id, [rel for _, rel, _, _ in prepared])

        logger.info(f"Wiki 已确认写入: pages={[item['path'] for item in pages]}")
        return {"success": True, "pages": pages}

    @contextmanager
    def _interprocess_lock(self) -> Iterator[None]:
        """跨进程写锁。同一进程内还由 asyncio.Lock 串行。"""
        lock_path = self.wiki_dir / ".write.lock"
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        if not lock_path.exists() or lock_path.stat().st_size == 0:
            lock_path.write_bytes(b"\0")
        handle = lock_path.open("r+b")
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def _index_page(self, full: Path) -> tuple[int, str | None]:
        """把页面写入向量库。失败只返回错误，不回滚已经落盘的 Markdown。"""
        try:
            from app.services.vector_index_service import vector_index_service

            return vector_index_service.index_single_file(str(full)), None
        except Exception as exc:
            logger.error(f"Wiki 页面入库失败: {full.name}, {exc}")
            return 0, str(exc)

    def build_index(self) -> str:
        """按目录扫描页面标题和摘要，生成 index.md。不交给 LLM 修改。"""
        lines = [
            "# Wiki 目录",
            "",
            "本文件由系统自动生成，请勿手工编辑。",
            "",
        ]
        for category in ALLOWED_CATEGORIES:
            lines.append(f"## {CATEGORY_LABELS[category]}")
            lines.append("")
            folder = self.wiki_dir / category
            files = sorted(folder.glob("*.md")) if folder.is_dir() else []
            if not files:
                lines.append("- （暂无）")
            else:
                for file in files:
                    content = file.read_text(encoding="utf-8")
                    title, summary = self._parse_meta(content, file.stem)
                    rel = f"{category}/{file.name}"
                    lines.append(f"- [{title}]({rel})：{summary}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    async def _select_pages(
        self,
        source_type: str,
        session_id: str,
        question: str,
        answer: str,
        index_text: str,
        guide: str,
    ) -> list[PlannedPage]:
        """让 LLM 从目录中选出要改的页面。结构化输出失败时退回 JSON 解析。"""
        llm = self._wiki_llm()
        system = (
            _SELECT_SYSTEM.format(max_pages=config.wiki_max_pages_per_proposal)
            + "\n"
            + _SELECTION_JSON_HINT
        )
        human = self._material_prompt(source_type, session_id, question, answer, index_text, guide)
        messages = [SystemMessage(content=system), HumanMessage(content=human)]

        try:
            structured = llm.with_structured_output(WikiPageSelection)
            result = await structured.ainvoke(messages)
            if isinstance(result, WikiPageSelection):
                return result.pages
            if isinstance(result, dict):
                return WikiPageSelection.model_validate(result).pages
            pages = getattr(result, "pages", None)
            if pages is not None:
                return WikiPageSelection.model_validate({"pages": pages}).pages
        except Exception as exc:
            logger.warning(f"结构化选择 Wiki 页面失败，改为解析 JSON: {exc}")

        fallback = await llm.ainvoke(messages)
        raw = fallback.content if hasattr(fallback, "content") else str(fallback)
        return self._parse_selection(str(raw))

    async def _rewrite_page(
        self,
        rel: str,
        action: str,
        reason: str,
        before: str,
        source_type: str,
        session_id: str,
        question: str,
        answer: str,
        guide: str,
    ) -> str:
        """生成单页修改后的完整 Markdown，并补齐摘要行和来源小节。"""
        page_type = PAGE_TYPES[rel.split("/", 1)[0]]
        llm = self._wiki_llm()
        existing = before.strip() or "（新页面，当前没有正文）"
        human = (
            f"页面路径: {rel}\n"
            f"动作: {action}\n"
            f"修改原因: {reason}\n"
            f"来源类型: {source_type}\n"
            f"session_id: {session_id or 'unknown'}\n\n"
            f"【写作规范】\n{_clip(guide)}\n\n"
            f"【当前页面】\n{_clip(existing)}\n\n"
            f"【用户问题】\n{question or '（无）'}\n\n"
            f"【待沉淀内容】\n{_clip(answer)}\n"
        )
        response = await llm.ainvoke(
            [
                SystemMessage(content=_REWRITE_SYSTEM.format(page_type=page_type)),
                HumanMessage(content=human),
            ]
        )
        raw = response.content if hasattr(response, "content") else str(response)
        content = _strip_fence(str(raw))
        if not content.strip():
            raise RuntimeError(f"LLM 未生成页面内容: {rel}")
        return self._ensure_shape(content, rel, source_type, session_id, reason)

    def _normalize_intents(
        self,
        selected: list[PlannedPage],
        source_type: str,
        question: str,
        answer: str,
    ) -> list[tuple[str, str]]:
        """丢弃非法路径，并在诊断场景补上案例页和告警页。"""
        intents: list[tuple[str, str]] = []
        seen: set[str] = set()
        for page in selected:
            try:
                rel, _ = self._resolve(page.path, allow_root=False)
            except WikiPathError as exc:
                logger.info(f"忽略非法 Wiki 路径: {page.path}, {exc}")
                continue
            if rel in seen:
                continue
            seen.add(rel)
            reason = (page.reason or "").strip() or "合并本次材料"
            intents.append((rel, reason))

        material = f"{question}\n{answer}"
        if source_type == "incident":
            if not any(path.startswith("incidents/") for path, _ in intents):
                rel = self._fresh_incident_path(material)
                intents.insert(0, (rel, "记录本次诊断案例"))
            if not any(path.startswith("alerts/") for path, _ in intents):
                rel = self._alert_page_path(material)
                intents.insert(1, (rel, "把本次诊断补进告警页"))
        elif not intents:
            rel = f"topics/{self._slug(question or answer)}.md"
            intents.append((rel, "沉淀本次问答"))

        limit = config.wiki_max_pages_per_proposal
        if source_type != "incident":
            return intents[:limit]

        forced = [item for item in intents if item[0].startswith(("incidents/", "alerts/"))]
        rest = [item for item in intents if item not in forced]
        # 诊断提案至少保留案例页和告警页，其余页面再按上限截断
        kept_forced = forced[:limit]
        room = max(limit - len(kept_forced), 0)
        return kept_forced + rest[:room]

    def _fresh_incident_path(self, text: str) -> str:
        """生成不与现有文件冲突的诊断案例路径。"""
        date = datetime.now().strftime("%Y-%m-%d")
        slug = self._guess_alert(text) or "incident"
        rel = f"incidents/{date}_{slug}.md"
        if (self.wiki_dir / rel).exists():
            stamp = datetime.now().strftime("%H%M%S")
            rel = f"incidents/{date}_{slug}_{stamp}.md"
        return rel

    def _alert_page_path(self, text: str) -> str:
        """告警页路径。中文告警名会保留，无法识别时才用 PendingAlert。"""
        name = self._guess_alert(text) or "PendingAlert"
        try:
            rel, _ = self._resolve(f"alerts/{name}.md", allow_root=False)
            return rel
        except WikiPathError:
            return "alerts/PendingAlert.md"

    def _wiki_llm(self):
        """Wiki 改写用的模型，带超时，避免提案一直占着连接。"""
        return llm_factory.create_chat_model(
            temperature=0.2,
            streaming=False,
            timeout=config.wiki_llm_timeout_seconds,
        )

    def _resolve(self, rel_path: str, *, allow_root: bool) -> tuple[str, Path]:
        """把相对路径规范成 wiki 目录内的绝对路径。"""
        rel = (rel_path or "").strip().replace("\\", "/")
        if rel.startswith("./"):
            rel = rel[2:]
        if rel.lower().startswith("wiki/"):
            rel = rel[5:]
        rel = rel.lstrip("/")
        if not rel or ".." in Path(rel).parts:
            raise WikiPathError(f"非法 Wiki 路径: {rel_path}")

        parts = Path(rel).parts
        if len(parts) == 1:
            if not allow_root or parts[0] not in ROOT_READABLE:
                raise WikiPathError(f"不允许写入根目录文件: {rel}")
        elif len(parts) == 2:
            category, name = parts
            name = name.replace(" ", "_")
            if category not in ALLOWED_CATEGORIES or not name.endswith(".md"):
                raise WikiPathError(f"页面必须位于白名单目录: {rel}")
            stem = name[:-3]
            if not stem or not _SAFE_STEM.match(stem):
                raise WikiPathError(f"文件名包含非法字符: {name}")
            rel = f"{category}/{name}"
        else:
            raise WikiPathError(f"不支持多级目录: {rel}")

        full = (self.wiki_dir / rel).resolve()
        try:
            full.relative_to(self.wiki_dir)
        except ValueError as exc:
            raise WikiPathError(f"路径越界: {rel_path}") from exc
        return rel, full

    def _append_log(self, source_type: str, session_id: str | None, paths: list[str]) -> None:
        """追加一条变更记录。"""
        stamp = datetime.now().isoformat(timespec="seconds")
        line = (
            f"- {stamp} | {source_type} | session_id: {session_id or 'unknown'} | "
            f"pages: {', '.join(paths)}\n"
        )
        log_file = self.wiki_dir / "log.md"
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def _ensure_shape(
        self,
        content: str,
        rel: str,
        source_type: str,
        session_id: str,
        reason: str,
    ) -> str:
        """补齐标题、摘要引用和来源小节，避免页面无法被目录扫描。"""
        page_type = PAGE_TYPES[rel.split("/", 1)[0]]
        lines = content.splitlines()
        heading_idx = next((i for i, line in enumerate(lines) if line.startswith("# ")), None)
        if heading_idx is None:
            lines.insert(0, f"# {Path(rel).stem}")
            heading_idx = 0
        window = lines[heading_idx + 1 : heading_idx + 4]
        if not any(line.strip().startswith(">") for line in window):
            summary = (reason or Path(rel).stem).replace("\n", " ").strip()
            lines.insert(heading_idx + 1, f"> 类型: {page_type} | 摘要: {summary}")

        text = "\n".join(lines).strip()
        stamp = datetime.now().isoformat(timespec="seconds")
        bullet = f"- {stamp} | {source_type} | session_id: {session_id or 'unknown'}"
        if "## 来源" not in text:
            text = f"{text}\n\n## 来源\n{bullet}"
        elif bullet not in text:
            text = f"{text.rstrip()}\n{bullet}"
        return text if text.endswith("\n") else text + "\n"

    @staticmethod
    def _parse_meta(content: str, fallback_title: str) -> tuple[str, str]:
        """从页面提取一级标题和摘要行。"""
        title = fallback_title
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip() or fallback_title
                break
        summary = "暂无摘要"
        matched = re.search(r"摘要\s*[:：]\s*(.+)", content)
        if matched:
            summary = matched.group(1).strip() or summary
        return title, summary

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _guess_alert(self, text: str) -> str:
        """从诊断文本里提取告警名。识别不到时返回空字符串。"""
        patterns = (
            r"告警(?:名称|名)\s*[*`_]*\s*[:：]\s*[*`_\[]*([^\n\]`|]{1,40})",
            r"告警根因分析\s*\d*\s*[-–—:：]\s*\[?([^\]\n]{1,40})",
        )
        stopwords = {"名称", "名", "级别", "详情", "清单", "分析", "根因", "根因分析"}
        for pattern in patterns:
            matched = re.search(pattern, text or "")
            if not matched:
                continue
            slug = self._filename_slug(matched.group(1))
            if slug and slug not in stopwords:
                return slug
        return ""

    @staticmethod
    def _filename_slug(text: str) -> str:
        """保留中文和英文，收成安全文件名。"""
        cleaned = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", text or "")
        cleaned = re.sub(r"_+", "_", cleaned).strip("._")
        return (cleaned[:40] or "").rstrip("_")

    @staticmethod
    def _slug(text: str) -> str:
        """把标题或告警名收成安全文件名。"""
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{1,}", text or "")
        if tokens:
            slug = "_".join(tokens[:3])
        else:
            zh = re.findall(r"[\u4e00-\u9fff]{2,12}", text or "")
            slug = "_".join(zh[:2]) if zh else "page"
        slug = re.sub(r"[^\w\-\u4e00-\u9fff]", "_", slug)
        slug = re.sub(r"_+", "_", slug).strip("._")
        return (slug[:40] or "page").rstrip("_")

    @staticmethod
    def _material_prompt(
        source_type: str,
        session_id: str,
        question: str,
        answer: str,
        index_text: str,
        guide: str,
    ) -> str:
        return (
            f"来源类型: {source_type}\n"
            f"session_id: {session_id or 'unknown'}\n\n"
            f"【现有目录】\n{_clip(index_text)}\n\n"
            f"【写作规范】\n{_clip(guide)}\n\n"
            f"【用户问题】\n{question or '（无）'}\n\n"
            f"【待沉淀内容】\n{_clip(answer)}\n"
        )

    @staticmethod
    def _parse_selection(raw: str) -> list[PlannedPage]:
        """从模型原文里抽出 JSON 页面列表。"""
        text = _strip_fence(raw)
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            logger.warning("Wiki 页面选择结果不是 JSON")
            return []
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            logger.warning(f"Wiki 页面选择 JSON 解析失败: {exc}")
            return []
        pages = payload.get("pages") if isinstance(payload, dict) else None
        if not isinstance(pages, list):
            return []
        parsed: list[PlannedPage] = []
        for item in pages:
            if not isinstance(item, dict) or not item.get("path"):
                continue
            parsed.append(
                PlannedPage(
                    path=str(item.get("path")),
                    action=str(item.get("action") or "create"),
                    reason=str(item.get("reason") or ""),
                )
            )
        return parsed


def _clip(text: str, limit: int = _SOURCE_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n...（已截断）"


def _strip_fence(text: str) -> str:
    """去掉模型偶尔包上的 markdown 代码围栏。"""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


wiki_service = WikiService()
