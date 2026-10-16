"""运维 Wiki 读取工具。"""

from langchain_core.tools import tool
from loguru import logger

from app.services.wiki_service import WikiPathError, wiki_service

_MAX_CHARS = 8000


@tool
def read_wiki(path: str = "index.md") -> str:
    """读取运维 Wiki。先读 index.md 找到相关页面路径，再读具体页面。
    Wiki 是整理过的结论与历史案例；需要原文细节或证据时再用 retrieve_knowledge。"""
    target = (path or "index.md").strip() or "index.md"
    try:
        content = wiki_service.read_page(target)
    except WikiPathError:
        logger.info(f"Wiki 路径不合法: {target}")
        return "无法读取 Wiki：路径不合法。请先读取 index.md。"
    except FileNotFoundError:
        logger.info(f"Wiki 页面不存在: {target}")
        return f"页面不存在：{target}。请先读取 index.md 确认路径。"
    except Exception as exc:
        logger.error(f"读取 Wiki 失败: {target}, {exc}")
        return f"读取 Wiki 失败：{exc}"

    if len(content) <= _MAX_CHARS:
        return content
    return content[:_MAX_CHARS] + "\n\n...（内容已截断，可按章节继续提问）"
