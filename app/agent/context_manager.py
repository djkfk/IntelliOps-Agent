"""对话上下文管理

超过 token 阈值时，将旧消息压成累积摘要，并保留最近若干条原文。
基于 LangChain SummarizationMiddleware，自动保证 tool_call / ToolMessage 成对保留。
"""

from collections.abc import Iterable, Sequence
from contextvars import ContextVar
from textwrap import dedent
from typing import Any

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import BaseMessage, RemoveMessage
from langchain_core.messages.utils import get_buffer_string
from langchain_qwq import ChatQwen
from langgraph.runtime import Runtime
from loguru import logger

from app.config import config

SUMMARY_PROMPT = dedent(
    """
    你是对话上下文压缩助手。请从下列历史中提取后续对话仍需要的要点，
    输出将替换这些历史消息。只输出摘要正文，不要前后缀。

    请按以下小节填写；某节没有内容时写「无」：

    ## 用户意图
    用户的核心目标和当前任务。

    ## 关键信息
    已确认的事实、结论、约束、工具查询到的重要结果。

    ## 未决事项
    尚未完成或仍需跟进的问题。

    <messages>
    {messages}
    </messages>
    """
).strip()

_context_usage: ContextVar[dict[str, Any] | None] = ContextVar("context_usage", default=None)


def estimate_tokens(messages: Iterable[Any]) -> int:
    """按字符数估算 token，中文场景偏保守（约 1 字 ≈ 1 token）。"""
    message_list = [m for m in messages if not isinstance(m, RemoveMessage)]
    if not message_list:
        return 0
    try:
        text = get_buffer_string(message_list)
    except Exception:
        parts: list[str] = []
        for msg in message_list:
            content = getattr(msg, "content", msg)
            parts.append(content if isinstance(content, str) else str(content))
        text = "".join(parts)
    return max(0, len(text))


def build_context_snapshot(
    messages: Sequence[Any] | None = None,
    *,
    compressed: bool = False,
    tokens_before: int | None = None,
    tokens: int | None = None,
) -> dict[str, Any]:
    """构造前端可用的上下文占用快照。"""
    usable = [m for m in (messages or []) if not isinstance(m, RemoveMessage)]
    token_count = estimate_tokens(usable) if tokens is None else tokens
    max_tokens = max(config.context_max_tokens, 1)
    snapshot: dict[str, Any] = {
        "enabled": config.context_enabled,
        "tokens": token_count,
        "max_tokens": config.context_max_tokens,
        "percent": min(100, int(token_count / max_tokens * 100)),
        "message_count": len(usable),
        "compressed": compressed,
    }
    if tokens_before is not None:
        snapshot["tokens_before"] = tokens_before
    return snapshot


def record_context_snapshot(
    messages: Sequence[Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """写入当前请求的上下文占用，供 SSE / 接口读取。"""
    snapshot = build_context_snapshot(messages, **kwargs)
    _context_usage.set(snapshot)
    return snapshot


def get_context_usage() -> dict[str, Any] | None:
    """读取本轮请求记录的上下文占用。"""
    return _context_usage.get()


def reset_context_usage() -> None:
    """请求开始时清空上一轮残留。"""
    _context_usage.set(None)


class ContextSummarizationMiddleware(SummarizationMiddleware):
    """官方摘要中间件 + 占用记录 / 压缩日志。"""

    def before_model(self, state: dict[str, Any], runtime: Runtime) -> dict[str, Any] | None:
        return self._record_usage(state["messages"], super().before_model(state, runtime))

    async def abefore_model(
        self, state: dict[str, Any], runtime: Runtime
    ) -> dict[str, Any] | None:
        return self._record_usage(
            state["messages"],
            await super().abefore_model(state, runtime),
        )

    @staticmethod
    def _record_usage(
        messages: Sequence[BaseMessage],
        result: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        tokens_before = estimate_tokens(messages)
        if result is None:
            record_context_snapshot(messages, compressed=False)
            return None

        kept = [m for m in result.get("messages", []) if not isinstance(m, RemoveMessage)]
        snapshot = record_context_snapshot(
            kept,
            compressed=True,
            tokens_before=tokens_before,
        )
        logger.info(
            "上下文已压缩: {} 条/约 {} tokens -> {} 条/约 {} tokens",
            len(messages),
            tokens_before,
            snapshot["message_count"],
            snapshot["tokens"],
        )
        return result


def build_context_middleware() -> ContextSummarizationMiddleware | None:
    """按配置构建上下文压缩中间件；关闭时返回 None。"""
    if not config.context_enabled:
        logger.info("上下文管理已关闭")
        return None

    summary_model = ChatQwen(
        model=config.rag_model,
        api_key=config.dashscope_api_key,
        temperature=0,
        streaming=False,
    )
    logger.info(
        "上下文管理已启用: 阈值={} tokens, 保留最近 {} 条",
        config.context_max_tokens,
        config.context_keep_messages,
    )
    return ContextSummarizationMiddleware(
        model=summary_model,
        trigger=("tokens", config.context_max_tokens),
        keep=("messages", config.context_keep_messages),
        token_counter=estimate_tokens,
        summary_prompt=SUMMARY_PROMPT,
        trim_tokens_to_summarize=8000,
    )
