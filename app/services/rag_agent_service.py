"""RAG Agent 服务 - 基于 LangGraph 的智能代理

使用 langchain_qwq 的 ChatQwen 原生集成，
支持真正的流式输出和更好的模型适配。
"""

import asyncio
from typing import Any, AsyncGenerator, Dict

from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_qwq import ChatQwen
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from app.agent.context_manager import (
    build_context_middleware,
    build_context_snapshot,
    get_context_usage,
    reset_context_usage,
)
from app.agent.mcp_client import get_mcp_client_with_retry
from app.config import config
from app.tools import get_current_time, retrieve_knowledge
from app.tools.knowledge_tool import (
    clear_last_rag_result,
    get_last_rag_result,
    get_rag_phase,
)

# 阿里千问大模型和langchain集成参考： https://docs.langchain.com/oss/python/integrations/chat/qwen
# 注意：需要配置环境变量 DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1 否则默认访问的是新加坡站点
# 同时也需要配置环境变量 DASHSCOPE_API_KEY=your_api_key


class RagAgentService:
    """RAG Agent 服务 - 使用 LangGraph + ChatQwen 原生集成"""

    def __init__(self, streaming: bool = True):
        """初始化 RAG Agent 服务

        Args:
            streaming: 是否启用流式输出，默认为 True
        """
        self.model_name = config.rag_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()


        self.model = ChatQwen(
            model=self.model_name,
            api_key=config.dashscope_api_key,
            temperature=0.7,
            streaming=streaming,
        )

        # 定义基础工具
        self.tools = [retrieve_knowledge, get_current_time]

        # MCP 客户端（延迟初始化，使用全局管理）
        self.mcp_tools: list = []

        # 创建内存检查点（用于会话管理）
        self.checkpointer = MemorySaver()

        # Agent 初始化（会在异步方法中完成）
        self.agent = None
        self._agent_initialized = False

        logger.info(f"RAG Agent 服务初始化完成 (ChatQwen), model={self.model_name}, streaming={streaming}")

    async def _initialize_agent(self):
        """异步初始化 Agent（包括 MCP 工具）"""
        if self._agent_initialized:
            return

        # 使用全局 MCP 客户端管理器（带重试拦截器）
        mcp_client = await get_mcp_client_with_retry()

        # 获取 MCP 工具
        mcp_tools = await mcp_client.get_tools()
        logger.info(f"成功加载 {len(mcp_tools)} 个 MCP 工具")

        # 将 MCP 工具添加到实例变量中
        self.mcp_tools = mcp_tools

        # 合并所有工具
        all_tools = self.tools + self.mcp_tools

        middleware = []
        context_middleware = build_context_middleware()
        if context_middleware is not None:
            middleware.append(context_middleware)

        self.agent = create_agent(
            self.model,
            tools=all_tools,
            system_prompt=self.system_prompt,
            middleware=middleware,
            checkpointer=self.checkpointer,
        )

        self._agent_initialized = True


        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")

    def _load_checkpoint_messages(self, session_id: str) -> list:
        """从 checkpointer 读取当前会话消息。"""
        try:
            thread_config = {"configurable": {"thread_id": session_id}}
            checkpoint_tuple = self.checkpointer.get(thread_config)
            if not checkpoint_tuple:
                return []
            if hasattr(checkpoint_tuple, "checkpoint"):
                checkpoint_data = checkpoint_tuple.checkpoint  # type: ignore[union-attr]
            else:
                checkpoint_data = checkpoint_tuple[0] if checkpoint_tuple else {}
            return checkpoint_data.get("channel_values", {}).get("messages", []) or []
        except Exception as exc:
            logger.debug(f"读取会话消息失败: {session_id}, {exc}")
            return []

    def _resolve_context_snapshot(self, session_id: str, extra_messages: list | None = None) -> dict:
        """按检查点估算当前占用；若本轮发生过压缩则带上压缩标记。"""
        messages = list(self._load_checkpoint_messages(session_id))
        if extra_messages:
            messages.extend(extra_messages)
        snapshot = build_context_snapshot(messages)
        recorded = get_context_usage()
        if recorded and recorded.get("compressed"):
            snapshot["compressed"] = True
            if recorded.get("tokens_before") is not None:
                snapshot["tokens_before"] = recorded["tokens_before"]
        return snapshot

    def _build_system_prompt(self) -> str:
        """
        构建系统提示词

        注意：LangChain 框架会自动将工具信息传递给 LLM，
        因此系统提示词中无需列举具体的工具列表。

        Returns:
            str: 系统提示词
        """
        from textwrap import dedent

        return dedent("""
            你是一个专业的AI助手，能够使用多种工具来帮助用户解决问题。

            工作原则:
            1. 理解用户需求，选择合适的工具来完成任务
            2. 当需要获取实时信息或专业知识时，主动使用相关工具
            3. 基于工具返回的结果提供准确、专业的回答
            4. 如果工具无法提供足够信息，请诚实地告知用户

            回答要求:
            - 保持友好、专业的语气
            - 回答简洁明了，重点突出
            - 基于事实，不编造信息
            - 如有不确定的地方，明确说明

            请根据用户的问题，灵活使用可用工具，提供高质量的帮助。
        """).strip()

    async def query(
        self,
        question: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        非流式处理用户问题（一次性返回完整答案）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Returns:
            Dict[str, Any]: {"answer": str, "rag": dict | None}
        """
        try:
            await self._initialize_agent()
            clear_last_rag_result()
            reset_context_usage()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（非流式）: {question}")

            # 系统提示由 create_agent(system_prompt=...) 注入，此处只追加本轮用户问题
            messages = [HumanMessage(content=question)]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            result = await self.agent.ainvoke(
                input=agent_input,
                config=config_dict,
            )

            # 提取最终答案
            messages_result = result.get("messages", [])
            answer = ""
            if messages_result:
                last_message = messages_result[-1]
                answer = last_message.content if hasattr(last_message, 'content') else str(last_message)

                # 记录工具调用
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                    logger.info(f"[会话 {session_id}] Agent 调用了工具: {tool_names}")

                logger.info(f"[会话 {session_id}] RAG Agent 查询完成（非流式）")
            else:
                logger.warning(f"[会话 {session_id}] Agent 返回结果为空")

            rag_meta = get_last_rag_result()
            return {
                "answer": answer,
                "rag": rag_meta,
                "context": self._resolve_context_snapshot(session_id),
            }

        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（非流式）: {e}")
            raise

    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理用户问题（逐步返回答案片段）

        Yields:
            Dict[str, Any]: type = status | tool_call | search_results | content | complete | error
        """
        try:
            await self._initialize_agent()
            clear_last_rag_result()
            reset_context_usage()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（流式）: {question}")

            # 立刻推送，刷新 SSE，前端可先显示头像与思考中
            yield {
                "type": "status",
                "data": {"stage": "thinking", "message": "思考中…"},
            }
            yield {
                "type": "context",
                "data": build_context_snapshot(
                    list(self._load_checkpoint_messages(session_id))
                    + [HumanMessage(content=question)]
                ),
            }

            messages = [HumanMessage(content=question)]
            agent_input = {"messages": messages}
            config_dict = {"configurable": {"thread_id": session_id}}

            event_queue: asyncio.Queue = asyncio.Queue()
            rag_searching_emitted = False
            rag_result_emitted = False
            watch_stop = asyncio.Event()

            async def _watch_rag_phase() -> None:
                """轮询工具侧写入的检索阶段，及时下发过程提示"""
                nonlocal rag_searching_emitted, rag_result_emitted
                while not watch_stop.is_set():
                    phase = get_rag_phase()
                    if phase == "searching" and not rag_searching_emitted:
                        rag_searching_emitted = True
                        logger.info(f"[会话 {session_id}] RAG 阶段: searching → 推送前端")
                        await event_queue.put(
                            (
                                "event",
                                {
                                    "type": "tool_call",
                                    "data": {
                                        "tool": "retrieve_knowledge",
                                        "status": "start",
                                        "message": "正在检索知识库…",
                                    },
                                },
                                None,
                            )
                        )
                    elif phase == "done" and not rag_result_emitted:
                        # 若轮询错过 searching，补发 start，保证前端有过程提示
                        if not rag_searching_emitted:
                            rag_searching_emitted = True
                            await event_queue.put(
                                (
                                    "event",
                                    {
                                        "type": "tool_call",
                                        "data": {
                                            "tool": "retrieve_knowledge",
                                            "status": "start",
                                            "message": "正在检索知识库…",
                                        },
                                    },
                                    None,
                                )
                            )
                        rag_meta = get_last_rag_result() or {
                            "hit": False,
                            "count": 0,
                            "sources": [],
                            "message": "知识库未命中相关内容",
                        }
                        rag_result_emitted = True
                        logger.info(
                            f"[会话 {session_id}] RAG 阶段: done → 推送前端 "
                            f"hit={rag_meta.get('hit')} count={rag_meta.get('count')}"
                        )
                        await event_queue.put(
                            ("event", {"type": "search_results", "data": rag_meta}, None)
                        )
                        await event_queue.put(
                            (
                                "event",
                                {
                                    "type": "tool_call",
                                    "data": {
                                        "tool": "retrieve_knowledge",
                                        "status": "end",
                                        "message": rag_meta.get("message", ""),
                                        "hit": rag_meta.get("hit", False),
                                        "count": rag_meta.get("count", 0),
                                    },
                                },
                                None,
                            )
                        )
                    await asyncio.sleep(0.03)

            async def _produce_tokens() -> None:
                try:
                    async for token, metadata in self.agent.astream(
                        input=agent_input,
                        config=config_dict,
                        stream_mode="messages",
                    ):
                        await event_queue.put(("token", token, metadata))
                    await event_queue.put(("done", None, None))
                except Exception as exc:
                    await event_queue.put(("error", exc, None))

            watcher = asyncio.create_task(_watch_rag_phase())
            producer = asyncio.create_task(_produce_tokens())

            try:
                while True:
                    kind, payload, metadata = await event_queue.get()

                    if kind == "event":
                        yield payload
                        continue

                    if kind == "error":
                        raise payload

                    if kind == "done":
                        break

                    token = payload
                    node_name = (
                        metadata.get("langgraph_node", "unknown")
                        if isinstance(metadata, dict)
                        else "unknown"
                    )
                    message_type = type(token).__name__

                    if message_type in ("AIMessage", "AIMessageChunk") or isinstance(
                        token, (AIMessage, AIMessageChunk)
                    ):
                        content_blocks = getattr(token, "content_blocks", None)
                        if content_blocks and isinstance(content_blocks, list):
                            for block in content_blocks:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text_content = block.get("text", "")
                                    if text_content:
                                        yield {
                                            "type": "content",
                                            "data": text_content,
                                            "node": node_name,
                                        }
            finally:
                watch_stop.set()
                # 再等一小会，让 watcher 把 done 阶段事件入队
                await asyncio.sleep(0.08)
                while True:
                    try:
                        kind, payload, _metadata = event_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if kind == "event":
                        yield payload
                for task in (watcher, producer):
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

            # 兜底补发检索结果
            if not rag_result_emitted:
                rag_meta = get_last_rag_result()
                if rag_meta is not None:
                    yield {"type": "search_results", "data": rag_meta}
                    yield {
                        "type": "tool_call",
                        "data": {
                            "tool": "retrieve_knowledge",
                            "status": "end",
                            "message": rag_meta.get("message", ""),
                            "hit": rag_meta.get("hit", False),
                            "count": rag_meta.get("count", 0),
                        },
                    }

            logger.info(f"[会话 {session_id}] RAG Agent 查询完成（流式）")
            context_data = self._resolve_context_snapshot(session_id)
            yield {"type": "context", "data": context_data}
            yield {
                "type": "complete",
                "data": {"rag": get_last_rag_result(), "context": context_data},
            }

        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（流式）: {e}")
            yield {
                "type": "error",
                "data": str(e),
            }
            raise

    def get_session_history(self, session_id: str) -> list:
        """
        获取会话历史（从 MemorySaver checkpointer 中读取）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            list: 消息历史列表 [{"role": "user|assistant", "content": "...", "timestamp": "..."}]
        """
        try:
            messages = self._load_checkpoint_messages(session_id)
            if not messages:
                logger.info(f"获取会话历史: {session_id}, 消息数量: 0")
                return []
            
            # 转换为前端需要的格式
            history = []
            for msg in messages:
                # 跳过系统消息和内部压缩摘要
                if isinstance(msg, SystemMessage):
                    continue
                if getattr(msg, "additional_kwargs", {}).get("lc_source") == "summarization":
                    continue
                    
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, 'content') else str(msg)
                
                # 提取时间戳（如果有的话）
                timestamp = getattr(msg, 'timestamp', None)
                if timestamp:
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": timestamp
                    })
                else:
                    from datetime import datetime
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": datetime.now().isoformat()
                    })
            
            logger.info(f"获取会话历史: {session_id}, 消息数量: {len(history)}")
            return history
            
        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []

    def clear_session(self, session_id: str) -> bool:
        """
        清空会话历史（从 MemorySaver checkpointer 中删除）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            bool: 是否成功
        """
        try:
            # 使用 checkpointer 的 delete_thread 方法删除该 thread 的所有检查点
            self.checkpointer.delete_thread(session_id)
            
            logger.info(f"已清除会话历史: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理 RAG Agent 服务资源...")
            # MCP 客户端由全局管理器统一管理，无需手动清理
            logger.info("RAG Agent 服务资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


# 全局单例 - 启用流式输出
rag_agent_service = RagAgentService(streaming=True)
