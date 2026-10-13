"""
Executor 节点：执行单个步骤
基于 LangGraph 官方教程实现
"""

from typing import Any, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from langgraph.prebuilt import ToolNode
from loguru import logger

from app.config import config
from app.tools import get_current_time, retrieve_knowledge
from app.agent.mcp_client import get_mcp_client_with_retry
from .state import PlanExecuteState


def _normalize_tool_calls(tool_calls: Any) -> list[dict]:
    """把 LangChain 工具调用收成 {name, args}，便于轨迹评测比对。"""
    normalized: list[dict] = []
    for call in tool_calls or []:
        if isinstance(call, dict):
            name = str(call.get("name") or "")
            raw_args = call.get("args") if isinstance(call.get("args"), dict) else {}
        else:
            name = str(getattr(call, "name", "") or "")
            raw_args = getattr(call, "args", {}) or {}
            if not isinstance(raw_args, dict):
                raw_args = {}
        if name:
            normalized.append({"name": name, "args": raw_args})
    return normalized


def _message_usage(message: Any) -> tuple[int, int]:
    """从单次模型响应取出 prompt / completion token。没有用量时返回 0。"""
    meta = getattr(message, "usage_metadata", None) or {}
    if not isinstance(meta, dict):
        prompt = int(getattr(meta, "input_tokens", 0) or 0)
        completion = int(getattr(meta, "output_tokens", 0) or 0)
        return prompt, completion
    if meta.get("input_tokens") or meta.get("output_tokens"):
        return int(meta.get("input_tokens") or 0), int(meta.get("output_tokens") or 0)
    raw = getattr(message, "response_metadata", None) or {}
    usage = raw.get("token_usage") or raw.get("usage") or {} if isinstance(raw, dict) else {}
    if not isinstance(usage, dict):
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        return prompt, completion
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return prompt, completion


def _trace_item(
    task: str,
    calls: list[dict],
    prompt_tokens: int,
    completion_tokens: int,
    error: str = "",
) -> dict:
    """单步轨迹。calls 为空表示这一步没有调工具。"""
    item = {
        "step": task,
        "calls": calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if error:
        item["error"] = error
    return item


async def executor(state: PlanExecuteState) -> Dict[str, Any]:
    """
    执行节点：执行计划中的下一个步骤
    
    使用 LangGraph 的 ToolNode 自动处理工具调用
    """
    logger.info("=== Executor：执行步骤 ===")

    plan = state.get("plan", [])

    # 如果计划为空，不执行
    if not plan:
        logger.info("计划为空，跳过执行")
        return {}

    # 取出第一个步骤
    task = plan[0]
    logger.info(f"当前任务: {task}")
    calls: list[dict] = []
    prompt_tokens = 0
    completion_tokens = 0

    try:
        # 获取本地工具
        local_tools = [
            get_current_time,
            retrieve_knowledge
        ]

        # 获取 MCP 工具
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        # 合并所有工具
        all_tools = local_tools + mcp_tools

        # 创建 LLM（绑定工具）
        llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0
        )
        llm_with_tools = llm.bind_tools(all_tools)

        # 创建工具节点（自动执行工具调用）
        tool_node = ToolNode(all_tools)

        # 构建消息（只包含当前步骤，避免原始任务干扰）
        messages = [
            SystemMessage(content="""你是一个能力强大的助手，负责执行具体的任务步骤。

你可以使用各种工具来完成任务。对于每个步骤：
1. 理解步骤的目标
2. 选择合适的工具，如果已经指定了工具，则使用指定的工具
3. 调用工具获取信息
4. 返回执行结果

注意：
- 如果工具调用失败，请说明失败原因
- 不要编造数据，只返回实际获取的信息
- 执行结果要清晰、准确
- 专注于当前步骤，不要考虑其他任务"""),
            HumanMessage(content=f"请执行以下任务: {task}")
        ]

        # 第一步：LLM 决定是否调用工具
        llm_response = await llm_with_tools.ainvoke(messages)
        in_tokens, out_tokens = _message_usage(llm_response)
        prompt_tokens += in_tokens
        completion_tokens += out_tokens
        logger.info(f"LLM 响应类型: {type(llm_response)}")

        # 第二步：如果有工具调用，执行工具
        if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
            calls = _normalize_tool_calls(llm_response.tool_calls)
            logger.info(f"检测到 {len(calls)} 个工具调用")

            # 使用 ToolNode 自动执行工具
            messages.append(llm_response)
            tool_messages = await tool_node.ainvoke({"messages": messages})

            # 第三步：将工具结果返回给 LLM 生成最终答案
            messages.extend(tool_messages["messages"])
            final_response = await llm_with_tools.ainvoke(messages)
            in_tokens, out_tokens = _message_usage(final_response)
            prompt_tokens += in_tokens
            completion_tokens += out_tokens
            content = getattr(final_response, "content", None)
            result = content if isinstance(content, str) else str(final_response)
        else:
            # 没有工具调用，直接使用 LLM 的输出
            logger.info("LLM 未调用工具，直接返回结果")
            result = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

        logger.info(f"步骤执行完成，结果长度: {len(result)}")

        # 返回更新：移除已执行的步骤，添加执行历史
        return {
            "plan": plan[1:],  # 移除第一个步骤
            "past_steps": [(task, result)],  # 使用 operator.add 追加
            "tool_trace": [_trace_item(task, calls, prompt_tokens, completion_tokens)],
        }

    except Exception as e:
        logger.error(f"执行步骤失败: {e}", exc_info=True)
        return {
            "plan": plan[1:],
            "past_steps": [(task, f"执行失败: {str(e)}")],
            "tool_trace": [
                _trace_item(task, calls, prompt_tokens, completion_tokens, error=str(e))
            ],
        }
