"""
通用 Plan-Execute-Replan 状态定义
基于 LangGraph 官方教程实现
"""

from typing import Annotated, List, NotRequired, TypedDict
import operator


class PlanExecuteState(TypedDict):
    """Plan-Execute-Replan 状态"""

    # 用户输入（任务描述）
    input: str

    # 执行计划（步骤列表）
    plan: List[str]

    # 已执行的步骤历史
    # 使用 operator.add 实现追加式更新（而非覆盖）
    past_steps: Annotated[List[tuple], operator.add]

    # 每步实际工具调用，供轨迹评测断言。项为 {step, calls, prompt_tokens, completion_tokens}
    tool_trace: NotRequired[Annotated[List[dict], operator.add]]

    # 最终响应/报告
    response: str
