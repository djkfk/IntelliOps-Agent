"""请求数据模型

定义 API 请求的 Pydantic 模型
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求"""

    id: str = Field(..., description="会话 ID", alias="Id")
    question: str = Field(..., description="用户问题", alias="Question")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Id": "session-123",
                "Question": "什么是向量数据库？"
            }
        }


class ClearRequest(BaseModel):
    """清空会话请求"""

    session_id: str = Field(..., description="会话 ID", alias="sessionId")

    class Config:
        populate_by_name = True


class KnowledgeDraftRequest(BaseModel):
    """知识点草稿生成请求"""

    session_id: str = Field(..., description="会话 ID")
    question: str = Field(..., description="用户问题", min_length=1)
    answer: str = Field(..., description="助手回答", min_length=1)

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session-123",
                "question": "CPU 使用率过高怎么排查？",
                "answer": "可以先查看 top 进程，再核对近期发布与监控指标。",
            }
        }


class KnowledgeConfirmRequest(BaseModel):
    """知识点确认入库请求"""

    title: str = Field(..., description="知识点标题", min_length=1)
    content: str = Field(..., description="FAQ Markdown 正文", min_length=1)
    session_id: str | None = Field(default=None, description="会话 ID（可选）")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "CPU 使用率过高排查",
                "content": "# CPU 使用率过高排查\n\n## 问题\n...",
                "session_id": "session-123",
            }
        }
