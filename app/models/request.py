"""请求数据模型

定义 API 请求的 Pydantic 模型
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


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


class WikiProposeRequest(BaseModel):
    """Wiki 修改提案请求。诊断场景下 answer 为报告正文，question 可空。"""

    source_type: Literal["qa", "incident"] = Field(..., description="来源：qa 问答 / incident 诊断")
    session_id: str = Field(default="", description="会话 ID")
    question: str | None = Field(default=None, description="用户问题，诊断场景可空")
    answer: str = Field(..., description="回答或诊断报告", min_length=1)

    @model_validator(mode="after")
    def qa_requires_question(self) -> "WikiProposeRequest":
        """问答沉淀必须带上原问题，诊断报告允许只有正文。"""
        if self.source_type == "qa" and not (self.question or "").strip():
            raise ValueError("问答沉淀需要提供问题")
        return self


class WikiChange(BaseModel):
    """人工确认后的单页修改"""

    path: str = Field(..., description="相对 wiki 根目录的路径", min_length=1)
    action: Literal["create", "update"] = Field(..., description="create 新建，update 更新")
    content: str = Field(..., description="确认后的完整 Markdown", min_length=1)
    base_hash: str = Field(..., description="提案生成时原页面内容的 md5", min_length=1)


class WikiConfirmRequest(BaseModel):
    """确认写入 Wiki"""

    changes: list[WikiChange] = Field(..., description="要写入的页面", min_length=1)
    source_type: Literal["qa", "incident"] = Field(..., description="来源：qa 问答 / incident 诊断")
    session_id: str | None = Field(default=None, description="会话 ID（可选）")
