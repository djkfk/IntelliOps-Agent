"""知识点沉淀接口

提供 Chat 问答提炼草稿与人工确认入库能力。
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.models.request import KnowledgeConfirmRequest, KnowledgeDraftRequest
from app.services.knowledge_precipitate_service import knowledge_precipitate_service

router = APIRouter()


@router.post("/knowledge/draft")
async def create_knowledge_draft(request: KnowledgeDraftRequest):
    """将本轮问答提炼为可编辑的 FAQ 草稿"""
    try:
        logger.info(
            f"[会话 {request.session_id}] 收到知识点草稿请求: "
            f"question={request.question[:80]}"
        )
        draft = await knowledge_precipitate_service.generate_draft(
            session_id=request.session_id,
            question=request.question,
            answer=request.answer,
        )
        return {
            "code": 200,
            "message": "success",
            "data": draft,
        }
    except ValueError as e:
        logger.warning(f"知识点草稿参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"知识点草稿生成失败: {e}")
        return {
            "code": 500,
            "message": "error",
            "data": {
                "title": None,
                "content": None,
                "errorMessage": str(e),
            },
        }


@router.post("/knowledge/confirm")
async def confirm_knowledge(request: KnowledgeConfirmRequest):
    """人工确认后落盘并写入向量知识库"""
    try:
        logger.info(
            f"[会话 {request.session_id or '-'}] 确认沉淀知识点: title={request.title}"
        )
        result = knowledge_precipitate_service.confirm_and_index(
            title=request.title,
            content=request.content,
            session_id=request.session_id,
        )
        return {
            "code": 200,
            "message": "success",
            "data": result,
        }
    except ValueError as e:
        logger.warning(f"知识点确认参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"知识点确认入库失败: {e}")
        return {
            "code": 500,
            "message": "error",
            "data": {
                "success": False,
                "filename": None,
                "file_path": None,
                "chunk_count": 0,
                "errorMessage": str(e),
            },
        }
