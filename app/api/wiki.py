"""运维 Wiki 接口

提供修改提案与人工确认写入。
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.models.request import WikiConfirmRequest, WikiProposeRequest
from app.services.wiki_service import WikiConflictError, WikiPathError, wiki_service

router = APIRouter()


@router.post("/wiki/propose")
async def propose_wiki_changes(request: WikiProposeRequest):
    """根据问答或诊断报告生成 Wiki 修改提案，不落盘。"""
    try:
        logger.info(
            f"[会话 {request.session_id or '-'}] Wiki 提案: source={request.source_type}"
        )
        proposal = await wiki_service.propose(
            source_type=request.source_type,
            session_id=request.session_id,
            question=request.question,
            answer=request.answer,
        )
        return {"code": 200, "message": "success", "data": proposal}
    except WikiPathError as exc:
        logger.warning(f"Wiki 提案参数错误: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Wiki 提案生成失败: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/wiki/confirm")
async def confirm_wiki_changes(request: WikiConfirmRequest):
    """人工确认后写入 Wiki 并重新入库。"""
    try:
        logger.info(
            f"[会话 {request.session_id or '-'}] 确认 Wiki: "
            f"pages={[item.path for item in request.changes]}"
        )
        result = await wiki_service.confirm(
            changes=[item.model_dump() for item in request.changes],
            source_type=request.source_type,
            session_id=request.session_id,
        )
        return {"code": 200, "message": "success", "data": result}
    except WikiConflictError as exc:
        logger.warning(f"Wiki 确认冲突: {exc.paths}")
        raise HTTPException(status_code=409, detail="页面已被修改，请重新生成") from exc
    except WikiPathError as exc:
        logger.warning(f"Wiki 确认参数错误: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Wiki 确认写入失败: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
