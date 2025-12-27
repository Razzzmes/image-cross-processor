from fastapi import APIRouter, HTTPException, status, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any

from app.services.captcha_service import captcha_service
from app.models.schemas import CaptchaResponse, CaptchaRequest, ErrorResponse

router = APIRouter()

@router.get(
    "/generate",
    response_model=CaptchaResponse,
    summary="Генерация новой капчи",
    description="Генерирует новое изображение капчи с уникальным ID",
    tags=["captcha"]
)
async def generate_captcha() -> Dict[str, Any]:
    """
    Генерирует новую капчу для защиты от ботов.
    
    Returns:
        CaptchaResponse: Объект с данными капчи
    """
    try:
        captcha_data = captcha_service.generate_captcha()
        return {
            "captcha_id": captcha_data["captcha_id"],
            "text": captcha_data["text"],  # В реальном приложении не отправляем!
            "image": captcha_data["image"],
            "expires_at": captcha_data["expires_at"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка генерации капчи: {str(e)}"
        )

@router.post(
    "/validate",
    summary="Проверка капчи",
    description="Проверяет введенную пользователем капчу",
    tags=["captcha"]
)
async def validate_captcha(request: CaptchaRequest) -> Dict[str, Any]:
    """
    Проверяет правильность введенной капчи.
    
    Args:
        request: CaptchaRequest с ID капчи и введенным текстом
        
    Returns:
        Dict с результатом проверки
    """
    try:
        is_valid = captcha_service.validate_captcha(
            captcha_id=request.captcha_id,
            user_input=request.captcha_text
        )
        
        if is_valid:
            return {
                "success": True,
                "message": "Капча верна",
                "valid": True
            }
        else:
            return {
                "success": False,
                "message": "Неверная капча или срок действия истек",
                "valid": False
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка проверки капчи: {str(e)}"
        )

@router.get(
    "/status",
    summary="Статус капчи",
    description="Возвращает информацию о капче по ID",
    tags=["captcha"]
)
async def get_captcha_status(captcha_id: str) -> Dict[str, Any]:
    """
    Получает статус капчи по ID.
    
    Args:
        captcha_id: ID капчи
        
    Returns:
        Dict с информацией о капче
    """
    captcha_info = captcha_service.get_captcha_info(captcha_id)
    
    if not captcha_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Капча не найдена или срок действия истек"
        )
    
    return {
        "success": True,
        "captcha_id": captcha_id,
        "created_at": captcha_info["created_at"].isoformat(),
        "expires_at": captcha_info["expires_at"].isoformat(),
        "is_expired": captcha_info["expires_at"] < captcha_service._cleanup_expired_captchas
    }

@router.get(
    "/stats",
    summary="Статистика капч",
    description="Возвращает статистику по активным капчам",
    tags=["captcha"]
)
async def get_captcha_stats() -> Dict[str, Any]:
    """
    Возвращает статистику по активным капчам.
    
    Returns:
        Dict со статистикой
    """
    active_count = captcha_service.get_active_captchas_count()
    
    return {
        "success": True,
        "active_captchas": active_count,
        "captcha_length": captcha_service.captcha_generator._truefonts,
        "expire_seconds": 300  # 5 минут
    }