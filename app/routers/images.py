import io
import base64
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import (
    APIRouter, 
    UploadFile, 
    File, 
    Form, 
    HTTPException, 
    status,
    Request,
    Depends
)
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from app.models.enums import CrossType
from app.models.schemas import (
    ProcessImageRequest,
    ProcessImageResponse,
    ErrorResponse,
    HistogramResponse
)
from app.services.image_service import image_service
from app.services.captcha_service import captcha_service
from app.utils.file_utils import FileUtils
from app.config import settings

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
file_utils = FileUtils()

@router.post(
    "/process",
    response_model=ProcessImageResponse,
    summary="Обработка изображения",
    description="Загружает изображение, рисует крест и возвращает результат с гистограммами",
    tags=["images"]
)
async def process_image(
    file: UploadFile = File(..., description="Изображение для обработки"),
    cross_type: CrossType = Form(..., description="Тип креста (horizontal/vertical/both)"),
    color: str = Form(..., description="Цвет креста в HEX формате (#RRGGBB)"),
    width: int = Form(10, description="Ширина креста в пикселях (1-50)", ge=1, le=50),
    captcha: str = Form(..., description="Введенная капча"),
    captcha_id: Optional[str] = Form(None, description="ID капчи (если используется)")
):
    """
    Обрабатывает загруженное изображение:
    1. Проверяет капчу
    2. Рисует крест заданного типа и цвета
    3. Вычисляет гистограммы распределения цветов
    4. Возвращает оригинальное и обработанное изображение
    
    Поддерживаемые форматы: JPG, PNG, GIF, BMP
    Максимальный размер: 5MB
    """
    
    try:
        # Проверка размера файла
        file.file.seek(0, 2)  # Переходим в конец файла
        file_size = file.file.tell()
        file.file.seek(0)  # Возвращаемся в начало
        
        if file_size > settings.max_file_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Файл слишком большой. Максимальный размер: {settings.max_file_size // 1024 // 1024}MB"
            )
        
        # Проверка расширения файла
        filename = file.filename or "image"
        file_ext = Path(filename).suffix.lower()
        if file_ext not in settings.allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Неподдерживаемый формат файла. Поддерживаемые: {', '.join(settings.allowed_extensions)}"
            )
        
        # Валидация цвета HEX
        import re
        if not re.match(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$', color):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Неверный формат цвета. Используйте HEX формат: #RRGGBB или #RGB"
            )
        
        # Проверка капчи (в реальном приложении)
        # Для демо пропускаем проверку или используем простую
        if settings.debug:
            print(f"[DEBUG] Капча: {captcha}, ID: {captcha_id}")
            # В режиме отладки пропускаем проверку
        else:
            # В реальном приложении проверяем капчу
            if not captcha_id or not captcha_service.validate_captcha(captcha_id, captcha):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Неверная капча. Пожалуйста, введите правильный код."
                )
        
        # Обработка изображения
        result = image_service.process_image(
            image_file=file,
            cross_type=cross_type,
            color=color,
            width=width
        )
        
        # Добавляем информацию о запросе
        result.update({
            "cross_type": cross_type.value,
            "color": color,
            "width": width,
            "original_filename": filename,
            "timestamp": datetime.now().isoformat()
        })
        
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка обработки изображения: {str(e)}"
        )

@router.get(
    "/histogram/{image_id}",
    summary="Гистограмма изображения",
    description="Возвращает гистограмму распределения цветов для изображения",
    tags=["images"]
)
async def get_histogram(
    image_id: str,
    channel: Optional[str] = "all"
) -> Dict[str, Any]:
    """
    Возвращает гистограмму распределения цветов для изображения.
    
    Args:
        image_id: ID изображения
        channel: Канал цвета (red/green/blue/all)
        
    Returns:
        Гистограмма и статистика
    """
    try:
        # В реальном приложении получаем изображение из базы или хранилища
        # Здесь для демо создаем тестовую гистограмму
        
        # Тестовые данные
        import numpy as np
        np.random.seed(42)
        
        # Создаем тестовую гистограмму
        hist_r = np.random.normal(128, 50, 256).clip(0, 10000).astype(int).tolist()
        hist_g = np.random.normal(128, 40, 256).clip(0, 10000).astype(int).tolist()
        hist_b = np.random.normal(128, 60, 256).clip(0, 10000).astype(int).tolist()
        
        full_histogram = hist_r + hist_g + hist_b
        
        # Фильтруем по каналу если нужно
        if channel == "red":
            histogram = hist_r
        elif channel == "green":
            histogram = hist_g
        elif channel == "blue":
            histogram = hist_b
        else:
            histogram = full_histogram
        
        # Анализируем гистограмму
        from app.utils.image_utils import ImageUtils
        stats = ImageUtils().analyze_histogram(full_histogram)
        
        return {
            "success": True,
            "image_id": image_id,
            "channel": channel,
            "histogram": histogram,
            "stats": stats,
            "chart": ImageUtils().create_histogram_chart(full_histogram, f"Гистограмма ({channel})")
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка получения гистограммы: {str(e)}"
        )

@router.post(
    "/analyze",
    summary="Анализ изображения",
    description="Анализирует изображение без рисования креста",
    tags=["images"]
)
async def analyze_image(
    file: UploadFile = File(..., description="Изображение для анализа")
) -> Dict[str, Any]:
    """
    Анализирует загруженное изображение:
    1. Вычисляет гистограмму распределения цветов
    2. Анализирует доминирующие цвета
    3. Вычисляет различные метрики изображения
    """
    try:
        # Сохраняем временный файл
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Открываем изображение
            from PIL import Image
            image = Image.open(tmp_path)
            
            # Анализируем
            from app.utils.image_utils import ImageUtils
            utils = ImageUtils()
            
            # Гистограмма
            histogram = utils.calculate_histogram(image)
            hist_stats = utils.analyze_histogram(histogram)
            
            # Анализ цветов
            colors = image_service.analyze_colors(image)
            
            # Статистика изображения
            stats = image_service._get_image_info(image, file.filename or "image")
            
            # Создаем график
            chart = utils.create_histogram_chart(histogram, "Анализ изображения")
            
            result = {
                "success": True,
                "filename": file.filename,
                "histogram": histogram[:256],  # Только красный канал для краткости
                "histogram_stats": hist_stats,
                "color_analysis": colors,
                "image_stats": stats,
                "chart": chart,
                "is_grayscale": colors.get("is_grayscale", False),
                "colorfulness": colors.get("colorfulness_score", 0)
            }
            
            return result
            
        finally:
            # Удаляем временный файл
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка анализа изображения: {str(e)}"
        )

@router.get(
    "/compare",
    summary="Сравнение изображений",
    description="Сравнивает два изображения и вычисляет разницу",
    tags=["images"]
)
async def compare_images(
    url1: str,
    url2: str
) -> Dict[str, Any]:
    """
    Сравнивает два изображения по URL.
    
    Args:
        url1: URL первого изображения
        url2: URL второго изображения
        
    Returns:
        Результат сравнения с метриками
    """
    try:
        # В реальном приложении загружаем изображения по URL
        # Здесь для демо возвращаем тестовые данные
        
        import numpy as np
        
        return {
            "success": True,
            "url1": url1,
            "url2": url2,
            "comparison": {
                "similarity": 85.5,
                "psnr": 32.1,
                "ssim": 0.92,
                "changed_pixels": 1500,
                "change_percentage": 15.3,
                "mean_difference": 12.5
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка сравнения изображений: {str(e)}"
        )

@router.get(
    "/result",
    summary="Страница результата",
    description="Отображает страницу с результатами обработки",
    tags=["pages"]
)
async def result_page(request: Request):
    """
    Отображает HTML страницу с результатами обработки изображения.
    """
    return templates.TemplateResponse("result.html", {"request": request})

@router.get(
    "/download/{filename}",
    summary="Скачать изображение",
    description="Скачивает обработанное изображение",
    tags=["images"]
)
async def download_image(filename: str):
    """
    Скачивает изображение по имени файла.
    
    Args:
        filename: Имя файла для скачивания
    """
    file_path = Path(settings.upload_folder) / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Файл не найден"
        )
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="image/png"
    )

@router.get(
    "/health",
    summary="Проверка здоровья сервиса",
    description="Проверяет работоспособность сервиса обработки изображений",
    tags=["health"]
)
async def health_check() -> Dict[str, Any]:
    """
    Проверяет работоспособность сервиса обработки изображений.
    
    Returns:
        Статус сервиса и информация о версиях
    """
    import PIL
    import numpy as np
    
    return {
        "status": "healthy",
        "service": "image-processor",
        "version": "1.0.0",
        "dependencies": {
            "PIL": PIL.__version__,
            "numpy": np.__version__,
            "fastapi": "0.104.1"
        },
        "timestamp": datetime.now().isoformat(),
        "features": {
            "image_processing": True,
            "histogram_analysis": True,
            "color_analysis": True,
            "file_upload": True
        }
    }