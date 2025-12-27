from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import re

from app.models.enums import CrossType, ImageFormat

# Базовые модели
class CaptchaBase(BaseModel):
    """Базовая модель капчи"""
    text: str = Field(..., min_length=4, max_length=6, description="Текст капчи")
    image: str = Field(..., description="Изображение капчи в base64")

class ImageProcessingBase(BaseModel):
    """Базовая модель для обработки изображений"""
    cross_type: CrossType = Field(..., description="Тип креста")
    color: str = Field(..., pattern="^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$", description="Цвет в HEX формате")
    width: int = Field(10, ge=1, le=50, description="Ширина креста (1-50px)")

    @validator('color')
    def validate_hex_color(cls, v):
        """Валидация HEX цвета"""
        if len(v) == 4:  # Сокращенный формат #RGB
            v = '#' + v[1] * 2 + v[2] * 2 + v[3] * 2
        return v.upper()

class HistogramData(BaseModel):
    """Данные гистограммы"""
    channel: str
    data: List[int]
    mean: float
    median: int
    std: float
    dominant_value: int
    dominant_count: int

class ImageStats(BaseModel):
    """Статистика изображения"""
    width: int
    height: int
    format: ImageFormat
    mode: str
    size_bytes: int
    color_count: Optional[int]
    brightness: Optional[float]
    contrast: Optional[float]

# Модели запросов
class ProcessImageRequest(ImageProcessingBase):
    """Запрос на обработку изображения"""
    captcha: str = Field(..., min_length=4, max_length=6, description="Введенная капча")
    
    class Config:
        json_schema_extra = {
            "example": {
                "cross_type": "horizontal",
                "color": "#FF0000",
                "width": 10,
                "captcha": "AB12CD"
            }
        }

class CaptchaRequest(BaseModel):
    """Запрос для проверки капчи"""
    captcha_id: Optional[str] = None
    captcha_text: str = Field(..., min_length=4, max_length=6)

class ClassifyImageRequest(BaseModel):
    """Запрос на классификацию изображения"""
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)

# Модели ответов
class CaptchaResponse(CaptchaBase):
    """Ответ с капчей"""
    captcha_id: str
    expires_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class HistogramResponse(BaseModel):
    """Ответ с гистограммой"""
    original: List[int]
    processed: List[int]
    original_stats: Dict[str, Any]
    processed_stats: Dict[str, Any]
    charts: Dict[str, str]  # base64 изображения графиков

class ClassificationResult(BaseModel):
    """Результат классификации"""
    label: str
    probability: float
    confidence: str  # low/medium/high

class ClassificationResponse(BaseModel):
    """Ответ классификации"""
    predictions: List[ClassificationResult]
    model_name: str
    model_version: str
    inference_time: float
    top_prediction: ClassificationResult

class ProcessImageResponse(BaseModel):
    """Полный ответ обработки изображения"""
    success: bool = True
    message: str = "Изображение успешно обработано"
    
    # Изображения
    original_image: str  # base64 или URL
    processed_image: str
    original_filename: str
    processed_filename: str
    
    # Информация
    original_stats: ImageStats
    processed_stats: ImageStats
    processing_params: ImageProcessingBase
    
    # Гистограммы
    histograms: HistogramResponse
    
    # Классификация (опционально)
    classification: Optional[ClassificationResponse] = None
    
    # Метаданные
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ErrorResponse(BaseModel):
    """Модель ошибки"""
    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Модели для фронтенда
class ImageComparison(BaseModel):
    """Сравнение изображений для фронтенда"""
    original_url: str
    processed_url: str
    difference_percentage: float
    changed_pixels: int

class ColorAnalysis(BaseModel):
    """Анализ цветов"""
    dominant_colors: List[Dict[str, Any]]
    color_palette: List[str]
    colorfulness_score: float
    is_grayscale: bool