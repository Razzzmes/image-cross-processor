import time
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import (
    APIRouter, 
    UploadFile, 
    File, 
    HTTPException, 
    status,
    Form
)

from app.models.schemas import ClassificationResponse, ClassificationResult
from app.config import settings

router = APIRouter()

# Загружаем модель при старте (если есть)
class MLModel:
    """Простая модель для классификации изображений"""
    
    def __init__(self):
        self.labels = [
            "самолет", "автомобиль", "птица", "кот", "олень",
            "собака", "лягушка", "лошадь", "корабль", "грузовик",
            "пейзаж", "портрет", "город", "природа", "животное",
            "еда", "техника", "люди", "архитектура", "искусство"
        ]
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Загружает модель (в реальном приложении загружаем настоящую модель)"""
        try:
            # В реальном приложении здесь загрузка TensorFlow/PyTorch модели
            # Для демо создаем "фейковую" модель
            print("[ML] Загружена модель классификации изображений")
            self.model_loaded = True
            return True
        except Exception as e:
            print(f"[ML] Ошибка загрузки модели: {e}")
            self.model_loaded = False
            return False
    
    def predict(self, image_data: bytes) -> List[Dict[str, Any]]:
        """Предсказывает класс изображения"""
        if not self.model_loaded:
            # Возвращаем случайные предсказания для демо
            np.random.seed(hash(image_data) % 10000)
            
            # Генерируем "предсказания"
            predictions = []
            selected_labels = np.random.choice(self.labels, size=5, replace=False)
            
            for label in selected_labels:
                prob = np.random.uniform(0.1, 0.9)
                predictions.append({
                    "label": label,
                    "probability": float(prob),
                    "confidence": "high" if prob > 0.7 else "medium" if prob > 0.4 else "low"
                })
            
            # Сортируем по вероятности
            predictions.sort(key=lambda x: x["probability"], reverse=True)
            
            # Нормализуем вероятности
            total = sum(p["probability"] for p in predictions)
            for p in predictions:
                p["probability"] = p["probability"] / total
            
            return predictions
        
        # В реальном приложении здесь будет вызов реальной модели
        # Например:
        # image = preprocess_image(image_data)
        # predictions = self.model.predict(image)
        # return process_predictions(predictions)
        
        return []

# Создаем экземпляр модели
ml_model = MLModel()

@router.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Классификация изображения",
    description="Классифицирует изображение с помощью нейросети",
    tags=["machine learning"]
)
async def classify_image(
    file: UploadFile = File(..., description="Изображение для классификации"),
    confidence_threshold: float = Form(0.5, description="Порог уверенности", ge=0.0, le=1.0)
) -> Dict[str, Any]:
    """
    Классифицирует загруженное изображение с помощью нейросети.
    
    Поддерживает классификацию по категориям:
    - Животные
    - Транспорт
    - Пейзажи
    - Люди
    - и другие
    
    Returns:
        Предсказания с вероятностями
    """
    start_time = time.time()
    
    try:
        # Проверяем размер файла
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > settings.max_file_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Файл слишком большой. Максимальный размер: {settings.max_file_size // 1024 // 1024}MB"
            )
        
        # Читаем данные файла
        image_data = await file.read()
        
        # Классифицируем изображение
        predictions = ml_model.predict(image_data)
        
        # Фильтруем по порогу уверенности
        filtered_predictions = [
            p for p in predictions 
            if p["probability"] >= confidence_threshold
        ]
        
        if not filtered_predictions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Не удалось классифицировать изображение с заданным порогом уверенности"
            )
        
        # Вычисляем время выполнения
        inference_time = time.time() - start_time
        
        # Формируем ответ
        return {
            "predictions": filtered_predictions,
            "model_name": "ImageClassifier v1.0",
            "model_version": "1.0.0",
            "inference_time": inference_time,
            "top_prediction": filtered_predictions[0] if filtered_predictions else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка классификации изображения: {str(e)}"
        )

@router.get(
    "/models",
    summary="Доступные модели",
    description="Возвращает список доступных моделей для классификации",
    tags=["machine learning"]
)
async def get_available_models() -> Dict[str, Any]:
    """
    Возвращает информацию о доступных моделях машинного обучения.
    
    Returns:
        Список моделей и их характеристики
    """
    return {
        "success": True,
        "models": [
            {
                "id": "classifier_v1",
                "name": "Image Classifier v1.0",
                "description": "Базовая модель для классификации изображений",
                "version": "1.0.0",
                "classes": len(ml_model.labels),
                "accuracy": 0.85,
                "size_mb": 45.2,
                "framework": "TensorFlow 2.15",
                "supported_formats": ["JPEG", "PNG", "BMP"]
            },
            {
                "id": "object_detector",
                "name": "Object Detector",
                "description": "Детекция объектов на изображениях",
                "version": "0.9.0",
                "classes": 80,
                "accuracy": 0.78,
                "size_mb": 120.5,
                "framework": "PyTorch 2.0",
                "supported_formats": ["JPEG", "PNG"]
            }
        ],
        "current_model": "classifier_v1" if ml_model.model_loaded else None,
        "status": "loaded" if ml_model.model_loaded else "not_loaded"
    }

@router.post(
    "/train",
    summary="Обучение модели",
    description="Запускает процесс обучения модели (для админов)",
    tags=["machine learning"]
)
async def train_model(
    epochs: int = Form(10, description="Количество эпох", ge=1, le=100),
    dataset: str = Form("cifar10", description="Название датасета")
) -> Dict[str, Any]:
    """
    Запускает процесс обучения модели (имитация).
    В реальном приложении это был бы долгий процесс.
    """
    try:
        # В реальном приложении здесь запуск обучения
        # Для демо просто имитируем
        
        return {
            "success": True,
            "message": "Обучение модели запущено",
            "job_id": f"train_{int(time.time())}",
            "parameters": {
                "epochs": epochs,
                "dataset": dataset,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "estimated_time": epochs * 30,  # 30 секунд на эпоху
            "start_time": time.time(),
            "status": "training"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка запуска обучения: {str(e)}"
        )

@router.get(
    "/labels",
    summary="Метки классов",
    description="Возвращает список меток классов для классификации",
    tags=["machine learning"]
)
async def get_class_labels() -> Dict[str, Any]:
    """
    Возвращает список всех меток классов, которые может распознавать модель.
    
    Returns:
        Список меток с описаниями
    """
    category_groups = {
        "animals": ["птица", "кот", "олень", "собака", "лягушка", "лошадь", "животное"],
        "vehicles": ["самолет", "автомобиль", "корабль", "грузовик", "техника"],
        "scenes": ["пейзаж", "город", "природа", "архитектура"],
        "people": ["портрет", "люди"],
        "other": ["еда", "искусство"]
    }
    
    return {
        "success": True,
        "total_labels": len(ml_model.labels),
        "labels": ml_model.labels,
        "categories": category_groups,
        "description": "Модель может классифицировать изображения по 20 основным категориям"
    }

@router.get(
    "/stats",
    summary="Статистика модели",
    description="Возвращает статистику использования модели",
    tags=["machine learning"]
)
async def get_model_stats() -> Dict[str, Any]:
    """
    Возвращает статистику использования модели.
    
    Returns:
        Статистика и метрики
    """
    # В реальном приложении собираем статистику из базы данных
    # Для демо возвращаем тестовые данные
    
    return {
        "success": True,
        "model": "ImageClassifier v1.0",
        "total_predictions": 1247,
        "accuracy": 0.872,
        "avg_inference_time": 0.145,
        "most_common_classes": [
            {"label": "пейзаж", "count": 245},
            {"label": "портрет", "count": 198},
            {"label": "животное", "count": 176}
        ],
        "last_trained": "2024-01-15T10:30:00Z",
        "status": "operational"
    }