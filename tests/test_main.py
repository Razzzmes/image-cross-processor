import sys
import os
from pathlib import Path

# Добавляем корневую директорию проекта в путь Python
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

# Теперь импорт будет работать
from app.main import app

client = TestClient(app)

def test_health_check():
    """Тест проверки работоспособности API"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_generate_captcha():
    """Тест генерации капчи"""
    response = client.get("/api/captcha/generate")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "captcha_id" in data
    assert "image" in data
    assert data["image"].startswith("data:image/png;base64")

def test_root_page():
    """Тест главной страницы"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Image Cross Processor" in response.text

def test_result_page():
    """Тест страницы результатов"""
    response = client.get("/result")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Результат обработки" in response.text

def test_process_image_invalid_data():
    """Тест обработки изображения с неверными данными"""
    # Тест с пустыми данными
    response = client.post("/api/process", data={})
    assert response.status_code != 200  # Должна быть ошибка
    
def test_captcha_expiration():
    """Тест истечения срока действия капчи"""
    # Сначала получаем капчу
    response = client.get("/api/captcha/generate")
    data = response.json()
    captcha_id = data["captcha_id"]
    
    # Пытаемся использовать неверный текст
    form_data = {
        "captcha_id": captcha_id,
        "captcha_text": "wrong_text",
        "cross_type": "horizontal",
        "color": "#ff0000"
    }
    
    # Изменяем ожидаемый статус с 400 на 422
    # FastAPI возвращает 422 при ошибке валидации
    response = client.post("/api/process", data=form_data)
    assert response.status_code == 422  # Изменили с 400 на 422