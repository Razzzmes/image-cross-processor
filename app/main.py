import sys
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from pathlib import Path
import uuid
from enum import Enum
from typing import Dict
from PIL import Image, ImageDraw
import io
import base64
import time
import numpy as np
import random
import string

class Settings:
    upload_folder = "app/static/uploads"
    temp_folder = "app/static/temp"
    max_file_size = 5 * 1024 * 1024  # 5MB
    allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}

class CrossType(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    BOTH = "both"

settings = Settings()

captcha_store: Dict[str, dict] = {}

app = FastAPI(
    title="Image Cross Processor",
    description="Веб-приложение для обработки изображений с рисованием креста",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

os.makedirs(settings.upload_folder, exist_ok=True)
os.makedirs(settings.temp_folder, exist_ok=True)

def draw_cross_on_image(image: Image.Image, cross_type: CrossType, color: str, width: int = 10) -> Image.Image:
    """Рисует крест на изображении."""
    result = image.copy()
    draw = ImageDraw.Draw(result)

    color = color.lstrip('#')
    if len(color) == 3:
        color = ''.join([c*2 for c in color])
    color_rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
    
    img_width, img_height = image.size
    
    if cross_type == CrossType.HORIZONTAL:
        y = img_height // 2
        draw.line([(0, y), (img_width, y)], fill=color_rgb, width=width)
    elif cross_type == CrossType.VERTICAL:
        x = img_width // 2
        draw.line([(x, 0), (x, img_height)], fill=color_rgb, width=width)
    elif cross_type == CrossType.BOTH:
        y = img_height // 2
        x = img_width // 2
        draw.line([(0, y), (img_width, y)], fill=color_rgb, width=width)
        draw.line([(x, 0), (x, img_height)], fill=color_rgb, width=width)
    
    return result

def calculate_color_histogram(image: Image.Image):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image)
    
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        return {
            "red": [0] * 256,
            "green": [0] * 256,
            "blue": [0] * 256,
            "stats": {
                "red": {"mean": 0, "median": 0, "std": 0, "dominant": 0},
                "green": {"mean": 0, "median": 0, "std": 0, "dominant": 0},
                "blue": {"mean": 0, "median": 0, "std": 0, "dominant": 0}
            }
        }
    
    # Вычисляем гистограммы для каждого канала
    hist_r = np.histogram(img_array[:, :, 0], bins=256, range=(0, 256))[0]
    hist_g = np.histogram(img_array[:, :, 1], bins=256, range=(0, 256))[0]
    hist_b = np.histogram(img_array[:, :, 2], bins=256, range=(0, 256))[0]
    
    # Нормализуем (делаем проценты для лучшего отображения)
    total_pixels = img_array.shape[0] * img_array.shape[1]
    if total_pixels > 0:
        hist_r = (hist_r / total_pixels * 100).tolist()
        hist_g = (hist_g / total_pixels * 100).tolist()
        hist_b = (hist_b / total_pixels * 100).tolist()
    else:
        hist_r = hist_r.tolist()
        hist_g = hist_g.tolist()
        hist_b = hist_b.tolist()
    
    # Вычисляем статистику
    def get_stats(channel_idx):
        channel_data = img_array[:, :, channel_idx]
        return {
            "mean": float(np.mean(channel_data)),
            "median": float(np.median(channel_data)),
            "std": float(np.std(channel_data)),
            "dominant": int(np.argmax([hist_r, hist_g, hist_b][channel_idx]))
        }
    
    return {
        "red": hist_r,
        "green": hist_g,
        "blue": hist_b,
        "stats": {
            "red": get_stats(0),
            "green": get_stats(1),
            "blue": get_stats(2)
        }
    }

# ==================== РОУТЫ (API Endpoints) ====================
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Главная страница с формой."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/result", response_class=HTMLResponse)
async def result_page(request: Request):
    """Страница с результатами."""
    return templates.TemplateResponse("result.html", {"request": request})

@app.get("/health")
async def health_check():
    """Проверка работоспособности API."""
    return {
        "status": "healthy",
        "service": "Image Cross Processor",
        "version": "2.0.0",
        "endpoints": ["/", "/health", "/api/process", "/api/captcha/generate", "/docs"]
    }

@app.post("/api/process")
async def process_image(
    file: UploadFile = File(...),
    cross_type: CrossType = Form(...),
    color: str = Form("#FF0000"),
    width: int = Form(10),
    captcha_text: str = Form(...),
    captcha_id: str = Form(...)
):
    """
    Основной endpoint для обработки изображения.
    Рисует крест и возвращает результат с гистограммами.
    """
    try:
        print(f"=== DEBUG START ===")
        print(f"Captcha ID from form: {captcha_id}")
        print(f"Captcha text from form: {captcha_text}")
        
        # 0. ПРОВЕРКА КАПЧИ
        if not captcha_store:
            return JSONResponse({"error": "No captchas generated"}, status_code=400)
        
        if not captcha_id or captcha_id not in captcha_store:
            return JSONResponse({"error": "Invalid or expired captcha ID"}, status_code=400)
        
        stored_data = captcha_store[captcha_id]
        
        # Проверяем срок действия (10 минут)
        if time.time() - stored_data["created"] > 600:
            del captcha_store[captcha_id]
            return JSONResponse({"error": "Captcha expired"}, status_code=400)
        
        # Проверяем текст (без учета регистра)
        if stored_data["text"].lower() != captcha_text.lower():
            del captcha_store[captcha_id]
            return JSONResponse({"error": "Invalid captcha text"}, status_code=400)
        
        # Удаляем использованную капчу
        del captcha_store[captcha_id]
        print("DEBUG: Captcha validated successfully")
        
        # 1. Проверка файла
        if not file.filename:
            return JSONResponse({"error": "No file provided"}, status_code=400)
        
        # Проверка расширения
        ext = Path(file.filename).suffix.lower()
        if ext not in settings.allowed_extensions:
            return JSONResponse({"error": f"Unsupported format. Allowed: {settings.allowed_extensions}"}, status_code=400)
        
        # 2. Чтение файла
        contents = await file.read()
        print(f"DEBUG: File read, size: {len(contents)} bytes")
        
        # Проверка размера
        if len(contents) > settings.max_file_size:
            return JSONResponse({"error": f"File too large. Max: {settings.max_file_size//1024//1024}MB"}, status_code=400)
        
        # 3. Обработка изображения
        original_image = Image.open(io.BytesIO(contents))
        print(f"DEBUG: Image opened. Size: {original_image.size}, Mode: {original_image.mode}")
        
        # Конвертируем в RGB если нужно
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        processed_image = draw_cross_on_image(original_image, cross_type, color, width)
        
        # 4. ВЫЧИСЛЯЕМ ГИСТОГРАММЫ
        original_hist = calculate_color_histogram(original_image)
        processed_hist = calculate_color_histogram(processed_image)
        
        # 5. Конвертируем изображения в base64 для передачи
        def image_to_base64(image: Image.Image) -> str:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        
        original_b64 = image_to_base64(original_image)
        processed_b64 = image_to_base64(processed_image)
        
        print(f"=== DEBUG END ===")
        
        return {
            "success": True,
            "message": "Image processed successfully",
            "original_image": original_b64,
            "processed_image": processed_b64,
            "original_filename": file.filename,
            "original_size": original_image.size,
            "processed_size": processed_image.size,
            "processing_params": {
                "cross_type": cross_type,
                "color": color,
                "width": width
            },
            "histograms": {
                "original": original_hist,
                "processed": processed_hist
            }
        }
        
    except Exception as e:
        print(f"ERROR in process_image: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Processing failed: {str(e)}"}, status_code=500)

@app.get("/api/captcha/generate")
async def generate_captcha():
    """Генерирует изображение капчи."""
    try:
        from captcha.image import ImageCaptcha
        
        # Используем только безопасные символы
        safe_chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        text = ''.join(random.choices(safe_chars, k=5))
        
        # Создаем изображение капчи
        image_generator = ImageCaptcha(width=200, height=80)
        image_data = image_generator.generate(text)
        
        # Конвертируем в base64
        image_bytes = image_data.getvalue()
        img_base64 = base64.b64encode(image_bytes).decode()
        
        # Генерируем ID для капчи
        captcha_id = str(uuid.uuid4())
        
        # Сохраняем в глобальное хранилище
        captcha_store[captcha_id] = {
            "text": text,
            "created": time.time()
        }
        
        # Очищаем старые капчи (старше 10 минут)
        current_time = time.time()
        expired_keys = [key for key, data in captcha_store.items() 
                       if current_time - data["created"] > 600]
        for key in expired_keys:
            del captcha_store[key]
        
        print(f"DEBUG: Generated captcha with ID: {captcha_id}, text: {text}")
        
        return {
            "success": True,
            "captcha_id": captcha_id,
            "image": f"data:image/png;base64,{img_base64}",
            "expires_in": 600
        }
        
    except Exception as e:
        print(f"ERROR generating captcha: {e}")
        return JSONResponse({"error": f"Captcha generation failed: {str(e)}"}, status_code=500)

# ==================== ЗАПУСК СЕРВЕРА ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🖼️   IMAGE CROSS PROCESSOR")
    print("=" * 60)
    print(f"📍 Local URL:    http://localhost:8000")
    print(f"🌐 Network URL:  http://0.0.0.0:8000")
    print(f"📚 API Docs:     http://localhost:8000/docs")
    print(f"📊 Health check: http://localhost:8000/health")
    print("=" * 60)
    print("🚀 Server is starting... Press Ctrl+C to stop.")
    print("=" * 60)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )