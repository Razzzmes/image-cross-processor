from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Основные настройки
    app_name: str = "Image Cross Processor"
    debug: bool = True
    
    # Настройки файлов
    upload_folder: str = "app/static/uploads"
    temp_folder: str = "app/static/temp"
    max_file_size: int = 5 * 1024 * 1024  # 5MB
    allowed_extensions: set = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
    
    # Настройки капчи
    captcha_length: int = 5
    captcha_expire_seconds: int = 300  # 5 минут
    
    # Настройки изображений
    default_cross_width: int = 10
    colors: dict = {
        "red": "#FF0000",
        "green": "#00FF00",
        "blue": "#0000FF",
        "black": "#000000",
        "white": "#FFFFFF"
    }
    
    class Config:
        env_file = ".env"

settings = Settings()