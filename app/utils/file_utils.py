import os
import uuid
import shutil
from pathlib import Path
from typing import Optional, Tuple
from fastapi import UploadFile
# ИСПРАВЛЕНИЕ: Удален import imghdr. Вместо него используется PIL.
from PIL import Image, ImageFile

from app.config import settings

# Разрешаем Pillow загружать большие изображения
ImageFile.LOAD_TRUNCATED_IMAGES = True

class FileUtils:
    """Утилиты для работы с файлами"""
    
    @staticmethod
    def save_upload_file(upload_file: UploadFile, subfolder: str = "") -> Tuple[Optional[str], Optional[str]]:
        """
        Сохраняет загруженный файл
        
        Args:
            upload_file: Загруженный файл
            subfolder: Подпапка для сохранения
            
        Returns:
            Tuple[путь к файлу, имя файла] или (None, None) при ошибке
        """
        try:
            # Проверка расширения
            ext = Path(upload_file.filename).suffix.lower()
            if ext not in settings.allowed_extensions:
                return None, None
            
            # Создаем уникальное имя файла
            file_id = str(uuid.uuid4())
            filename = f"{file_id}{ext}"
            
            # Создаем папку если не существует
            save_dir = Path(settings.upload_folder) / subfolder
            save_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = save_dir / filename
            
            # Сохраняем файл
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(upload_file.file, buffer)
            
            return str(file_path), filename
            
        except Exception as e:
            print(f"Error saving file: {e}")
            return None, None
    
    @staticmethod
    def validate_image_file(file_path: str) -> bool:
        """Проверяет, что файл является валидным изображением (исправленная версия)"""
        try:
            # 1. Проверяем размер файла
            if os.path.getsize(file_path) > settings.max_file_size:
                return False
            
            # 2. ИСПРАВЛЕНИЕ: Вместо imghdr используем Pillow для проверки
            # Пытаемся открыть файл как изображение
            with Image.open(file_path) as img:
                # Быстрая проверка формата
                img.verify()  # Эта операция только проверяет целостность файла
                
                # Дополнительно проверяем, что формат поддерживается
                if img.format not in ['JPEG', 'PNG', 'GIF', 'BMP', 'WEBP']:
                    return False
                    
            return True
            
        except Exception as e:
            # Если возникла любая ошибка при открытии/проверке - файл невалидный
            print(f"Image validation error: {e}")
            return False
    
    @staticmethod
    def delete_file(file_path: str) -> bool:
        """Удаляет файл"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception:
            return False
    
    @staticmethod
    def cleanup_old_files(directory: str, max_age_hours: int = 24):
        """Удаляет старые файлы из директории"""
        import time
        from pathlib import Path
        
        dir_path = Path(directory)
        if not dir_path.exists():
            return
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in dir_path.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                    except Exception:
                        pass
    
    @staticmethod
    def get_file_info(file_path: str) -> Optional[dict]:
        """Получает информацию о файле"""
        try:
            stat = os.stat(file_path)
            return {
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "accessed": stat.st_atime
            }
        except Exception:
            return None
    
    @staticmethod
    def create_temp_file(content: bytes, extension: str = ".tmp") -> Optional[str]:
        """Создает временный файл"""
        try:
            temp_dir = Path(settings.temp_folder)
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            file_id = str(uuid.uuid4())
            filename = f"temp_{file_id}{extension}"
            file_path = temp_dir / filename
            
            with open(file_path, "wb") as f:
                f.write(content)
            
            return str(file_path)
        except Exception:
            return None
