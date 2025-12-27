import uuid
import base64
import io
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import random
import string

from captcha.image import ImageCaptcha
from app.config import settings

class CaptchaService:
    """Сервис для работы с капчей"""
    
    def __init__(self):
        self.captchas: Dict[str, dict] = {}  # Хранилище капч
        self.captcha_generator = ImageCaptcha(
            width=200,
            height=80,
            fonts=['app/static/fonts/Arial.ttf', 'app/static/fonts/Verdana.ttf']
        )
    
    def generate_captcha(self) -> Dict[str, any]:
        """
        Генерирует новую капчу
        
        Returns:
            Словарь с данными капчи
        """
        # Генерируем случайный текст
        text = ''.join(
            random.choices(
                string.ascii_uppercase + string.digits,
                k=settings.captcha_length
            )
        )
        
        # Генерируем изображение капчи
        image_data = self.captcha_generator.generate(text)
        
        # Конвертируем в base64
        buffered = io.BytesIO()
        image_data.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Создаем ID капчи
        captcha_id = str(uuid.uuid4())
        
        # Сохраняем капчу
        self.captchas[captcha_id] = {
            "text": text,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(seconds=settings.captcha_expire_seconds)
        }
        
        # Очищаем просроченные капчи
        self._cleanup_expired_captchas()
        
        return {
            "captcha_id": captcha_id,
            "text": text,  # В реальном приложении не отправляем на клиент!
            "image": f"data:image/png;base64,{img_base64}",
            "expires_at": self.captchas[captcha_id]["expires_at"].isoformat()
        }
    
    def validate_captcha(self, captcha_id: str, user_input: str) -> bool:
        """
        Проверяет капчу
        
        Args:
            captcha_id: ID капчи
            user_input: Введенный пользователем текст
            
        Returns:
            True если капча верна
        """
        # Проверяем существование капчи
        if captcha_id not in self.captchas:
            return False
        
        captcha_data = self.captchas[captcha_id]
        
        # Проверяем срок действия
        if datetime.now() > captcha_data["expires_at"]:
            # Удаляем просроченную капчу
            del self.captchas[captcha_id]
            return False
        
        # Сравниваем текст (без учета регистра)
        is_valid = captcha_data["text"].lower() == user_input.lower()
        
        # Удаляем капчу после проверки (одноразовая)
        if captcha_id in self.captchas:
            del self.captchas[captcha_id]
        
        return is_valid
    
    def _cleanup_expired_captchas(self):
        """Очищает просроченные капчи"""
        current_time = datetime.now()
        expired_ids = []
        
        for captcha_id, data in self.captchas.items():
            if current_time > data["expires_at"]:
                expired_ids.append(captcha_id)
        
        for captcha_id in expired_ids:
            del self.captchas[captcha_id]
    
    def get_captcha_info(self, captcha_id: str) -> Optional[dict]:
        """Получает информацию о капче"""
        return self.captchas.get(captcha_id)
    
    def get_active_captchas_count(self) -> int:
        """Возвращает количество активных капч"""
        return len(self.captchas)

# Создаем экземпляр сервиса
captcha_service = CaptchaService()