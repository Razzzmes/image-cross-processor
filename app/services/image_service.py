import time
import io
import base64
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path
import numpy as np
import math

from PIL import Image, ImageStat
from fastapi import UploadFile

from app.models.enums import CrossType
from app.utils.image_utils import ImageUtils
from app.utils.file_utils import FileUtils
from app.config import settings

class ImageService:
    """Сервис для обработки изображений"""
    
    def __init__(self):
        self.image_utils = ImageUtils()
        self.file_utils = FileUtils()
    
    def process_image(
        self,
        image_file: UploadFile,
        cross_type: CrossType,
        color: str,
        width: int = 10
    ) -> Dict[str, Any]:
        """
        Обрабатывает изображение - рисует крест
        
        Args:
            image_file: Загруженный файл изображения
            cross_type: Тип креста
            color: Цвет креста
            width: Ширина креста
            
        Returns:
            Словарь с результатами обработки
        """
        start_time = time.time()
        
        try:
            # Сохраняем загруженный файл
            file_path, filename = self.file_utils.save_upload_file(image_file)
            if not file_path:
                raise ValueError("Не удалось сохранить файл")
            
            # Открываем изображение
            original_image = self.image_utils.open_image(file_path)
            if not original_image:
                raise ValueError("Не удалось открыть изображение")
            
            # Получаем статистику оригинального изображения
            original_stats = self._get_image_info(original_image, filename)
            
            # Рисуем крест
            processed_image = self.image_utils.draw_cross(
                image=original_image,
                cross_type=cross_type,
                color=color,
                width=width
            )
            
            # Получаем статистику обработанного изображения
            processed_filename = f"processed_{filename}"
            processed_stats = self._get_image_info(processed_image, processed_filename)
            
            # Вычисляем гистограммы
            original_histogram = self.image_utils.calculate_histogram(original_image)
            processed_histogram = self.image_utils.calculate_histogram(processed_image)
            
            # Анализируем гистограммы
            original_hist_stats = self.image_utils.analyze_histogram(original_histogram)
            processed_hist_stats = self.image_utils.analyze_histogram(processed_histogram)
            
            # Создаем графики гистограмм
            original_chart = self.image_utils.create_histogram_chart(
                original_histogram, 
                "Исходное изображение"
            )
            processed_chart = self.image_utils.create_histogram_chart(
                processed_histogram,
                "Обработанное изображение"
            )
            
            # Анализируем цвета
            original_colors = self.analyze_colors(original_image)
            processed_colors = self.analyze_colors(processed_image)
            
            # Сравниваем изображения
            comparison = self.compare_images(original_image, processed_image)
            
            # Конвертируем изображения в base64
            original_base64 = self.image_utils.image_to_base64(original_image)
            processed_base64 = self.image_utils.image_to_base64(processed_image)
            
            # Вычисляем время обработки
            processing_time = time.time() - start_time
            
            # Формируем ответ
            result = {
                "success": True,
                "message": "Изображение успешно обработано",
                
                # Изображения
                "original_image": original_base64,
                "processed_image": processed_base64,
                "original_filename": filename,
                "processed_filename": processed_filename,
                
                # Статистика
                "original_stats": original_stats,
                "processed_stats": processed_stats,
                
                # Параметры обработки
                "processing_params": {
                    "cross_type": cross_type.value,
                    "color": color,
                    "width": width
                },
                
                # Гистограммы
                "histograms": {
                    "original": original_histogram,
                    "processed": processed_histogram,
                    "original_stats": original_hist_stats,
                    "processed_stats": processed_hist_stats,
                    "charts": {
                        "original": original_chart or "",
                        "processed": processed_chart or ""
                    }
                },
                
                # Анализ цветов
                "color_analysis": {
                    "original": original_colors,
                    "processed": processed_colors
                },
                
                # Сравнение
                "comparison": comparison,
                
                # Метаданные
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                
                # Дополнительная информация
                "file_info": {
                    "original_path": file_path,
                    "file_size": Path(file_path).stat().st_size,
                    "mime_type": image_file.content_type
                }
            }
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            raise Exception(f"Ошибка обработки изображения: {str(e)}") from e
    
    def analyze_colors(self, image: Image.Image) -> Dict[str, Any]:
        """Анализирует цвета изображения"""
        # Извлекаем доминирующие цвета
        dominant_colors = self.image_utils.extract_dominant_colors(image, num_colors=8)
        
        # Вычисляем показатель "красочности"
        colorfulness = self._calculate_colorfulness(image)
        
        # Проверяем, является ли изображение grayscale
        is_grayscale = image.mode in ['L', 'LA', 'P'] or self._is_grayscale(image)
        
        # Создаем палитру
        palette = [color["hex"] for color in dominant_colors[:5]]
        
        # Анализируем тональность
        tone_analysis = self._analyze_image_tone(image)
        
        return {
            "dominant_colors": dominant_colors,
            "color_palette": palette,
            "colorfulness_score": colorfulness,
            "is_grayscale": is_grayscale,
            "tone_analysis": tone_analysis,
            "vibrance": self._calculate_vibrance(image),
            "saturation": self._calculate_saturation(image)
        }
    
    def compare_images(
        self, 
        image1: Image.Image, 
        image2: Image.Image
    ) -> Dict[str, Any]:
        """Сравнивает два изображения"""
        # Приводим к одинаковому размеру
        if image1.size != image2.size:
            image2 = image2.resize(image1.size)
        
        # Конвертируем в RGB если нужно
        if image1.mode != 'RGB':
            image1 = image1.convert('RGB')
        if image2.mode != 'RGB':
            image2 = image2.convert('RGB')
        
        # Преобразуем в массивы numpy
        arr1 = np.array(image1).astype(np.float32)
        arr2 = np.array(image2).astype(np.float32)
        
        # Вычисляем разницу
        diff = np.abs(arr1 - arr2)
        
        # Процент измененных пикселей (пиксель считается измененным если разница > 10 в любом канале)
        diff_threshold = 10
        changed_pixels_mask = np.any(diff > diff_threshold, axis=2)
        changed_pixels = np.sum(changed_pixels_mask)
        total_pixels = image1.width * image1.height
        change_percentage = (changed_pixels / total_pixels) * 100
        
        # Средняя разница по каналам
        mean_difference_rgb = np.mean(diff, axis=(0, 1))
        mean_difference_total = np.mean(diff)
        
        # Создаем визуализацию разницы
        diff_visualization = self._create_diff_visualization(image1, image2)
        
        # Вычисляем PSNR (Peak Signal-to-Noise Ratio)
        psnr = self._calculate_psnr(arr1, arr2)
        
        # Вычисляем SSIM (Structural Similarity Index)
        ssim = self._calculate_ssim(arr1, arr2)
        
        return {
            "changed_pixels": int(changed_pixels),
            "change_percentage": float(change_percentage),
            "mean_difference": {
                "r": float(mean_difference_rgb[0]),
                "g": float(mean_difference_rgb[1]),
                "b": float(mean_difference_rgb[2]),
                "total": float(mean_difference_total)
            },
            "psnr": float(psnr) if psnr else None,
            "ssim": float(ssim) if ssim else None,
            "diff_visualization": diff_visualization,
            "is_similar": change_percentage < 5.0 and ssim > 0.95 if ssim else change_percentage < 5.0
        }
    
    def _get_image_info(self, image: Image.Image, filename: str) -> Dict[str, Any]:
        """Получает информацию об изображении"""
        stats = ImageStat.Stat(image)
        
        info = {
            "filename": filename,
            "width": image.width,
            "height": image.height,
            "format": image.format or "Unknown",
            "mode": image.mode,
            "size_bytes": len(image.tobytes()),
            "aspect_ratio": round(image.width / image.height, 2) if image.height > 0 else 0,
            "megapixels": round((image.width * image.height) / 1_000_000, 2),
            "dpi": image.info.get('dpi', (72, 72)),
            "has_alpha": 'A' in image.mode,
            "is_animated": getattr(image, 'is_animated', False),
            "n_frames": getattr(image, 'n_frames', 1)
        }
        
        # Добавляем статистику цвета если изображение в RGB
        if image.mode == 'RGB':
            info.update({
                "mean_r": stats.mean[0] if len(stats.mean) > 0 else 0,
                "mean_g": stats.mean[1] if len(stats.mean) > 1 else 0,
                "mean_b": stats.mean[2] if len(stats.mean) > 2 else 0,
                "brightness": sum(stats.mean) / 3 if stats.mean else 0,
                "std_r": stats.stddev[0] if hasattr(stats, 'stddev') and len(stats.stddev) > 0 else 0,
                "std_g": stats.stddev[1] if hasattr(stats, 'stddev') and len(stats.stddev) > 1 else 0,
                "std_b": stats.stddev[2] if hasattr(stats, 'stddev') and len(stats.stddev) > 2 else 0,
                "contrast": sum(stats.stddev) / 3 if hasattr(stats, 'stddev') and stats.stddev else 0
            })
        
        # Подсчет уникальных цветов (только для небольших изображений)
        if image.width * image.height < 500000:  # 0.5MP
            try:
                colors = image.getcolors(maxcolors=65536)
                if colors:
                    info["unique_colors"] = len(colors)
                    # Находим самый частый цвет
                    most_common = max(colors, key=lambda x: x[0])
                    info["most_common_color"] = {
                        "count": most_common[0],
                        "percentage": (most_common[0] / (image.width * image.height)) * 100,
                        "rgb": most_common[1] if isinstance(most_common[1], tuple) else None
                    }
            except Exception:
                info["unique_colors"] = "N/A"
        else:
            info["unique_colors"] = "Too large to calculate"
        
        return info
    
    def _calculate_colorfulness(self, image: Image.Image) -> float:
        """Вычисляет показатель 'красочности' изображения (Hasler & Susstrunk, 2003)"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Преобразуем в numpy array
            arr = np.array(image).astype(np.float32) / 255.0
            
            # Разделяем на каналы
            r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
            
            # Вычисляем разности
            rg = r - g
            yb = 0.5 * (r + g) - b
            
            # Вычисляем стандартные отклонения и средние значения
            std_rg = np.std(rg)
            std_yb = np.std(yb)
            mean_rg = np.mean(rg)
            mean_yb = np.mean(yb)
            
            # Вычисляем цветность по формуле Hasler & Susstrunk
            std_root = np.sqrt(std_rg ** 2 + std_yb ** 2)
            mean_root = np.sqrt(mean_rg ** 2 + mean_yb ** 2)
            colorfulness = std_root + 0.3 * mean_root
            
            # Нормализуем до 0-100
            colorfulness_normalized = min(100, colorfulness * 100)
            
            return round(float(colorfulness_normalized), 2)
            
        except Exception as e:
            print(f"Error calculating colorfulness: {e}")
            return 0.0
    
    def _calculate_vibrance(self, image: Image.Image) -> float:
        """Вычисляет вибрантность (насыщенность цветов)"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            arr = np.array(image).astype(np.float32)
            
            # Преобразуем в HSV
            from colorsys import rgb_to_hsv
            
            # Вычисляем насыщенность для каждого пикселя
            saturation_values = []
            for row in arr:
                for pixel in row:
                    r, g, b = pixel / 255.0
                    _, s, _ = rgb_to_hsv(r, g, b)
                    saturation_values.append(s)
            
            vibrance = np.mean(saturation_values) * 100
            return round(float(vibrance), 2)
            
        except Exception:
            return 0.0
    
    def _calculate_saturation(self, image: Image.Image) -> float:
        """Вычисляет общую насыщенность"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Простой подход: разница между максимальным и минимальным каналом
            arr = np.array(image).astype(np.float32)
            
            max_channel = np.max(arr, axis=2)
            min_channel = np.min(arr, axis=2)
            
            # Избегаем деления на ноль
            mask = max_channel > 0
            saturation = np.zeros_like(max_channel)
            saturation[mask] = 1 - (min_channel[mask] / max_channel[mask])
            
            avg_saturation = np.mean(saturation) * 100
            return round(float(avg_saturation), 2)
            
        except Exception:
            return 0.0
    
    def _is_grayscale(self, image: Image.Image) -> bool:
        """Проверяет, является ли изображение оттенками серого"""
        if image.mode in ['L', 'LA', 'P']:
            return True
        
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            arr = np.array(image)
            # Проверяем, одинаковы ли значения R, G, B для всех пикселей
            diff = np.max(np.abs(arr[:,:,0] - arr[:,:,1])) + np.max(np.abs(arr[:,:,1] - arr[:,:,2]))
            return diff < 10  # Порог
            
        except Exception:
            return False
    
    def _analyze_image_tone(self, image: Image.Image) -> Dict[str, float]:
        """Анализирует тональность изображения (теплые/холодные тона)"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            arr = np.array(image).astype(np.float32)
            
            # Разделяем каналы
            r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
            
            # Вычисляем теплоту (отношение красного к синему)
            total_red = np.sum(r)
            total_blue = np.sum(b)
            total_green = np.sum(g)
            
            if total_blue > 0:
                warmth_ratio = total_red / total_blue
            else:
                warmth_ratio = total_red
            
            # Нормализуем до 0-100
            warmth_score = min(100, warmth_ratio / 2 * 100)
            
            # Определяем преобладающую температуру
            if warmth_score > 60:
                temperature = "warm"
            elif warmth_score < 40:
                temperature = "cool"
            else:
                temperature = "neutral"
            
            # Вычисляем баланс
            total_all = total_red + total_green + total_blue
            if total_all > 0:
                red_percent = (total_red / total_all) * 100
                green_percent = (total_green / total_all) * 100
                blue_percent = (total_blue / total_all) * 100
            else:
                red_percent = green_percent = blue_percent = 0
            
            return {
                "warmth_score": round(float(warmth_score), 2),
                "temperature": temperature,
                "red_percent": round(float(red_percent), 2),
                "green_percent": round(float(green_percent), 2),
                "blue_percent": round(float(blue_percent), 2),
                "is_balanced": abs(red_percent - green_percent) < 10 and abs(green_percent - blue_percent) < 10
            }
            
        except Exception as e:
            print(f"Error analyzing tone: {e}")
            return {
                "warmth_score": 50.0,
                "temperature": "neutral",
                "red_percent": 33.33,
                "green_percent": 33.33,
                "blue_percent": 33.33,
                "is_balanced": True
            }
    
    def _create_diff_visualization(self, img1: Image.Image, img2: Image.Image) -> str:
        """Создает визуализацию разницы между изображениями"""
        try:
            # Приводим к одинаковому размеру
            if img1.size != img2.size:
                img2 = img2.resize(img1.size)
            
            # Конвертируем в RGB
            if img1.mode != 'RGB':
                img1 = img1.convert('RGB')
            if img2.mode != 'RGB':
                img2 = img2.convert('RGB')
            
            # Преобразуем в массивы
            arr1 = np.array(img1).astype(np.float32)
            arr2 = np.array(img2).astype(np.float32)
            
            # Вычисляем разницу
            diff = np.abs(arr1 - arr2)
            
            # Нормализуем для лучшей видимости
            diff_normalized = (diff / 255.0 * 255).astype(np.uint8)
            
            # Создаем изображение разницы
            diff_img = Image.fromarray(diff_normalized)
            
            # Конвертируем в base64
            return self.image_utils.image_to_base64(diff_img)
            
        except Exception:
            return ""
    
    def _calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> Optional[float]:
        """Вычисляет PSNR (Peak Signal-to-Noise Ratio)"""
        try:
            # Вычисляем MSE (Mean Squared Error)
            mse = np.mean((img1 - img2) ** 2)
            
            # Если MSE равно 0, изображения идентичны
            if mse == 0:
                return float('inf')
            
            # Максимально возможное значение пикселя
            max_pixel = 255.0
            
            # Вычисляем PSNR
            psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
            
            return float(psnr)
            
        except Exception:
            return None
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray, window_size: int = 11) -> Optional[float]:
        """Вычисляет SSIM (Structural Similarity Index)"""
        try:
            # Простая реализация SSIM (упрощенная)
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            
            # Вычисляем средние значения
            mu1 = np.mean(img1)
            mu2 = np.mean(img2)
            
            # Вычисляем дисперсии
            sigma1_sq = np.var(img1)
            sigma2_sq = np.var(img2)
            sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]
            
            # Вычисляем SSIM
            numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
            denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
            
            ssim = numerator / denominator
            
            return float(ssim)
            
        except Exception:
            return None
    
    def resize_image(self, image: Image.Image, max_width: int = 800, max_height: int = 600) -> Image.Image:
        """Изменяет размер изображения с сохранением пропорций"""
        original_width, original_height = image.size
        
        # Вычисляем новые размеры
        ratio = min(max_width / original_width, max_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        if ratio < 1:
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def create_thumbnail(self, image: Image.Image, size: Tuple[int, int] = (200, 200)) -> Image.Image:
        """Создает миниатюру изображения"""
        return image.copy().thumbnail(size, Image.Resampling.LANCZOS)

# Создаем экземпляр сервиса
image_service = ImageService()