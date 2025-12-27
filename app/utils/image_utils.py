import io
import base64
from PIL import Image, ImageDraw, ImageFilter, ImageStat, ImageOps
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import colorsys
from pathlib import Path

from app.models.enums import CrossType
from app.config import settings

class ImageUtils:
    """Утилиты для работы с изображениями"""
    
    @staticmethod
    def open_image(file_path: str) -> Optional[Image.Image]:
        """Открывает изображение"""
        try:
            return Image.open(file_path)
        except Exception as e:
            print(f"Error opening image: {e}")
            return None
    
    @staticmethod
    def draw_cross(
        image: Image.Image,
        cross_type: CrossType,
        color: str,
        width: int = 10
    ) -> Image.Image:
        """
        Рисует крест на изображении
        
        Args:
            image: Изображение PIL
            cross_type: Тип креста
            color: Цвет в HEX формате
            width: Ширина линии
            
        Returns:
            Изображение с крестом
        """
        # Создаем копию изображения
        result = image.copy()
        draw = ImageDraw.Draw(result)
        
        # Конвертируем HEX в RGB
        color_rgb = ImageUtils.hex_to_rgb(color)
        
        # Получаем размеры изображения
        img_width, img_height = image.size
        
        if cross_type == CrossType.HORIZONTAL:
            # Горизонтальная линия посередине
            y = img_height // 2
            draw.line([(0, y), (img_width, y)], fill=color_rgb, width=width)
            
        elif cross_type == CrossType.VERTICAL:
            # Вертикальная линия посередине
            x = img_width // 2
            draw.line([(x, 0), (x, img_height)], fill=color_rgb, width=width)
            
        elif cross_type == CrossType.BOTH:
            # И горизонтальная, и вертикальная
            y = img_height // 2
            x = img_width // 2
            
            # Горизонтальная
            draw.line([(0, y), (img_width, y)], fill=color_rgb, width=width)
            # Вертикальная
            draw.line([(x, 0), (x, img_height)], fill=color_rgb, width=width)
        
        return result
    
    @staticmethod
    def calculate_histogram(image: Image.Image) -> List[int]:
        """
        Вычисляет гистограмму изображения
        
        Returns:
            Список из 768 значений (256 для каждого канала RGB)
        """
        # Конвертируем в RGB если нужно
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Получаем данные пикселей
        pixels = list(image.getdata())
        
        # Инициализируем гистограммы
        hist_r = [0] * 256
        hist_g = [0] * 256
        hist_b = [0] * 256
        
        # Заполняем гистограммы
        for r, g, b in pixels:
            hist_r[r] += 1
            hist_g[g] += 1
            hist_b[b] += 1
        
        # Объединяем в один список
        return hist_r + hist_g + hist_b
    
    @staticmethod
    def analyze_histogram(histogram: List[int]) -> Dict[str, Any]:
        """Анализирует гистограмму и возвращает статистику"""
        # Разделяем по каналам
        hist_r = histogram[:256]
        hist_g = histogram[256:512]
        hist_b = histogram[512:768]
        
        def channel_stats(channel_hist):
            """Статистика для одного канала"""
            channel_array = np.array(channel_hist)
            total_pixels = np.sum(channel_array)
            
            if total_pixels == 0:
                return {
                    "mean": 0,
                    "median": 0,
                    "std": 0,
                    "dominant": 0,
                    "dominant_count": 0
                }
            
            # Среднее значение (взвешенное)
            indices = np.arange(256)
            mean = np.average(indices, weights=channel_array)
            
            # Медиана
            cumulative = np.cumsum(channel_array)
            median_idx = np.searchsorted(cumulative, total_pixels / 2)
            
            # Стандартное отклонение
            std = np.sqrt(np.average((indices - mean) ** 2, weights=channel_array))
            
            # Доминирующее значение
            dominant_idx = np.argmax(channel_array)
            dominant_count = channel_array[dominant_idx]
            
            return {
                "mean": float(mean),
                "median": int(median_idx),
                "std": float(std),
                "dominant": int(dominant_idx),
                "dominant_count": int(dominant_count)
            }
        
        stats_r = channel_stats(hist_r)
        stats_g = channel_stats(hist_g)
        stats_b = channel_stats(hist_b)
        
        # Находим доминирующий RGB цвет
        dominant_r = stats_r["dominant"]
        dominant_g = stats_g["dominant"]
        dominant_b = stats_b["dominant"]
        
        return {
            "r": stats_r,
            "g": stats_g,
            "b": stats_b,
            "dominant_rgb": (dominant_r, dominant_g, dominant_b),
            "dominant_hex": f"#{dominant_r:02x}{dominant_g:02x}{dominant_b:02x}",
            "brightness": (stats_r["mean"] * 0.299 + 
                          stats_g["mean"] * 0.587 + 
                          stats_b["mean"] * 0.114) / 255,
            "contrast": (stats_r["std"] + stats_g["std"] + stats_b["std"]) / 3 / 255
        }
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Конвертирует HEX цвет в RGB"""
        hex_color = hex_color.lstrip('#')
        
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @staticmethod
    def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        """Конвертирует RGB цвет в HEX"""
        return '#{:02x}{:02x}{:02x}'.format(*rgb)
    
    @staticmethod
    def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
        """Конвертирует PIL Image в base64 строку"""
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{img_str}"
    
    @staticmethod
    def base64_to_image(base64_str: str) -> Optional[Image.Image]:
        """Конвертирует base64 строку в PIL Image"""
        try:
            # Удаляем префикс data URL если есть
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            
            img_data = base64.b64decode(base64_str)
            return Image.open(io.BytesIO(img_data))
        except Exception:
            return None
    
    @staticmethod
    def get_image_stats(image: Image.Image) -> Dict[str, Any]:
        """Получает статистику изображения"""
        stats = ImageStat.Stat(image)
        
        result = {
            "width": image.width,
            "height": image.height,
            "format": image.format or "Unknown",
            "mode": image.mode,
            "size_bytes": len(image.tobytes()),
            "color_count": None,
            "brightness": None,
            "contrast": None
        }
        
        # Вычисляем яркость и контраст если возможно
        if image.mode == 'RGB':
            # Яркость (среднее по всем каналам)
            result["brightness"] = sum(stats.mean) / 3 / 255
            
            # Контраст (стандартное отклонение)
            if hasattr(stats, 'stddev'):
                result["contrast"] = sum(stats.stddev) / 3 / 255
        
        # Подсчет уникальных цветов (только для небольших изображений)
        if image.width * image.height < 1000000:  # 1MP
            try:
                colors = image.getcolors(maxcolors=1000000)
                if colors:
                    result["color_count"] = len(colors)
            except Exception:
                pass
        
        return result
    
    @staticmethod
    def extract_dominant_colors(image: Image.Image, num_colors: int = 5) -> List[Dict[str, Any]]:
        """Извлекает доминирующие цвета из изображения"""
        # Уменьшаем размер для ускорения обработки
        small_image = image.copy()
        small_image.thumbnail((100, 100))
        
        # Конвертируем в RGB
        if small_image.mode != 'RGB':
            small_image = small_image.convert('RGB')
        
        # Получаем цвета
        colors = small_image.getcolors(maxcolors=10000)
        if not colors:
            return []
        
        # Сортируем по частоте
        colors.sort(key=lambda x: x[0], reverse=True)
        
        result = []
        for count, color in colors[:num_colors]:
            r, g, b = color
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            
            result.append({
                "rgb": color,
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "count": count,
                "percentage": count / (small_image.width * small_image.height) * 100,
                "hsv": (h, s, v)
            })
        
        return result
    
    @staticmethod
    def create_histogram_chart(histogram: List[int], title: str = "Histogram") -> Optional[str]:
        """Создает изображение гистограммы с помощью matplotlib"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Для работы без GUI
            
            # Разделяем гистограмму по каналам
            hist_r = histogram[:256]
            hist_g = histogram[256:512]
            hist_b = histogram[512:768]
            
            # Создаем график
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = range(256)
            ax.plot(x, hist_r, color='red', label='Red', alpha=0.7)
            ax.plot(x, hist_g, color='green', label='Green', alpha=0.7)
            ax.plot(x, hist_b, color='blue', label='Blue', alpha=0.7)
            
            ax.set_xlabel('Color Intensity (0-255)')
            ax.set_ylabel('Pixel Count')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Сохраняем в буфер
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Конвертируем в base64
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode()
            return f"data:image/png;base64,{img_base64}"
            
        except ImportError:
            print("Matplotlib не установлен. Установите: pip install matplotlib")
            return None
        except Exception as e:
            print(f"Error creating histogram chart: {e}")
            return None