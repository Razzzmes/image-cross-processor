from enum import Enum

class CrossType(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    BOTH = "both"

class ImageFormat(str, Enum):
    JPEG = "JPEG"
    PNG = "PNG"
    GIF = "GIF"
    BMP = "BMP"
    WEBP = "WEBP"

class ColorChannel(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    ALL = "all"