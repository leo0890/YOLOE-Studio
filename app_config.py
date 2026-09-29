"""
Application configuration.

Frozen dataclasses — единый источник правды для всех параметров.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    weights: str = "yoloe-26s-seg.pt"
    default_prompt: tuple[str, ...] = ("person", "car", "bus")
    cache_size: int = 32


@dataclass(frozen=True)
class CameraConfig:
    max_fails: int = 5
    reconnect_delay: float = 1.0
    buffer_size: int = 1
    read_poll_interval: float = 0.01


@dataclass(frozen=True)
class UIConfig:
    default_conf: float = 0.25
    default_iou: float = 0.45
    default_imgsz: int = 640
    default_min_area: int = 0
    default_timer: float = 0.5
    image_height: int = 400


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


CFG = AppConfig()