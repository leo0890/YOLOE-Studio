"""
Gradio component factories.

Каждая фабрика возвращает готовый компонент с корректными
параметрами. UI-код становится декларативным: что, а не как.
"""
from __future__ import annotations

from typing import Any, Optional

import gradio as gr

from config import CFG


def make_header() -> gr.Markdown:
    """Заголовок с описанием."""
    return gr.Markdown(
        """
        <div style="text-align: center; padding: 8px 0 4px;">
            <h1 style="margin: 0; font-weight: 700; font-size: 28px;">
                🔍 YOLOE-26
            </h1>
            <p style="margin: 4px 0 0; color: #64748b; font-size: 14px;">
                Обнаружение и сегментация объектов в реальном времени
            </p>
        </div>
        """
    )


def make_video_pane(label: str, height: int) -> gr.Image:
    """Панель для видео/изображения (read-only)."""
    return gr.Image(
        label=label,
        type="numpy",
        interactive=False,
        height=height,
        show_label=True,
        container=True,
    )


def make_status() -> gr.Textbox:
    """Строка статистики."""
    return gr.Textbox(
        label="📊 Статистика",
        lines=2,
        interactive=False,
        show_copy_button=False,
    )


def make_prompt_input(default: str) -> gr.Textbox:
    return gr.Textbox(
        label="Текстовые подсказки",
        value=default,
        placeholder="person, car, bus, dog, cat...",
        lines=1,
    )


def make_slider(
    label: str,
    minimum: float,
    maximum: float,
    value: float,
    step: float,
    info: Optional[str] = None,
) -> gr.Slider:
    return gr.Slider(
        minimum=minimum,
        maximum=maximum,
        value=value,
        step=step,
        label=label,
        info=info,
    )


def make_upload() -> gr.Image:
    return gr.Image(
        label="Загрузить изображение",
        type="numpy",
        sources=["upload", "clipboard"],
        height=180,
        show_download_button=False,
    )


def make_json_view() -> gr.Code:
    return gr.Code(
        label="JSON с детекциями",
        language="json",
        lines=12,
        interactive=False,
    )


def make_source_selector() -> gr.Radio:
    return gr.Radio(
        choices=["IP-камера", "Веб-камера", "Видеофайл"],
        value="IP-камера",
        label="Тип источника",
        info="IP-камера: rtsp/http, Веб-камера: 0, Видеофайл: путь к mp4",
    )


def make_camera_url() -> gr.Textbox:
    return gr.Textbox(
        label="URL / путь / индекс",
        placeholder="http://IP:8080/video | rtsp://... | 0 | C:\\video.mp4",
        lines=1,
    )


def make_camera_status() -> gr.Textbox:
    return gr.Textbox(
        label="Статус подключения",
        lines=1,
        interactive=False,
    )


def make_rec_path() -> gr.Textbox:
    return gr.Textbox(
        label="Путь для сохранения",
        value="output.mp4",
        lines=1,
    )


def make_rec_status() -> gr.Textbox:
    return gr.Textbox(
        label="Статус записи",
        lines=1,
        interactive=False,
    )


def make_hints() -> gr.Markdown:
    return gr.Markdown(
        """
        ---
        ### 📖 Подсказки

        - **Источники видео**
          - IP-камера: `http://IP:8080/video` или `rtsp://user:pass@IP:554/stream`
          - Веб-камера: `0` (или индекс устройства)
          - Видеофайл: полный путь, например `C:\\video.mp4`
        - **Трекинг** — присваивает объектам ID между кадрами (ByteTrack)
        - **Запись** — сохраняет видео с разметкой в mp4
        - **JSON** — выгружается после каждой обработки, удобно для интеграции
        - Если объектов 0 при высоком пороге — снизьте «Порог уверенности»
        """
    )