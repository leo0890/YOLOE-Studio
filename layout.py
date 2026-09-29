"""
Layout builder.

Собирает всё дерево компонентов Gradio и привязывает обработчики.
Никакой логики — только декларация структуры.
"""
from __future__ import annotations

from typing import Any, Callable

import gradio as gr

from config import CFG
from ui import components as C
from ui.theme import build_theme


# Тип функции-обработчика: принимает *args, возвращает *outputs
Handler = Callable[..., Any]


class Handlers:
    """
    Контейнер обработчиков.

    Позволяет UI-слою ничего не знать о деталях сервисов —
    он получает уже готовые callable.
    """

    def __init__(
        self,
        process_image: Handler,
        process_stream: Handler,
        on_connect: Handler,
        on_disconnect: Handler,
        on_update_timer: Handler,
        on_start_recording: Handler,
        on_stop_recording: Handler,
    ) -> None:
        self.process_image = process_image
        self.process_stream = process_stream
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.on_update_timer = on_update_timer
        self.on_start_recording = on_start_recording
        self.on_stop_recording = on_stop_recording


def build_ui(h: Handlers) -> gr.Blocks:
    """Собирает Blocks с компонентами и событиями."""
    ui_cfg = CFG.ui

    with gr.Blocks(
        title="YOLOE-26 — Detection & Segmentation",
        theme=build_theme(),
        css=_CUSTOM_CSS,
        fill_height=False,
    ) as demo:

        C.make_header()

        # ========================================================
        # VIDEO PANES
        # ========================================================
        with gr.Row(equal_height=True):
            raw_video = C.make_video_pane(
                "🎥 Сырое видео", ui_cfg.image_height
            )
            output_image = C.make_video_pane(
                "🎯 Результат YOLOE", ui_cfg.image_height
            )

        status_output = C.make_status()

        # ========================================================
        # CONTROLS
        # ========================================================
        with gr.Row(equal_height=False):

            # ---- Column 1: detection params ----
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### ⚙️ Параметры детекции")
                prompt_input = C.make_prompt_input(
                    ", ".join(CFG.model.default_prompt)
                )
                with gr.Row():
                    conf_slider = C.make_slider(
                        "Порог уверенности",
                        0.0, 1.0, ui_cfg.default_conf, 0.01,
                        info="Ниже — больше объектов",
                    )
                    iou_slider = C.make_slider(
                        "Порог IoU (NMS)",
                        0.1, 1.0, ui_cfg.default_iou, 0.05,
                    )
                with gr.Row():
                    imgsz_slider = C.make_slider(
                        "imgsz", 320, 1280, ui_cfg.default_imgsz, 32,
                        info="Кратно 32",
                    )
                    min_area_slider = C.make_slider(
                        "Мин. площадь (px²)",
                        0, 50_000, ui_cfg.default_min_area, 100,
                    )
                tracking_checkbox = gr.Checkbox(
                    label="🎯 Трекинг объектов (ID между кадрами)",
                    value=False,
                    info="Включает ByteTrack — объекты получают постоянный ID",
                )

            # ---- Column 2: image upload ----
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 📷 Обработка изображения")
                input_image = C.make_upload()
                process_img_btn = gr.Button(
                    "🖼 Обработать изображение",
                    variant="primary",
                    size="lg",
                )
                with gr.Accordion("📄 JSON с детекциями", open=False):
                    json_output = C.make_json_view()

            # ---- Column 3: video source ----
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 🎥 Источник видео")
                source_type = C.make_source_selector()
                camera_url = C.make_camera_url()
                with gr.Row():
                    connect_btn = gr.Button(
                        "🔌 Подключить", variant="primary", size="lg"
                    )
                    disconnect_btn = gr.Button("⛔ Отключить", size="lg")
                show_raw_checkbox = gr.Checkbox(
                    label="Показывать сырое видео", value=True
                )
                refresh_slider = C.make_slider(
                    "Частота обновления (сек)",
                    0.1, 2.0, ui_cfg.default_timer, 0.1,
                )
                camera_status = C.make_camera_status()

        # ========================================================
        # RECORDING
        # ========================================================
        with gr.Accordion("📹 Запись видео", open=False):
            with gr.Row():
                rec_path = C.make_rec_path()
                rec_start_btn = gr.Button(
                    "▶ Начать запись", variant="primary"
                )
                rec_stop_btn = gr.Button("⏹ Остановить")
            rec_status = C.make_rec_status()

        # ========================================================
        # TIMER (скрытый компонент)
        # ========================================================
        timer = gr.Timer(value=ui_cfg.default_timer, active=False)

        C.make_hints()

        # ========================================================
        # EVENTS
        # ========================================================
        process_img_btn.click(
            fn=h.process_image,
            inputs=[input_image, prompt_input, conf_slider, iou_slider,
                    imgsz_slider, min_area_slider],
            outputs=[raw_video, output_image, status_output, json_output],
        )

        connect_btn.click(
            fn=h.on_connect,
            inputs=[source_type, camera_url],
            outputs=[camera_status, timer],
        )

        disconnect_btn.click(
            fn=h.on_disconnect,
            inputs=[],
            outputs=[camera_status, timer],
        )

        refresh_slider.change(
            fn=h.on_update_timer,
            inputs=[refresh_slider],
            outputs=[refresh_slider],
        )

        timer.tick(
            fn=h.process_stream,
            inputs=[prompt_input, conf_slider, iou_slider, imgsz_slider,
                    min_area_slider, show_raw_checkbox, tracking_checkbox],
            outputs=[raw_video, output_image, status_output, json_output],
        )

        rec_start_btn.click(
            fn=h.on_start_recording,
            inputs=[rec_path],
            outputs=[rec_status],
        )

        rec_stop_btn.click(
            fn=h.on_stop_recording,
            inputs=[],
            outputs=[rec_status],
        )

    return demo


# ============================================================
# CUSTOM CSS (дополняет тему)
# ============================================================
_CUSTOM_CSS = """
/* Скрываем футер Gradio */
footer { display: none !important; }

/* Улучшаем визуал видео-панелей */
.gradio-container {
    max-width: 1600px !important;
    margin: 0 auto !important;
}

/* Кнопки */
button.primary {
    font-weight: 600 !important;
    letter-spacing: 0.02em;
}

/* Скругления для изображений */
.image-container img {
    border-radius: 8px !important;
}

/* Аккуратные аккордеоны */
.accordion-header {
    font-weight: 600 !important;
    color: #334155 !important;
}

/* Заголовки секций */
h3 {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #1e293b !important;
    margin-top: 4px !important;
    margin-bottom: 8px !important;
    padding-bottom: 6px;
    border-bottom: 1px solid #e2e8f0;
}

/* Статус */
textarea[readonly] {
    background: #f8fafc !important;
    color: #0f172a !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}
"""