"""
Service layer: adapter между domain и UI.

Здесь формируются callable'ы, которые UI передаёт в Gradio.
Все зависимости (engine, state, recorder) инжектируются снаружи.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

import gradio as gr
import numpy as np

from domain import CameraReader, DetectionEngine, StateStore, VideoRecorder
from config import CFG


log = logging.getLogger("yoloe.services")


class ServiceContainer:
    """DI-контейнер: собирает всё, что нужно UI."""

    def __init__(
        self,
        engine: DetectionEngine,
        state: StateStore,
        recorder: VideoRecorder,
    ) -> None:
        self.engine = engine
        self.state = state
        self.recorder = recorder

    # ============================================================
    # IMAGE
    # ============================================================
    def process_image(
        self,
        image: Optional[np.ndarray],
        prompt: str,
        conf: float,
        iou: float,
        imgsz: int,
        min_area: int,
    ) -> tuple[Any, Any, str, str]:
        if image is None:
            return None, None, "❌ Загрузите изображение.", "{}"

        try:
            annotated, status, detections = self.engine.infer(
                image, prompt, conf, iou, imgsz, min_area, track=False
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("process_image failed")
            return image, image, f"❌ Ошибка: {exc}", "{}"

        payload = self._payload(detections)
        log.info("Image processed: %s", status)
        return image, annotated, status, payload

    # ============================================================
    # STREAM
    # ============================================================
    def process_stream(
        self,
        prompt: str,
        conf: float,
        iou: float,
        imgsz: int,
        min_area: int,
        show_raw: bool,
        track: bool,
    ) -> tuple[Any, Any, Any, Any]:
        reader = self.state.reader
        if reader is None or not reader.running:
            return gr.update(), gr.update(), gr.update(), gr.update()

        frame = reader.read()
        if frame is None:
            self.state.register_fail()
            if self.state.should_reconnect():
                self.state.reset_fails()
                log.warning("Stream lost, reconnecting...")
            return gr.update(), gr.update(), gr.update(), gr.update()

        self.state.reset_fails()

        try:
            annotated, status, detections = self.engine.infer(
                frame, prompt, conf, iou, imgsz, min_area, track=track
            )
        except Exception:  # noqa: BLE001
            log.exception("process_stream failed")
            return gr.update(), gr.update(), gr.update(), gr.update()

        import cv2
        raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if show_raw else None
        if self.recorder.active:
            self.recorder.write(annotated)

        return raw_rgb, annotated, status, self._payload(detections)

    # ============================================================
    # CAMERA
    # ============================================================
    def on_connect(self, source_type: str, url: str) -> tuple[str, gr.Timer]:
        url = (url or "").strip()

        if source_type == "Веб-камера":
            source: str | int = self._int_or(url, 0) if url else 0
        else:
            if not url:
                return "❌ Укажите URL или путь.", gr.Timer(active=False)
            source = url

        try:
            reader = CameraReader(source, CFG.camera)
            if not reader.ok:
                return f"❌ Не удалось открыть: {source}", gr.Timer(active=False)
            reader.start()
            self.state.attach(reader, str(source))
            return f"✅ Источник подключён: {source}", gr.Timer(active=True)
        except Exception as exc:  # noqa: BLE001
            log.exception("on_connect failed")
            return f"❌ Ошибка подключения: {exc}", gr.Timer(active=False)

    def on_disconnect(self) -> tuple[str, gr.Timer]:
        self.state.detach()
        self.recorder.stop()
        return "⛔ Источник отключён.", gr.Timer(active=False)

    # ============================================================
    # TIMER
    # ============================================================
    @staticmethod
    def on_update_timer(value: float) -> Any:
        # Возвращаем gr.update() — Gradio сам найдёт компонент по output
        return gr.update()

    # ============================================================
    # RECORDER
    # ============================================================
    def on_start_recording(self, path: str) -> str:
        return self.recorder.start(path)

    def on_stop_recording(self) -> str:
        return self.recorder.stop()

    # ============================================================
    # HELPERS
    # ============================================================
    @staticmethod
    def _payload(detections: list[Any]) -> str:
        data = {
            "timestamp": datetime.now().isoformat(),
            "count": len(detections),
            "objects": [d.to_dict() for d in detections],
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    @staticmethod
    def _int_or(value: str, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
