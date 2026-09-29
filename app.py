"""
YOLOE-26 realtime detection & segmentation — v0.6.0

Читает с IP-камеры / вебки / файла, гоняет через YOLOE с текстовыми
промптами, рисует боксы + маски, пишет mp4 + JSONL/CSV, умеет
дообучаться онлайн по фидбэку пользователя, поддерживает горячие
клавиши, современный IT-интерфейс (тёмная тема, sidebar, dashboard).

Запуск:  python app.py
Ставил на: ultralytics>=8.3, gradio>=4.44, torch>=2.2, opencv-python, pyyaml

CHANGELOG
---------
0.6.0  — современный IT-интерфейс: тёмная тема, sidebar с хоткеями,
         стеклянные панели, метрики-дашборд, статус-бар снизу,
         kbd-бейджи, modal-help, toast-переработан
0.5.1  — глобальные горячие клавиши (Space/R/L/C/1/2/3/P/T/A/Y/S/H/?)
0.5.0  — онлайн-обучение (3 уровня): L1 per-class conf, L2 few-shot
         refinement, L3 экспорт кропов для ручного fine-tune
0.4.1  — убран deprecated half, fp16 через model.half(), авто-фолбэк imgsz
0.4.0  — async pipeline, неблокирующий reconnect, warmup, метрики,
         YAML-конфиг, ROI, JSONL-экспорт, webhook, health-check, SSRF
0.3.x  — фиксы reconnect, recorder, deadlock set_classes, fps
0.2.x  — v8-seg
0.1.0  — первый рабочий вариант
"""

from __future__ import annotations

import atexit
import csv
import hashlib
import io
import ipaddress
import json
import logging
import os
import queue
import random
import signal
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import gradio as gr
import numpy as np
import torch
from ultralytics import YOLOE

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


__author__ = "your_nick"
__version__ = "0.6.0"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
    datefmt="%H:%M:%S",
)

for _noisy in ("httpx", "httpcore", "urllib3", "urllib3.connectionpool",
               "asyncio", "python_multipart", "multipart"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

log = logging.getLogger("yoloe.app")


# =============================================================================
# 1. КОНФИГ
# =============================================================================

def _env(key: str, default: Any, cast=str) -> Any:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return cast(val)
    except Exception:
        log.warning("bad env %s=%r, using default %r", key, val, default)
        return default


@dataclass
class ModelConfig:
    weights: str = field(default_factory=lambda: _env("YOLOE_WEIGHTS", "yoloe-26s-seg.pt"))
    default_prompt: tuple[str, ...] = ("person", "car", "bus")
    cache_size: int = 32
    warmup_imgsz: int = 640
    max_infer_retries: int = 2
    use_fp16: bool = True
    auto_imgsz: bool = True
    auto_imgsz_p95_ms: float = 400.0
    auto_imgsz_min: int = 416
    auto_imgsz_step: int = 64
    auto_imgsz_check_every: int = 30


@dataclass
class CameraConfig:
    max_fails: int = 5
    reconnect_delay: float = 1.0
    reconnect_backoff_max: float = 15.0
    buffer_size: int = 1
    read_poll_interval: float = 0.25
    stop_timeout: float = 2.0
    open_timeout_ms: int = 5000
    read_timeout_ms: int = 5000
    release_settle_s: float = 0.4


@dataclass
class UIConfig:
    default_conf: float = 0.25
    default_iou: float = 0.45
    default_imgsz: int = 640
    default_min_area: int = 0
    default_timer: float = 0.5
    image_height: int = 400
    mask_alpha: float = 0.45


@dataclass
class ServerConfig:
    host: str = field(default_factory=lambda: _env("YOLOE_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env("YOLOE_PORT", 7860, int))
    share: bool = False
    output_dir: str = field(default_factory=lambda: _env("YOLOE_OUT", "recordings"))
    auth_user: str = field(default_factory=lambda: _env("YOLOE_USER", ""))
    auth_pass: str = field(default_factory=lambda: _env("YOLOE_PASS", ""))
    allow_url_schemes: tuple[str, ...] = ("http", "https", "rtsp", "rtmp", "rtmps")
    block_private_hosts: bool = True
    webhook_url: str = field(default_factory=lambda: _env("YOLOE_WEBHOOK", ""))
    webhook_timeout: float = 3.0


@dataclass
class PipelineConfig:
    infer_queue_size: int = 1
    result_queue_size: int = 4
    infer_thread_name: str = "infer"
    reconnect_thread_name: str = "reconnect"


@dataclass
class OnlineLearnConfig:
    enabled: bool = True
    buffer_size: int = 2000
    min_examples_per_class: int = 20
    conf_step: float = 0.02
    conf_min: float = 0.10
    conf_max: float = 0.90
    recalib_every: int = 50
    refine_every: int = 100
    refine_blend: float = 0.3
    buffer_dir: str = "online_buffer"
    save_crops: bool = True
    crop_min_area: int = 400


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    online: OnlineLearnConfig = field(default_factory=OnlineLearnConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        if not _HAS_YAML:
            log.warning("pyyaml not installed, using defaults")
            return cls()
        p = Path(path)
        if not p.exists():
            log.info("config %s not found, using defaults", p)
            return cls()
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            log.warning("config parse failed (%s), using defaults", exc)
            return cls()

        cfg = cls()
        for section in ("model", "camera", "ui", "server", "pipeline", "online"):
            data = raw.get(section) or {}
            target = getattr(cfg, section)
            for k, v in data.items():
                if hasattr(target, k):
                    cur = getattr(target, k)
                    if isinstance(cur, tuple) and isinstance(v, list):
                        v = tuple(v)
                    setattr(target, k, v)
                else:
                    log.warning("unknown config key %s.%s", section, k)
        return cfg


CFG = AppConfig.from_yaml(os.environ.get("YOLOE_CONFIG", "config.yaml"))


# =============================================================================
# 2. ЦВЕТА
# =============================================================================

_NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "person":     (255,  80,  80),
    "car":        ( 80, 255,  80),
    "bus":        ( 80, 160, 255),
    "truck":      (255, 200,  80),
    "bicycle":    (200,  80, 255),
    "motorcycle": (255,  80, 200),
    "dog":        (255, 255,  80),
    "cat":        ( 80, 255, 255),
    "bird":       (180, 255, 120),
    "horse":      (255, 160, 120),
}
DEFAULT_COLOR = (200, 200, 200)


def color_for(name: str) -> tuple[int, int, int]:
    if name in _NAMED_COLORS:
        return _NAMED_COLORS[name]
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:6], 16)
    return (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)


# =============================================================================
# 3. МЕТРИКИ
# =============================================================================

class RollingMetrics:
    def __init__(self, window: int = 30) -> None:
        self._buf: deque[float] = deque(maxlen=window)

    def push(self, ms: float) -> None:
        self._buf.append(ms)

    def p(self, pct: float) -> float:
        if not self._buf:
            return 0.0
        arr = np.fromiter(self._buf, dtype=np.float32)
        return float(np.percentile(arr, pct))

    def mean(self) -> float:
        if not self._buf:
            return 0.0
        return float(np.mean(np.fromiter(self._buf, dtype=np.float32)))

    def __len__(self) -> int:
        return len(self._buf)


class MetricsStore:
    def __init__(self) -> None:
        self.infer = RollingMetrics(30)
        self.pre = RollingMetrics(30)
        self.post = RollingMetrics(30)
        self._dropped_infer = 0
        self._dropped_recorder = 0
        self._lock = threading.Lock()

    def drop_infer(self) -> None:
        with self._lock:
            self._dropped_infer += 1

    def drop_recorder(self) -> None:
        with self._lock:
            self._dropped_recorder += 1

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            di, dr = self._dropped_infer, self._dropped_recorder
        return {
            "infer_p50": round(self.infer.p(50), 1),
            "infer_p95": round(self.infer.p(95), 1),
            "infer_mean": round(self.infer.mean(), 1),
            "dropped_infer": di,
            "dropped_recorder": dr,
        }


METRICS = MetricsStore()


# =============================================================================
# 4. БЕЗОПАСНОСТЬ URL (SSRF)
# =============================================================================

def _is_private_host(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except Exception:
            return False
        for info in infos:
            try:
                ip = ipaddress.ip_address(info[4][0])
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    return True
            except ValueError:
                continue
        return False


def validate_source_url(url: str) -> tuple[bool, str]:
    url = (url or "").strip()
    if not url:
        return True, ""
    if url.isdigit():
        return True, ""

    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme:
        return True, ""

    if parsed.scheme not in CFG.server.allow_url_schemes:
        return False, f"схема {parsed.scheme!r} запрещена (allowed: {CFG.server.allow_url_schemes})"

    host = parsed.hostname or ""
    if not host:
        return False, "нет хоста в URL"

    if CFG.server.block_private_hosts:
        try:
            ip = ipaddress.ip_address(host)
            if ip.is_link_local:
                return False, f"link-local адрес запрещён: {host}"
        except ValueError:
            pass

    return True, ""


# =============================================================================
# 5. КАМЕРА
# =============================================================================

class CameraReader:
    def __init__(self, source: str | int, cfg: CameraConfig) -> None:
        self._source = source
        self._cfg = cfg
        self._cap = cv2.VideoCapture(source)

        for prop, val in (
            (cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, cfg.open_timeout_ms),
            (cv2.CAP_PROP_READ_TIMEOUT_MSEC, cfg.read_timeout_ms),
            (cv2.CAP_PROP_BUFFERSIZE, cfg.buffer_size),
        ):
            try:
                self._cap.set(prop, val)
            except Exception as exc:
                log.debug("cannot set cap prop %s: %s", prop, exc)

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._release_done = threading.Event()

    @property
    def ok(self) -> bool:
        return self._cap.isOpened()

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> bool:
        if not self.ok:
            return False
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name=f"cam-{self._source}", daemon=True
        )
        self._thread.start()
        log.info("CameraReader started: %s", self._source)
        return True

    def _loop(self) -> None:
        try:
            while self._running:
                try:
                    ret, frame = self._cap.read()
                except Exception as exc:
                    log.warning("cap.read() raised: %s", exc)
                    break
                if ret and frame is not None:
                    with self._lock:
                        self._frame = frame
                else:
                    time.sleep(self._cfg.read_poll_interval)
        finally:
            try:
                self._cap.release()
            except Exception:
                pass
            self._release_done.set()
            log.debug("CameraReader loop exited: %s", self._source)

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._cfg.stop_timeout)
            if self._thread.is_alive():
                self._release_done.wait(timeout=1.0)
                if not self._release_done.is_set():
                    log.warning(
                        "camera thread did not stop in %.1fs, releasing anyway",
                        self._cfg.stop_timeout,
                    )
                    try:
                        self._cap.release()
                    except Exception:
                        pass
        with self._lock:
            self._frame = None
        try:
            self._cap.release()
        except Exception:
            pass
        log.info("CameraReader stopped: %s", self._source)


# =============================================================================
# 6. ДЕТЕКЦИЯ
# =============================================================================

@dataclass
class Detection:
    obj_id: Optional[int]
    cls: str
    confidence: float
    bbox: tuple[int, int, int, int]
    area: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.obj_id,
            "class": self.cls,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "area": self.area,
        }


@dataclass
class InferResult:
    annotated: np.ndarray
    status: str
    detections: list[Detection]
    infer_ms: float
    pre_ms: float
    post_ms: float
    imgsz_used: int


class DetectionEngine:
    def __init__(self, cfg: ModelConfig) -> None:
        log.info("loading YOLOE: %s", cfg.weights)
        self._model = YOLOE(cfg.weights)
        self._cfg = cfg

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._model.to(self._device)
        except Exception as exc:
            log.warning("cannot move to %s (%s), fallback cpu", self._device, exc)
            self._device = "cpu"

        self._fp16_enabled = False
        if self._device == "cuda" and cfg.use_fp16:
            self._try_enable_fp16()

        self._names: tuple[str, ...] = ()
        self._classes_lock = threading.Lock()
        self._pe_cache: "OrderedDict[tuple[str, ...], Any]" = OrderedDict()
        self._pe_lock = threading.Lock()

        self._auto_imgsz = int(cfg.warmup_imgsz)
        self._auto_imgsz_lock = threading.Lock()
        self._frames_since_check = 0

        log.info("model ready on %s (fp16=%s)", self._device, self._fp16_enabled)

    def _try_enable_fp16(self) -> None:
        candidates = (
            lambda: self._model.model.half(),
            lambda: setattr(self._model.model, "fp16", True),
            lambda: setattr(self._model, "half", True),
        )
        for i, fn in enumerate(candidates):
            try:
                fn()
                self._fp16_enabled = True
                log.info("fp16 enabled via strategy #%d", i + 1)
                return
            except Exception as exc:
                log.debug("fp16 strategy #%d failed: %s", i + 1, exc)
        log.info("fp16 not available, running in fp32")

    @property
    def device(self) -> str:
        return self._device

    @property
    def imgsz(self) -> int:
        with self._auto_imgsz_lock:
            return self._auto_imgsz

    @property
    def pe_cache(self) -> "OrderedDict[tuple[str, ...], Any]":
        return self._pe_cache

    @property
    def pe_lock(self) -> threading.Lock:
        return self._pe_lock

    def warmup(self) -> None:
        try:
            dummy = np.zeros((CFG.model.warmup_imgsz, CFG.model.warmup_imgsz, 3), dtype=np.uint8)
            log.info("warming up model (imgsz=%d)...", CFG.model.warmup_imgsz)
            t0 = time.perf_counter()
            self.infer(
                dummy,
                ", ".join(self._cfg.default_prompt),
                CFG.ui.default_conf,
                CFG.ui.default_iou,
                CFG.model.warmup_imgsz,
                0,
                track=False,
            )
            log.info("warmup done in %.0f ms", (time.perf_counter() - t0) * 1000)
        except Exception as exc:
            log.warning("warmup failed: %s", exc)

    def infer(
        self,
        frame: np.ndarray,
        prompt: str,
        conf: float,
        iou: float,
        imgsz: int,
        min_area: int,
        track: bool = False,
        roi: Optional[np.ndarray] = None,
    ) -> InferResult:
        names = self._parse_prompt(prompt)
        self._ensure_classes(names)

        requested_imgsz = self._round_to_32(imgsz)
        imgsz_fixed = self._effective_imgsz(requested_imgsz)

        t0 = time.perf_counter()
        r = None
        last_exc: Optional[Exception] = None
        for attempt in range(self._cfg.max_infer_retries + 1):
            r = self._run_model(frame, conf, iou, imgsz_fixed, track)
            if r is not None:
                break
            last_exc = Exception("model returned None")
            time.sleep(0.05 * (attempt + 1))
        infer_ms = (time.perf_counter() - t0) * 1000

        if r is None:
            return InferResult(
                annotated=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy(),
                status=f"ошибка инференса: {last_exc}",
                detections=[],
                infer_ms=infer_ms,
                pre_ms=0.0,
                post_ms=0.0,
                imgsz_used=imgsz_fixed,
            )

        t_post = time.perf_counter()
        if min_area > 0:
            r = self._filter_by_area(r, min_area)
        if roi is not None:
            r = self._filter_by_roi(r, roi)
        r = self._filter_by_per_class_conf(r)

        annotated = self._annotate(frame, r)
        detections = self._extract_detections(r)
        post_ms = (time.perf_counter() - t_post) * 1000

        status = self._build_status(detections, imgsz_fixed, track, infer_ms)
        METRICS.infer.push(infer_ms)
        METRICS.post.push(post_ms)

        self._maybe_adjust_imgsz(requested_imgsz, imgsz_fixed)

        return InferResult(annotated, status, detections, infer_ms, 0.0, post_ms, imgsz_fixed)

    def _effective_imgsz(self, requested: int) -> int:
        if not self._cfg.auto_imgsz:
            return requested
        with self._auto_imgsz_lock:
            cap = self._auto_imgsz
        return min(requested, cap)

    def _maybe_adjust_imgsz(self, requested: int, used: int) -> None:
        if not self._cfg.auto_imgsz:
            return

        self._frames_since_check += 1
        if self._frames_since_check < self._cfg.auto_imgsz_check_every:
            return
        self._frames_since_check = 0

        if len(METRICS.infer) < 10:
            return

        p95 = METRICS.infer.p(95)
        with self._auto_imgsz_lock:
            current = self._auto_imgsz

        if p95 > self._cfg.auto_imgsz_p95_ms and current > self._cfg.auto_imgsz_min:
            new_val = max(self._cfg.auto_imgsz_min, current - self._cfg.auto_imgsz_step)
            new_val = self._round_to_32(new_val)
            with self._auto_imgsz_lock:
                self._auto_imgsz = new_val
            log.warning(
                "auto-imgsz: p95=%.0fms > %.0fms, lowering %d → %d",
                p95, self._cfg.auto_imgsz_p95_ms, current, new_val,
            )
        elif p95 < self._cfg.auto_imgsz_p95_ms * 0.5 and current < requested:
            new_val = min(requested, current + self._cfg.auto_imgsz_step)
            new_val = self._round_to_32(new_val)
            if new_val != current:
                with self._auto_imgsz_lock:
                    self._auto_imgsz = new_val
                log.info(
                    "auto-imgsz: p95=%.0fms fast, raising %d → %d",
                    p95, current, new_val,
                )

    @staticmethod
    def _parse_prompt(prompt: str) -> list[str]:
        raw = [n.strip() for n in prompt.split(",") if n.strip()]
        return list(dict.fromkeys(raw)) or list(CFG.model.default_prompt)

    @staticmethod
    def _round_to_32(value: float) -> int:
        v = max(32, int(value))
        return max(32, ((v + 31) // 32) * 32)

    def _ensure_classes(self, names: list[str]) -> None:
        key = tuple(names)
        with self._classes_lock:
            if self._names == key:
                return
            pe = self._embeddings(key)
            if self._names == key:
                return
            self._model.set_classes(list(names), pe)
            self._names = key
        log.info("classes set: %s", names)

    def _embeddings(self, names: tuple[str, ...]) -> Any:
        with self._pe_lock:
            if names in self._pe_cache:
                self._pe_cache.move_to_end(names)
                return self._pe_cache[names]
        pe = self._model.get_text_pe(list(names))
        with self._pe_lock:
            if names in self._pe_cache:
                self._pe_cache.move_to_end(names)
                return self._pe_cache[names]
            if len(self._pe_cache) >= self._cfg.cache_size:
                self._pe_cache.popitem(last=False)
            self._pe_cache[names] = pe
        return pe

    def _run_model(self, frame, conf, iou, imgsz, track):
        kwargs = dict(conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        try:
            if track:
                results = self._model.track(
                    frame, persist=True, tracker="bytetrack.yaml", **kwargs
                )
            else:
                results = self._model(frame, **kwargs)
        except Exception as exc:
            log.exception("inference failed: %s", exc)
            return None
        return results[0] if results else None

    @staticmethod
    def _filter_by_area(r: Any, min_area: int) -> Any:
        if r.boxes is None or len(r.boxes) == 0:
            return r
        try:
            xyxy = r.boxes.xyxy.cpu().numpy()
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            keep = np.where(areas >= min_area)[0]
            return r[keep.tolist()] if len(keep) else r[:0]
        except Exception as exc:
            log.warning("area filter failed: %s", exc)
            return r

    @staticmethod
    def _filter_by_roi(r: Any, roi: np.ndarray) -> Any:
        if r.boxes is None or len(r.boxes) == 0 or roi is None:
            return r
        try:
            xyxy = r.boxes.xyxy.cpu().numpy()
            centers = np.stack(
                [(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2],
                axis=1,
            )
            keep = []
            for i, c in enumerate(centers):
                if cv2.pointPolygonTest(roi.astype(np.float32), (float(c[0]), float(c[1])), False) >= 0:
                    keep.append(i)
            return r[keep] if keep else r[:0]
        except Exception as exc:
            log.warning("roi filter failed: %s", exc)
            return r

    @staticmethod
    def _filter_by_per_class_conf(r: Any) -> Any:
        learner = globals().get("ONLINE_LEARNER")
        if learner is None:
            return r
        if r.boxes is None or len(r.boxes) == 0:
            return r
        try:
            names_map = r.names if isinstance(r.names, dict) else {i: n for i, n in enumerate(r.names)}
            keep = []
            cls_arr = r.boxes.cls.cpu().numpy().astype(int)
            conf_arr = r.boxes.conf.cpu().numpy()
            for i, (ci, cf) in enumerate(zip(cls_arr, conf_arr)):
                name = names_map.get(int(ci), f"class_{int(ci)}")
                thr = learner.get_conf_override(name)
                if thr is None or cf >= thr:
                    keep.append(i)
            return r[keep] if keep else r[:0]
        except Exception as exc:
            log.warning("per-class conf filter failed: %s", exc)
            return r

    def _annotate(self, frame: np.ndarray, r: Any) -> np.ndarray:
        base = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
        try:
            base = self._draw_masks(base, r)
            base = self._draw_custom_boxes(base, r)
        except Exception as exc:
            log.exception("annotate failed: %s", exc)
        return base

    def _draw_masks(self, img: np.ndarray, r: Any) -> np.ndarray:
        masks = getattr(r, "masks", None)
        if masks is None or masks.data is None or len(masks.data) == 0:
            return img

        names_map = self._names_map()
        cls_list = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []
        h, w = img.shape[:2]

        try:
            mdata = masks.data.cpu().numpy()
        except Exception as exc:
            log.warning("mask extract failed: %s", exc)
            return img

        overlay = img.copy()
        for i in range(mdata.shape[0]):
            cls_idx = int(cls_list[i]) if i < len(cls_list) else -1
            name = names_map.get(cls_idx, f"class_{cls_idx}")
            color = color_for(name)
            mask = mdata[i]
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            overlay[mask > 0.5] = color
        alpha = CFG.ui.mask_alpha
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    def _draw_custom_boxes(self, img: np.ndarray, r: Any) -> np.ndarray:
        if r.boxes is None or len(r.boxes) == 0:
            return img
        names_map = self._names_map()
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            name = names_map.get(int(cls), f"class_{int(cls)}")
            color = color_for(name)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                img, name, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
            )
        return img

    def _extract_detections(self, r: Any) -> list[Detection]:
        if r.boxes is None or len(r.boxes) == 0:
            return []
        names_map = self._names_map()
        ids = None
        if r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)

        out: list[Detection] = []
        for i, (cls, box, conf) in enumerate(zip(r.boxes.cls, r.boxes.xyxy, r.boxes.conf)):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            obj_id = int(ids[i]) if (ids is not None and i < len(ids)) else None
            out.append(Detection(
                obj_id=obj_id,
                cls=names_map.get(int(cls), f"class_{int(cls)}"),
                confidence=float(conf),
                bbox=(x1, y1, x2, y2),
                area=(x2 - x1) * (y2 - y1),
            ))
        return out

    def _names_map(self) -> dict[int, str]:
        names = self._model.names
        return names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}

    def _build_status(self, detections, imgsz, track, infer_ms) -> str:
        classes = sorted({d.cls for d in detections})
        parts = [f"объектов: {len(detections)}"]
        if classes:
            parts.append(", ".join(classes))
        parts.append(f"imgsz={imgsz}")
        parts.append(f"device={self._device}")
        parts.append(f"infer={infer_ms:.0f}ms")
        if self._cfg.auto_imgsz and imgsz != self._cfg.warmup_imgsz:
            parts.append("auto-imgsz")
        if track:
            ids = [d.obj_id for d in detections if d.obj_id is not None]
            if ids:
                shown = ids[:20]
                tail = f"…(+{len(ids) - 20})" if len(ids) > 20 else ""
                parts.append("id: " + ", ".join(map(str, shown)) + tail)
        return " | ".join(parts)


# =============================================================================
# 7. ЗАПИСЬ
# =============================================================================

class VideoRecorder:
    def __init__(self, fps: float = 20.0, output_dir: Path | None = None) -> None:
        self._fps = fps
        self._output_dir = Path(output_dir) if output_dir else Path(CFG.server.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._writer: Optional[cv2.VideoWriter] = None
        self._path: Optional[Path] = None
        self._size: Optional[tuple[int, int]] = None
        self._frames = 0
        self._lock = threading.Lock()

        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=64)
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

    @property
    def active(self) -> bool:
        return self._writer is not None

    @property
    def frames(self) -> int:
        return self._frames

    def _resolve_path(self, path: str) -> Path:
        out_root = self._output_dir.resolve()
        if path:
            candidate = Path(path)
            candidate = out_root / candidate.name
        else:
            candidate = out_root / f"output_{datetime.now():%Y%m%d_%H%M%S}.mp4"

        if candidate.is_symlink():
            candidate = out_root / candidate.name

        resolved = candidate.resolve()
        try:
            resolved.relative_to(out_root)
        except ValueError:
            resolved = out_root / resolved.name

        if resolved.suffix.lower() not in (".mp4", ".avi", ".mkv"):
            resolved = resolved.with_suffix(".mp4")
        return resolved

    def start(self, path: str, frame_rgb: Optional[np.ndarray] = None) -> str:
        if frame_rgb is None:
            return "нет кадра для определения размера"

        out_path = self._resolve_path(path)
        h, w = frame_rgb.shape[:2]

        with self._lock:
            if self._writer is not None:
                return "уже пишем"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, self._fps, (w, h))
            if not writer.isOpened():
                return f"не открылся файл: {out_path}"

            self._writer, self._path, self._frames = writer, out_path, 0
            self._size = (w, h)

            self._stop_flag.clear()
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._thread = threading.Thread(
                target=self._writer_loop, name="recorder", daemon=True
            )
            self._thread.start()

        log.info("recording started: %s (%dx%d @ %.1f fps)", out_path, w, h, self._fps)
        return f"пишем в {out_path} ({w}x{h})"

    def _writer_loop(self) -> None:
        while not self._stop_flag.is_set():
            try:
                frame = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is None:
                break
            try:
                with self._lock:
                    if self._writer is None:
                        break
                    self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    self._frames += 1
            except Exception as exc:
                log.warning("frame write failed: %s", exc)

    def write(self, frame_rgb: np.ndarray) -> None:
        if not self.active:
            return
        size = self._size
        if size is not None:
            h, w = frame_rgb.shape[:2]
            if (w, h) != size:
                try:
                    frame_rgb = cv2.resize(frame_rgb, size)
                except Exception as exc:
                    log.warning("resize for recorder failed: %s", exc)
                    return
        try:
            self._queue.put_nowait(frame_rgb)
        except queue.Full:
            METRICS.drop_recorder()
            log.warning("recorder queue full, dropping frame")

    def stop(self) -> str:
        with self._lock:
            if self._writer is None:
                return "не пишем"
            path, frames = self._path, self._frames
            self._stop_flag.set()

        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                log.warning("recorder thread did not stop in 3s")
            self._thread = None

        with self._lock:
            if self._writer is not None:
                self._writer.release()
            self._writer, self._path, self._frames = None, None, 0
            self._size = None
        log.info("recording stopped: %s (%d frames)", path, frames)
        return f"готово: {path} ({frames} кадров)"

    def release(self) -> None:
        self._stop_flag.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            self._size = None


# =============================================================================
# 8. ЭКСПОРТ ДЕТЕКЦИЙ (JSONL / CSV)
# =============================================================================

class DetectionLogger:
    def __init__(self, out_dir: Path) -> None:
        self._dir = out_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._jsonl: Optional[io.TextIOWrapper] = None
        self._csv: Optional[io.TextIOWrapper] = None
        self._csv_writer: Optional[csv.writer] = None
        self._lock = threading.Lock()
        self._path: Optional[Path] = None

    @property
    def active(self) -> bool:
        return self._jsonl is not None

    def start(self) -> str:
        with self._lock:
            if self._jsonl is not None:
                return "уже пишем"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            jsonl_path = self._dir / f"detections_{ts}.jsonl"
            csv_path = self._dir / f"detections_{ts}.csv"
            self._jsonl = open(jsonl_path, "w", encoding="utf-8")
            self._csv = open(csv_path, "w", encoding="utf-8", newline="")
            self._csv_writer = csv.writer(self._csv)
            self._csv_writer.writerow(["timestamp", "frame_id", "class", "conf",
                                       "x1", "y1", "x2", "y2", "area", "track_id"])
            self._path = jsonl_path
        return f"экспорт в {jsonl_path.parent}/"

    def log(self, frame_id: int, detections: list[Detection], fps: float) -> None:
        with self._lock:
            if self._jsonl is None or self._csv_writer is None:
                return
            ts = datetime.now().isoformat()
            payload = {
                "timestamp": ts,
                "frame_id": frame_id,
                "fps": round(fps, 2),
                "count": len(detections),
                "objects": [d.to_dict() for d in detections],
            }
            self._jsonl.write(json.dumps(payload, ensure_ascii=False) + "\n")
            for d in detections:
                x1, y1, x2, y2 = d.bbox
                self._csv_writer.writerow(
                    [ts, frame_id, d.cls, round(d.confidence, 4),
                     x1, y1, x2, y2, d.area, d.obj_id]
                )

    def stop(self) -> str:
        with self._lock:
            if self._jsonl is None:
                return "не пишем"
            path = self._path
            self._jsonl.close()
            if self._csv is not None:
                self._csv.close()
            self._jsonl, self._csv, self._csv_writer = None, None, None
        return f"экспорт остановлен: {path}"

    def release(self) -> None:
        try:
            self.stop()
        except Exception:
            pass


# =============================================================================
# 9. WEBHOOK
# =============================================================================

class WebhookNotifier:
    def __init__(self, url: str, timeout: float) -> None:
        self._url = url
        self._timeout = timeout
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=128)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="webhook", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if payload is None:
                break
            try:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(
                    self._url, data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    if resp.status >= 400:
                        log.warning("webhook returned %s", resp.status)
            except Exception as exc:
                log.warning("webhook post failed: %s", exc)

    def notify(self, payload: dict) -> None:
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            log.debug("webhook queue full, dropping")

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


# =============================================================================
# 10. СОСТОЯНИЕ
# =============================================================================

class StateStore:
    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        self._reader: Optional[CameraReader] = None
        self._fails = 0
        self._url = ""
        self._last_source: Any = None
        self._last_source_type: str = ""
        self._reconnect_count = 0
        self._reconnecting = False

    @property
    def reader(self) -> Optional[CameraReader]:
        return self._reader

    @property
    def running(self) -> bool:
        return self._reader is not None and self._reader.running

    @property
    def reconnecting(self) -> bool:
        with self._lock:
            return self._reconnecting

    def begin_reconnect(self) -> bool:
        with self._lock:
            if self._reconnecting:
                return False
            self._reconnecting = True
            return True

    def end_reconnect(self) -> None:
        with self._lock:
            self._reconnecting = False

    def attach(self, reader: CameraReader, url: str,
               source: Any = None, source_type: str = "") -> None:
        with self._lock:
            self._stop_reader_locked()
            self._reader, self._url, self._fails = reader, url, 0
            self._last_source = source
            self._last_source_type = source_type

    def detach(self) -> None:
        with self._lock:
            self._stop_reader_locked()
            self._reader, self._url, self._fails = None, "", 0

    def _stop_reader_locked(self) -> None:
        if self._reader is not None:
            try:
                self._reader.stop()
            except Exception as exc:
                log.warning("reader stop failed: %s", exc)

    def register_fail(self) -> int:
        with self._lock:
            self._fails += 1
            return self._fails

    def reset_fails(self) -> None:
        with self._lock:
            self._fails = 0
            self._reconnect_count = 0

    def should_reconnect(self) -> bool:
        with self._lock:
            return self._fails >= self._cfg.max_fails

    @property
    def url(self) -> str:
        return self._url

    def snapshot_reconnect_info(self) -> tuple[Any, str, int]:
        with self._lock:
            return self._last_source, self._last_source_type, self._reconnect_count

    def bump_reconnect(self) -> int:
        with self._lock:
            self._reconnect_count += 1
            return self._reconnect_count


# =============================================================================
# 11. ФОНОВЫЙ ПАЙПЛАЙН
# =============================================================================

@dataclass
class PipelineParams:
    prompt: str = ""
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    min_area: int = 0
    track: bool = False
    show_raw: bool = True
    roi: Optional[np.ndarray] = None


class FrameItem:
    __slots__ = ("frame", "params", "ts")

    def __init__(self, frame: np.ndarray, params: PipelineParams) -> None:
        self.frame = frame
        self.params = params
        self.ts = time.monotonic()


class Pipeline:
    def __init__(self, engine: DetectionEngine, state: StateStore) -> None:
        self._engine = engine
        self._state = state
        self._params = PipelineParams(prompt=", ".join(CFG.model.default_prompt))
        self._params_lock = threading.Lock()

        self._infer_q: queue.Queue[Optional[FrameItem]] = queue.Queue(
            maxsize=CFG.pipeline.infer_queue_size
        )
        self._result_q: queue.Queue[InferResult] = queue.Queue(
            maxsize=CFG.pipeline.result_queue_size
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_id = 0

        self._fps_ts = 0.0
        self._fps = 0.0
        self._fps_lock = threading.Lock()

        self._reconnect_thread: Optional[threading.Thread] = None
        self._reconnect_msg: Optional[str] = None
        self._reconnect_msg_lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._infer_loop, name=CFG.pipeline.infer_thread_name, daemon=True
        )
        self._thread.start()
        log.info("pipeline started")

    def stop(self) -> None:
        self._stop.set()
        try:
            self._infer_q.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._reconnect_thread is not None:
            self._reconnect_thread.join(timeout=2.0)
            self._reconnect_thread = None

    def set_params(self, **kwargs) -> None:
        with self._params_lock:
            for k, v in kwargs.items():
                if hasattr(self._params, k):
                    setattr(self._params, k, v)

    def submit(self, frame: np.ndarray) -> None:
        with self._params_lock:
            params = PipelineParams(**vars(self._params))
        item = FrameItem(frame, params)
        try:
            self._infer_q.put_nowait(item)
        except queue.Full:
            try:
                self._infer_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._infer_q.put_nowait(item)
            except queue.Full:
                METRICS.drop_infer()

    def latest(self) -> Optional[InferResult]:
        last: Optional[InferResult] = None
        while True:
            try:
                last = self._result_q.get_nowait()
            except queue.Empty:
                break
        return last

    def fps(self) -> float:
        with self._fps_lock:
            return round(self._fps, 1)

    def _calc_fps(self) -> float:
        now = time.monotonic()
        with self._fps_lock:
            if self._fps_ts > 0:
                dt = now - self._fps_ts
                if dt > 0:
                    inst = 1.0 / dt
                    self._fps = 0.85 * self._fps + 0.15 * inst
            self._fps_ts = now
            return self._fps

    def take_reconnect_msg(self) -> Optional[str]:
        with self._reconnect_msg_lock:
            msg = self._reconnect_msg
            self._reconnect_msg = None
            return msg

    def _infer_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._infer_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break

            self._state.reset_fails()

            learner = globals().get("ONLINE_LEARNER")
            if learner is not None:
                try:
                    learner.remember_frame(item.frame)
                except Exception:
                    pass

            try:
                result = self._engine.infer(
                    item.frame,
                    item.params.prompt,
                    item.params.conf,
                    item.params.iou,
                    item.params.imgsz,
                    item.params.min_area,
                    track=item.params.track,
                    roi=item.params.roi,
                )
            except Exception:
                log.exception("infer loop error")
                continue

            self._frame_id += 1
            self._calc_fps()

            try:
                self._result_q.put_nowait(result)
            except queue.Full:
                try:
                    self._result_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._result_q.put_nowait(result)
                except queue.Full:
                    pass

    def maybe_schedule_reconnect(self) -> None:
        if self._reconnect_thread is not None and self._reconnect_thread.is_alive():
            return
        if not self._state.begin_reconnect():
            return
        self._reconnect_thread = threading.Thread(
            target=self._reconnect_worker,
            name=CFG.pipeline.reconnect_thread_name,
            daemon=True,
        )
        self._reconnect_thread.start()

    def _reconnect_worker(self) -> None:
        try:
            source, source_type, attempt = self._state.snapshot_reconnect_info()
            if source is None or not source_type:
                return
            delay = min(
                CFG.camera.reconnect_delay * (attempt + 1),
                CFG.camera.reconnect_backoff_max,
            )
            log.info("reconnect attempt %d in %.1fs: %s", attempt + 1, delay, source)
            time.sleep(delay)

            self._state.detach()
            time.sleep(CFG.camera.release_settle_s)

            ok, msg = open_source(source_type, str(source))
            if ok:
                n = self._state.bump_reconnect()
                self._set_reconnect_msg(f"🟢 переподключено (попытка {n}): {source}")
            else:
                self._set_reconnect_msg(f"🔴 reconnect failed: {msg}")
                log.warning("reconnect failed: %s", msg)
        except Exception as exc:
            log.exception("reconnect worker crashed: %s", exc)
            self._set_reconnect_msg(f"🔴 reconnect error: {exc}")
        finally:
            self._state.end_reconnect()

    def _set_reconnect_msg(self, msg: str) -> None:
        with self._reconnect_msg_lock:
            self._reconnect_msg = msg


# =============================================================================
# 12. ОНЛАЙН-ОБУЧЕНИЕ
# =============================================================================

@dataclass
class FeedbackSample:
    cls: str
    bbox: tuple[int, int, int, int]
    confidence: float
    frame: np.ndarray
    ts: float
    kind: str


class OnlineLearner:
    def __init__(
        self,
        engine: DetectionEngine,
        cfg: OnlineLearnConfig,
        buffer_dir: Path,
    ) -> None:
        self._engine = engine
        self._cfg = cfg
        self._buffer_dir = Path(buffer_dir)
        self._buffer_dir.mkdir(parents=True, exist_ok=True)
        self._crops_dir = self._buffer_dir / "crops"
        self._crops_dir.mkdir(exist_ok=True)

        self._lock = threading.Lock()

        self._tp: dict[str, int] = defaultdict(int)
        self._fp: dict[str, int] = defaultdict(int)
        self._fn: dict[str, int] = defaultdict(int)

        self._conf_overrides: dict[str, float] = {}

        self._samples: deque[FeedbackSample] = deque(maxlen=cfg.buffer_size)
        self._samples_since_refine = 0
        self._samples_since_recalib = 0

        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_lock = threading.Lock()

    def remember_frame(self, frame_bgr: np.ndarray) -> None:
        with self._last_frame_lock:
            self._last_frame = frame_bgr.copy()

    def get_last_frame(self) -> Optional[np.ndarray]:
        with self._last_frame_lock:
            return None if self._last_frame is None else self._last_frame.copy()

    def submit_feedback(
        self,
        det: Detection,
        kind: str,
        new_cls: Optional[str] = None,
    ) -> str:
        if not self._cfg.enabled:
            return "онлайн-обучение отключено"

        frame = self.get_last_frame()
        if frame is None:
            return "нет последнего кадра — подожди тик"

        x1, y1, x2, y2 = det.bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return "битый bbox"

        crop_bgr = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        cls = new_cls if (kind == "relabel" and new_cls) else det.cls
        sample = FeedbackSample(
            cls=cls,
            bbox=(x1, y1, x2, y2),
            confidence=det.confidence,
            frame=crop_rgb,
            ts=time.monotonic(),
            kind=kind,
        )

        with self._lock:
            self._samples.append(sample)
            self._samples_since_refine += 1
            self._samples_since_recalib += 1

            if kind == "tp":
                self._tp[cls] += 1
            elif kind == "fp":
                self._fp[cls] += 1
                if new_cls and new_cls != det.cls:
                    self._fn[new_cls] += 1

        if self._cfg.save_crops and sample.frame.size >= self._cfg.crop_min_area * 3:
            self._save_crop(sample)

        recalib_msg = ""
        if self._samples_since_recalib >= self._cfg.recalib_every:
            self._recalibrate()
            with self._lock:
                self._samples_since_recalib = 0
            recalib_msg = " · recalibrated"

        refine_msg = ""
        if self._samples_since_refine >= self._cfg.refine_every:
            if self._refine_prompts():
                refine_msg = " · prompts refined"
            with self._lock:
                self._samples_since_refine = 0

        icon = {"tp": "✅", "fp": "❌", "relabel": "🏷"}.get(kind, "·")
        return f"{icon} {cls}{recalib_msg}{refine_msg}"

    def _recalibrate(self) -> None:
        with self._lock:
            classes = set(self._tp) | set(self._fp)
            new_overrides = dict(self._conf_overrides)

            for cls in classes:
                tp = self._tp.get(cls, 0)
                fp = self._fp.get(cls, 0)
                total = tp + fp
                if total < self._cfg.min_examples_per_class:
                    continue

                precision = tp / total
                cur = new_overrides.get(cls, CFG.ui.default_conf)
                if precision < 0.6:
                    new_overrides[cls] = min(self._cfg.conf_max, cur + self._cfg.conf_step)
                elif precision > 0.9 and tp >= 10:
                    new_overrides[cls] = max(self._cfg.conf_min, cur - self._cfg.conf_step)

            self._conf_overrides = new_overrides

        log.info("online calibration: %s", {
            k: round(v, 3) for k, v in self._conf_overrides.items()
        })

    def get_conf_override(self, cls: str) -> Optional[float]:
        with self._lock:
            return self._conf_overrides.get(cls)

    def all_overrides(self) -> dict[str, float]:
        with self._lock:
            return dict(self._conf_overrides)

    def _refine_prompts(self) -> bool:
        try:
            get_img_pe = getattr(self._engine._model, "get_image_pe", None)
            if get_img_pe is None:
                log.debug("get_image_pe not available, skipping refine")
                return False
        except Exception:
            return False

        with self._lock:
            by_class: dict[str, list[FeedbackSample]] = defaultdict(list)
            for s in self._samples:
                if s.kind in ("tp", "relabel"):
                    by_class[s.cls].append(s)

        refined_any = False
        for cls, samples in by_class.items():
            if len(samples) < 5:
                continue
            try:
                chosen = random.sample(samples, min(20, len(samples)))
                img_pe_list = []
                for s in chosen:
                    try:
                        pe = get_img_pe([s.frame])
                        if pe is not None:
                            arr = np.asarray(pe.detach().cpu() if hasattr(pe, "detach") else pe)
                            img_pe_list.append(arr.reshape(1, -1))
                    except Exception as exc:
                        log.debug("get_image_pe failed: %s", exc)
                        continue

                if not img_pe_list:
                    continue

                img_pe = np.concatenate(img_pe_list, axis=0).mean(axis=0, keepdims=True)

                with self._engine.pe_lock:
                    cached = self._engine.pe_cache.get((cls,))
                if cached is None:
                    cached = self._engine._model.get_text_pe([cls])

                cached_np = cached.detach().cpu().numpy() if hasattr(cached, "detach") else np.asarray(cached)
                if cached_np.shape != img_pe.shape:
                    img_pe = np.resize(img_pe, cached_np.shape)

                blended_np = (1 - self._cfg.refine_blend) * cached_np + self._cfg.refine_blend * img_pe
                blended = torch.from_numpy(blended_np).to(self._engine.device).float()

                with self._engine.pe_lock:
                    self._engine.pe_cache[(cls,)] = blended
                refined_any = True
                log.info("online refine: %s blended (%d crops)", cls, len(chosen))
            except Exception as exc:
                log.warning("refine failed for %s: %s", cls, exc)

        return refined_any

    def _save_crop(self, s: FeedbackSample) -> None:
        try:
            safe_cls = "".join(c if c.isalnum() else "_" for c in s.cls)
            cls_dir = self._crops_dir / safe_cls
            cls_dir.mkdir(exist_ok=True)
            name = f"{int(s.ts * 1000)}_{random.randint(0, 9999)}.png"
            cv2.imwrite(str(cls_dir / name), cv2.cvtColor(s.frame, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            log.debug("save_crop failed: %s", exc)

    def export_dataset(self) -> str:
        if not self._crops_dir.exists():
            return "нет сохранённых кропов"
        classes = sorted([d.name for d in self._crops_dir.iterdir() if d.is_dir()])
        counts = {c: len(list((self._crops_dir / c).glob("*.png"))) for c in classes}
        return (
            f"кропы: {self._crops_dir} · "
            f"классов: {len(classes)} · "
            f"всего: {sum(counts.values())} · "
            f"детали: {counts}"
        )

    def stats(self) -> str:
        with self._lock:
            tp = dict(self._tp)
            fp = dict(self._fp)
            n = len(self._samples)
            overrides = dict(self._conf_overrides)
        return json.dumps({
            "samples": n,
            "tp": tp,
            "fp": fp,
            "conf_overrides": {k: round(v, 3) for k, v in overrides.items()},
        }, ensure_ascii=False, indent=2)

    def reset(self) -> str:
        with self._lock:
            self._samples.clear()
            self._tp.clear()
            self._fp.clear()
            self._fn.clear()
            self._conf_overrides.clear()
            self._samples_since_refine = 0
            self._samples_since_recalib = 0
        return "буфер и пороги сброшены"


# =============================================================================
# 13. СИНГЛТОНЫ
# =============================================================================

log.info("init: loading model + recorder + pipeline")
engine = DetectionEngine(CFG.model)
engine.warmup()

state = StateStore(CFG.camera)
recorder = VideoRecorder(output_dir=Path(CFG.server.output_dir))
det_logger = DetectionLogger(Path(CFG.server.output_dir) / "logs")
webhook = WebhookNotifier(CFG.server.webhook_url, CFG.server.webhook_timeout)
webhook.start()

ONLINE_LEARNER: Optional[OnlineLearner] = None
if CFG.online.enabled:
    ONLINE_LEARNER = OnlineLearner(
        engine=engine,
        cfg=CFG.online,
        buffer_dir=Path(CFG.server.output_dir) / CFG.online.buffer_dir,
    )
    log.info("online learner enabled (crops=%s)", ONLINE_LEARNER._crops_dir)

pipeline = Pipeline(engine, state)
pipeline.start()


# =============================================================================
# 14. СЕРВИСНЫЕ ФУНКЦИИ
# =============================================================================

def _to_int_or_none(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_source(source_type: str, url: str) -> tuple[bool, Any, str]:
    url = (url or "").strip()

    if source_type == "Веб-камера":
        if url:
            idx = _to_int_or_none(url)
            if idx is None:
                return False, None, f"индекс вебки должен быть числом, а не {url!r}"
            if not (0 <= idx <= 100):
                return False, None, f"индекс вебки вне диапазона 0..100: {idx}"
            return True, idx, ""
        return True, 0, ""

    if not url:
        return False, None, "укажите URL или путь"

    p = Path(url)
    if p.exists():
        return True, str(p), ""

    ok, err = validate_source_url(url)
    if not ok:
        return False, None, err

    return True, url, ""


def open_source(source_type: str, url: str) -> tuple[bool, str]:
    ok, source, msg = _build_source(source_type, url)
    if not ok:
        return False, msg

    try:
        reader = CameraReader(source, CFG.camera)
        if not reader.ok:
            reader.stop()
            return False, f"не открылось: {source}"
        reader.start()
        state.attach(reader, str(source), source=source, source_type=source_type)
        return True, f"подключено: {source}"
    except Exception as exc:
        log.exception("open_source failed")
        return False, f"ошибка подключения: {exc}"


def close_source() -> tuple[bool, str]:
    state.detach()
    recorder.stop()
    det_logger.stop()
    return True, "отключено"


def process_image(image, prompt, conf, iou, imgsz, min_area):
    if image is None:
        return None, None, "нет картинки", "{}"

    try:
        res = engine.infer(image, prompt, conf, iou, imgsz, min_area, track=False)
    except Exception as exc:
        log.exception("process_image failed")
        return image, image, f"ошибка: {exc}", "{}"

    payload = {
        "timestamp": datetime.now().isoformat(),
        "count": len(res.detections),
        "infer_ms": round(res.infer_ms, 1),
        "imgsz": res.imgsz_used,
        "objects": [d.to_dict() for d in res.detections],
    }
    return image, res.annotated, res.status, json.dumps(payload, ensure_ascii=False, indent=2)


def process_stream(prompt, conf, iou, imgsz, min_area, show_raw, track,
                   roi_points_json: str):
    reader = state.reader
    recon_msg = pipeline.take_reconnect_msg()

    if reader is None or not reader.running:
        return (
            gr.update(), gr.update(),
            recon_msg or "источник не подключён",
            gr.update(), gr.update(), gr.update(), gr.update(),
        )

    roi = _parse_roi(roi_points_json)

    pipeline.set_params(
        prompt=prompt, conf=conf, iou=iou, imgsz=int(imgsz),
        min_area=int(min_area), track=bool(track), show_raw=bool(show_raw),
        roi=roi,
    )

    frame = reader.read()
    if frame is None:
        fails = state.register_fail()
        if state.should_reconnect():
            pipeline.maybe_schedule_reconnect()
        status = recon_msg or f"нет кадра ({fails}/{CFG.camera.max_fails})"
        return (
            gr.update(), gr.update(), status,
            gr.update(), gr.update(), gr.update(), gr.update(),
        )

    pipeline.submit(frame)

    res = pipeline.latest()
    if res is None:
        return (
            gr.update(), gr.update(),
            recon_msg or "ожидание первого кадра…",
            gr.update(), gr.update(), gr.update(), gr.update(),
        )

    fps = pipeline.fps()
    if recorder.active:
        recorder.write(res.annotated)
    if det_logger.active:
        det_logger.log(reader_id(), res.detections, fps)

    if webhook.enabled and res.detections:
        classes = sorted({d.cls for d in res.detections})
        webhook.notify({
            "timestamp": datetime.now().isoformat(),
            "count": len(res.detections),
            "classes": classes,
            "fps": fps,
        })

    raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if show_raw else None

    payload_objects = []
    for i, d in enumerate(res.detections):
        item = d.to_dict()
        item["index"] = i
        payload_objects.append(item)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "count": len(res.detections),
        "fps": fps,
        "infer_ms": round(res.infer_ms, 1),
        "imgsz": res.imgsz_used,
        "metrics": METRICS.snapshot(),
        "conf_overrides": ONLINE_LEARNER.all_overrides() if ONLINE_LEARNER else {},
        "objects": payload_objects,
    }
    classes = ", ".join(sorted({d.cls for d in res.detections})) or "—"

    status_line = res.status
    if recon_msg:
        status_line = f"{recon_msg} | {status_line}"

    return (
        raw_rgb if raw_rgb is not None else gr.update(),
        res.annotated,
        status_line,
        json.dumps(payload, ensure_ascii=False, indent=2),
        len(res.detections),
        classes,
        fps,
    )


_frame_id_lock = threading.Lock()
_frame_counter = 0


def reader_id() -> int:
    global _frame_counter
    with _frame_id_lock:
        _frame_counter += 1
        return _frame_counter


def _parse_roi(roi_json: str) -> Optional[np.ndarray]:
    if not roi_json:
        return None
    try:
        pts = json.loads(roi_json)
        if not pts or len(pts) < 3:
            return None
        return np.array([[int(p[0]), int(p[1])] for p in pts], dtype=np.int32)
    except Exception:
        return None


def _fb_dispatch(feedback_json_str: str, kind: str, new_cls: Optional[str]) -> str:
    if ONLINE_LEARNER is None:
        return "онлайн-обучение отключено"
    try:
        data = json.loads(feedback_json_str)
        det = Detection(
            obj_id=data.get("id"),
            cls=data["class"],
            confidence=float(data["confidence"]),
            bbox=tuple(data["bbox"]),
            area=int(data["area"]),
        )
    except Exception as exc:
        return f"некорректный JSON: {exc}"
    return ONLINE_LEARNER.submit_feedback(det, kind, new_cls or None)


def _pick_first_detection() -> str:
    res = pipeline.latest()
    if res is None or not res.detections:
        return ""
    d = res.detections[0]
    return json.dumps(d.to_dict(), ensure_ascii=False)


# =============================================================================
# 15. HEALTH-CHECK
# =============================================================================

def health_check() -> str:
    return json.dumps({
        "status": "ok",
        "version": __version__,
        "device": engine.device,
        "fp16": engine._fp16_enabled,
        "imgsz_current": engine.imgsz,
        "camera_running": state.running,
        "reconnecting": state.reconnecting,
        "recording": recorder.active,
        "logging": det_logger.active,
        "online_learning": ONLINE_LEARNER is not None,
        "metrics": METRICS.snapshot(),
        "fps": pipeline.fps(),
    }, ensure_ascii=False)


# =============================================================================
# 16. ЗАВЕРШЕНИЕ
# =============================================================================

_shutdown_lock = threading.Lock()
_shutdown_done = False


def _shutdown(*_):
    global _shutdown_done
    with _shutdown_lock:
        if _shutdown_done:
            return
        _shutdown_done = True

    log.info("shutting down...")
    for name, fn in (
        ("pipeline", pipeline.stop),
        ("state", state.detach),
        ("recorder", recorder.release),
        ("det_logger", det_logger.release),
        ("webhook", webhook.stop),
    ):
        try:
            fn()
        except Exception as exc:
            log.warning("%s shutdown failed: %s", name, exc)
    log.info("cleanup done")


atexit.register(_shutdown)


def _sig_handler(signum, _frame):
    log.info("signal %s received", signum)
    _shutdown()
    sys.exit(0)


for sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(sig, _sig_handler)
    except (ValueError, OSError):
        pass


# =============================================================================
# 17. UI — СОВРЕМЕННЫЙ IT-СТИЛЬ
# =============================================================================

CUSTOM_CSS = """
/* ============ BASE ============ */
:root {
  --bg: #0a0b10;
  --bg-1: #0f1117;
  --bg-2: #14161f;
  --bg-3: #1a1d28;
  --border: #232633;
  --border-hi: #2d3142;
  --fg: #e4e6eb;
  --fg-dim: #8b91a3;
  --fg-mute: #5a5f6f;
  --accent: #6366f1;
  --accent-2: #8b5cf6;
  --accent-glow: rgba(99, 102, 241, 0.35);
  --ok: #10b981;
  --warn: #f59e0b;
  --err: #ef4444;
  --mono: ui-monospace, "JetBrains Mono", "SF Mono", Menlo, monospace;
}

* { box-sizing: border-box; }

body, .gradio-container {
  background: var(--bg) !important;
  color: var(--fg) !important;
  font-family: Inter, system-ui, -apple-system, sans-serif !important;
}

.gradio-container {
  max-width: 1800px !important;
  margin: 0 auto !important;
  padding: 0 !important;
}

footer { display: none !important; }

/* ============ HERO ============ */
.hero {
  background:
    radial-gradient(1200px 400px at 0% 0%, rgba(99, 102, 241, 0.18), transparent 60%),
    radial-gradient(900px 400px at 100% 100%, rgba(139, 92, 246, 0.14), transparent 60%),
    linear-gradient(180deg, #12131c 0%, #0d0e15 100%);
  padding: 28px 36px;
  border-bottom: 1px solid var(--border);
  position: relative;
  overflow: hidden;
}
.hero::after {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
  background-size: 32px 32px;
  pointer-events: none;
  mask-image: linear-gradient(180deg, #000 0%, transparent 90%);
}
.hero h1 {
  margin: 0 0 8px 0;
  font-size: 24px;
  font-weight: 700;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, #a5b4fc 0%, #c4b5fd 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero p {
  margin: 0;
  color: var(--fg-dim);
  font-size: 13px;
  font-family: var(--mono);
  letter-spacing: -0.01em;
}
.hero-badges {
  display: flex;
  gap: 8px;
  margin-top: 16px;
  flex-wrap: wrap;
  position: relative;
  z-index: 1;
}
.badge {
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border-hi);
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 11px;
  font-family: var(--mono);
  color: var(--fg-dim);
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.badge.live { border-color: rgba(16, 185, 129, 0.5); color: #6ee7b7; }
.badge.live::before {
  content: "";
  width: 6px; height: 6px;
  border-radius: 50%;
  background: #10b981;
  box-shadow: 0 0 8px #10b981;
  animation: pulse 1.6s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.35; }
}

/* ============ PANELS ============ */
.panel {
  background: var(--bg-1);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px;
  position: relative;
}
.panel-title {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--fg-mute);
  margin: 0 0 14px 0;
  display: flex;
  align-items: center;
  gap: 8px;
}
.panel-title::before {
  content: "";
  width: 3px; height: 10px;
  border-radius: 2px;
  background: linear-gradient(180deg, var(--accent), var(--accent-2));
}

/* ============ KBD / HOTKEYS ============ */
kbd {
  background: linear-gradient(180deg, #1e2030 0%, #15171f 100%);
  border: 1px solid #2d3142;
  border-bottom-width: 2px;
  border-radius: 5px;
  padding: 1px 6px;
  font-family: var(--mono);
  font-size: 11px;
  color: #c7cbe0;
  min-width: 22px;
  text-align: center;
  display: inline-block;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
}

.hotkeys-card {
  background: var(--bg-1);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 18px;
  position: sticky;
  top: 12px;
}
.hotkeys-card h3 {
  margin: 0 0 4px 0;
  font-size: 13px;
  color: #a5b4fc;
  display: flex;
  align-items: center;
  gap: 8px;
}
.hotkeys-card .sub {
  font-size: 11px;
  color: var(--fg-mute);
  margin-bottom: 14px;
  font-family: var(--mono);
}
.hotkeys-section {
  font-size: 9px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--fg-mute);
  margin: 14px 0 8px 0;
}
.hotkeys-list { display: flex; flex-direction: column; gap: 5px; }
.hotkey-row {
  display: grid;
  grid-template-columns: 46px 1fr;
  align-items: center;
  gap: 10px;
  padding: 5px 8px;
  border-radius: 6px;
  font-size: 12px;
  color: var(--fg-dim);
  transition: background .12s ease, color .12s ease;
}
.hotkey-row:hover {
  background: rgba(99, 102, 241, 0.08);
  color: var(--fg);
}
.hotkey-row kbd { justify-self: start; }

/* ============ STATUS BAR ============ */
.status-bar {
  position: sticky;
  bottom: 0;
  background: linear-gradient(180deg, rgba(10,11,16,0.9) 0%, rgba(10,11,16,1) 100%);
  backdrop-filter: blur(12px);
  border-top: 1px solid var(--border);
  padding: 8px 20px;
  display: flex;
  align-items: center;
  gap: 20px;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--fg-dim);
  z-index: 100;
  margin-top: 16px;
}
.status-bar .sb-item {
  display: flex;
  align-items: center;
  gap: 6px;
}
.status-bar .sb-item .k { color: var(--fg-mute); }
.status-bar .sb-item .v { color: var(--fg); font-weight: 600; }
.status-bar .sb-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--err);
}
.status-bar .sb-dot.ok { background: var(--ok); box-shadow: 0 0 8px var(--ok); }
.status-bar .sb-dot.warn { background: var(--warn); box-shadow: 0 0 8px var(--warn); }

/* ============ METRICS DASHBOARD ============ */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 10px;
  margin: 12px 0;
}
.metric-card {
  background: var(--bg-1);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 14px;
  position: relative;
  overflow: hidden;
}
.metric-card::before {
  content: "";
  position: absolute;
  top: 0; left: 0;
  width: 2px; height: 100%;
  background: linear-gradient(180deg, var(--accent), transparent);
  opacity: 0.6;
}
.metric-card .label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--fg-mute);
  margin-bottom: 4px;
  font-weight: 600;
}
.metric-card .value {
  font-family: var(--mono);
  font-size: 22px;
  font-weight: 700;
  color: var(--fg);
  letter-spacing: -0.02em;
  line-height: 1.1;
}
.metric-card .value.accent {
  background: linear-gradient(135deg, #a5b4fc, #c4b5fd);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.metric-card .unit {
  font-size: 11px;
  color: var(--fg-mute);
  font-family: var(--mono);
  margin-left: 3px;
}

/* ============ GRADIO OVERRIDES ============ */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel {
  background: var(--bg-1) !important;
  border-color: var(--border) !important;
}
.gradio-container label,
.gradio-container .label-wrap > span {
  color: var(--fg-dim) !important;
  font-size: 11px !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 600 !important;
}
.gradio-container input[type=text],
.gradio-container input[type=number],
.gradio-container textarea,
.gradio-container .wrap input {
  background: var(--bg-2) !important;
  border: 1px solid var(--border) !important;
  color: var(--fg) !important;
  border-radius: 8px !important;
  font-family: var(--mono) !important;
  font-size: 12px !important;
  transition: border-color .15s ease;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important;
  outline: none !important;
}

.gradio-container button {
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 12px !important;
  transition: all .15s ease !important;
  border: 1px solid var(--border) !important;
  background: var(--bg-2) !important;
  color: var(--fg) !important;
}
.gradio-container button:hover {
  border-color: var(--accent) !important;
  background: var(--bg-3) !important;
  transform: translateY(-1px);
}
.gradio-container button.primary,
.gradio-container button[variant=primary],
.gradio-container .primary button {
  background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
  border: none !important;
  color: #fff !important;
  box-shadow: 0 4px 14px -4px var(--accent-glow);
}
.gradio-container button.primary:hover,
.gradio-container .primary button:hover {
  box-shadow: 0 6px 20px -4px var(--accent-glow);
  filter: brightness(1.08);
}

.image-frame img,
.gradio-container .image-container img {
  border-radius: 10px !important;
  border: 1px solid var(--border) !important;
  background: #000 !important;
}

.gradio-container .accordion {
  background: var(--bg-1) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  overflow: hidden;
}
.gradio-container .accordion > .label-wrap {
  background: var(--bg-1) !important;
  padding: 10px 14px !important;
}
.gradio-container .accordion > .label-wrap:hover {
  background: var(--bg-2) !important;
}

.gradio-container input[type=range] { accent-color: var(--accent); }

.gradio-container pre,
.gradio-container code,
.gradio-container .cm-editor {
  background: var(--bg-2) !important;
  color: #c7cbe0 !important;
  font-family: var(--mono) !important;
  font-size: 11px !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

.gradio-container input[type=checkbox] { accent-color: var(--accent); }

/* ============ TOAST ============ */
.hotkey-toast {
  position: fixed;
  top: 24px;
  right: 24px;
  background: linear-gradient(135deg, rgba(99,102,241,0.95), rgba(139,92,246,0.95));
  backdrop-filter: blur(20px);
  color: #fff;
  padding: 11px 18px;
  border-radius: 10px;
  font-size: 12px;
  font-weight: 600;
  font-family: var(--mono);
  box-shadow: 0 10px 40px -8px rgba(99,102,241,0.7),
              inset 0 1px 0 rgba(255,255,255,0.2);
  z-index: 99999;
  opacity: 0;
  transform: translateY(-12px) scale(0.96);
  transition: opacity .18s cubic-bezier(.2,.9,.3,1.2),
              transform .18s cubic-bezier(.2,.9,.3,1.2);
  pointer-events: none;
  border: 1px solid rgba(255,255,255,0.15);
}
.hotkey-toast.show {
  opacity: 1;
  transform: translateY(0) scale(1);
}

/* ============ MODAL HELP ============ */
.hotkey-help {
  position: fixed;
  top: 50%; left: 50%;
  transform: translate(-50%, -50%) scale(0.96);
  background: rgba(15, 17, 23, 0.98);
  backdrop-filter: blur(24px);
  color: var(--fg);
  padding: 28px 34px;
  border-radius: 16px;
  border: 1px solid var(--border-hi);
  box-shadow: 0 25px 80px -10px rgba(0,0,0,0.9),
              inset 0 1px 0 rgba(255,255,255,0.05);
  z-index: 99999;
  font-family: Inter, system-ui, sans-serif;
  font-size: 13px;
  min-width: 380px;
  display: none;
  opacity: 0;
  transition: opacity .18s ease, transform .18s ease;
}
.hotkey-help.show {
  display: block;
  opacity: 1;
  transform: translate(-50%, -50%) scale(1);
}
.hotkey-help::before {
  content: "";
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.6);
  backdrop-filter: blur(4px);
  z-index: -1;
}
.hotkey-help h3 {
  margin: 0 0 18px 0;
  font-size: 15px;
  color: #a5b4fc;
  display: flex;
  align-items: center;
  gap: 8px;
}
.hotkey-help table { width: 100%; border-collapse: collapse; }
.hotkey-help td {
  padding: 6px 10px;
  border-bottom: 1px solid rgba(255,255,255,0.03);
}
.hotkey-help td:first-child { text-align: right; width: 80px; }
.hotkey-help td:last-child { color: var(--fg-dim); font-size: 12px; }
.hotkey-help .hint {
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid var(--border);
  color: var(--fg-mute);
  font-size: 11px;
  text-align: center;
  font-family: var(--mono);
}

/* ============ MISC ============ */
.section-title {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--fg-mute);
  margin: 0 0 12px 0;
  display: flex;
  align-items: center;
  gap: 8px;
}
.section-title::before {
  content: "";
  width: 3px; height: 10px;
  border-radius: 2px;
  background: linear-gradient(180deg, var(--accent), var(--accent-2));
}

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb {
  background: var(--border-hi);
  border-radius: 5px;
  border: 2px solid var(--bg);
}
::-webkit-scrollbar-thumb:hover { background: #3d4154; }
"""


HOTKEYS_JS = r"""
function() {
  const map = {
    ' ': { action: 'toggle_connect', label: 'connect / disconnect' },
    'r': { action: 'toggle_record',  label: 'record toggle' },
    'l': { action: 'toggle_export',  label: 'export toggle' },
    'c': { action: 'pick_detection', label: 'pick detection' },
    '1': { action: 'fb_tp',          label: 'marked as ✓' },
    '2': { action: 'fb_fp',          label: 'marked as ✗' },
    '3': { action: 'fb_relabel',     label: 'relabel' },
    'p': { action: 'preset_people',  label: 'preset: people' },
    't': { action: 'preset_transport', label: 'preset: transport' },
    'a': { action: 'preset_animals', label: 'preset: animals' },
    'y': { action: 'toggle_tracking', label: 'tracking' },
    's': { action: 'snapshot',       label: 'snapshot saved' },
    'h': { action: 'health',         label: 'health-check' },
    '?': { action: 'help',           label: '' },
  };

  function toast(text) {
    if (!text) return;
    let el = document.querySelector('.hotkey-toast');
    if (!el) {
      el = document.createElement('div');
      el.className = 'hotkey-toast';
      document.body.appendChild(el);
    }
    el.innerHTML = '⌨️ &nbsp;' + text;
    el.classList.add('show');
    clearTimeout(el._t);
    el._t = setTimeout(() => el.classList.remove('show'), 1400);
  }

  function ensureHelp() {
    let el = document.querySelector('.hotkey-help');
    if (!el) {
      el = document.createElement('div');
      el.className = 'hotkey-help';
      el.innerHTML = `
        <h3>⌨️ Горячие клавиши</h3>
        <table>
          <tr><td><kbd>Space</kbd></td><td>подключить / отключить источник</td></tr>
          <tr><td><kbd>R</kbd></td><td>старт / стоп записи</td></tr>
          <tr><td><kbd>L</kbd></td><td>старт / стоп экспорта JSONL+CSV</td></tr>
          <tr><td><kbd>C</kbd></td><td>взять первую детекцию</td></tr>
          <tr><td><kbd>1</kbd></td><td>отметить детекцию как верную</td></tr>
          <tr><td><kbd>2</kbd></td><td>отметить как мусор</td></tr>
          <tr><td><kbd>3</kbd></td><td>переклассифицировать</td></tr>
          <tr><td><kbd>P</kbd></td><td>пресет «люди»</td></tr>
          <tr><td><kbd>T</kbd></td><td>пресет «транспорт»</td></tr>
          <tr><td><kbd>A</kbd></td><td>пресет «животные»</td></tr>
          <tr><td><kbd>Y</kbd></td><td>трекинг вкл/выкл</td></tr>
          <tr><td><kbd>S</kbd></td><td>скачать текущий кадр</td></tr>
          <tr><td><kbd>H</kbd></td><td>обновить health-check</td></tr>
          <tr><td><kbd>?</kbd></td><td>показать / скрыть эту справку</td></tr>
        </table>
        <div class="hint">нажми Esc чтобы закрыть · не работает внутри полей ввода</div>
      `;
      document.body.appendChild(el);
    }
    el.classList.toggle('show');
  }

  function clickButtonByText() {
    const needles = Array.from(arguments).map(s => s.toLowerCase());
    const nodes = Array.from(document.querySelectorAll('button'));
    for (const b of nodes) {
      const t = (b.textContent || '').trim().toLowerCase();
      for (const needle of needles) {
        if (t.includes(needle)) { b.click(); return true; }
      }
    }
    return false;
  }

  function toggleCheckboxByLabel(labelSubstr) {
    const labels = Array.from(document.querySelectorAll('label'));
    for (const lab of labels) {
      if ((lab.textContent || '').toLowerCase().includes(labelSubstr.toLowerCase())) {
        const cb = lab.querySelector('input[type=checkbox]')
          || (lab.closest('.block') && lab.closest('.block').querySelector('input[type=checkbox]'));
        if (cb) { cb.click(); return true; }
      }
    }
    return false;
  }

  function downloadSnapshot() {
    const imgs = Array.from(document.querySelectorAll('.image-frame img, .image-container img'));
    const target = imgs[imgs.length - 1];
    if (!target || !target.src) { toast('нет кадра'); return; }
    const a = document.createElement('a');
    a.href = target.src;
    a.download = 'yoloe_snapshot_' + Date.now() + '.png';
    a.click();
    toast('snapshot saved');
  }

  function handler(ev) {
    const t = ev.target;
    const tag = (t && t.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || (t && t.isContentEditable)) {
      if (ev.key === 'Escape') {
        const help = document.querySelector('.hotkey-help');
        if (help) help.classList.remove('show');
      }
      return;
    }
    if (ev.ctrlKey || ev.altKey || ev.metaKey) return;

    const key = ev.key;
    const cfg = map[key] || map[key.toLowerCase()];
    if (!cfg) return;

    ev.preventDefault();
    const { action, label } = cfg;

    switch (action) {
      case 'toggle_connect':   clickButtonByText('подключить') || clickButtonByText('отключить'); break;
      case 'toggle_record':    clickButtonByText('старт') || clickButtonByText('запись'); break;
      case 'toggle_export':    clickButtonByText('jsonl', 'csv'); break;
      case 'pick_detection':   clickButtonByText('взять первую'); break;
      case 'fb_tp':            clickButtonByText('верно'); break;
      case 'fb_fp':            clickButtonByText('мусор'); break;
      case 'fb_relabel':       clickButtonByText('переклассифицировать'); break;
      case 'preset_people':    clickButtonByText('люди'); break;
      case 'preset_transport': clickButtonByText('транспорт'); break;
      case 'preset_animals':   clickButtonByText('животные'); break;
      case 'toggle_tracking':  toggleCheckboxByLabel('трекинг'); break;
      case 'snapshot':         downloadSnapshot(); return;
      case 'health':           clickButtonByText('обновить'); toast('health-check'); return;
      case 'help':             ensureHelp(); return;
      default: return;
    }
    toast(label);
  }

  document.addEventListener('keydown', handler);
  console.log('[YOLOE] hotkeys installed');
}
"""


def _device_badge() -> str:
    dev = engine.device.upper()
    icon = "⚡" if dev == "CUDA" else "🖥"
    fp16 = " · fp16" if engine._fp16_enabled else ""
    return f"{icon} {dev}{fp16}"


def _hotkeys_sidebar_html() -> str:
    return """
    <div class="hotkeys-card">
      <h3>⌨️ Горячие клавиши</h3>
      <div class="sub">не работают внутри полей ввода</div>

      <div class="hotkeys-section">Источник</div>
      <div class="hotkeys-list">
        <div class="hotkey-row"><kbd>Space</kbd><span>подключить / отключить</span></div>
      </div>

      <div class="hotkeys-section">Запись</div>
      <div class="hotkeys-list">
        <div class="hotkey-row"><kbd>R</kbd><span>старт / стоп видео</span></div>
        <div class="hotkey-row"><kbd>L</kbd><span>старт / стоп JSONL+CSV</span></div>
        <div class="hotkey-row"><kbd>S</kbd><span>скачать кадр</span></div>
      </div>

      <div class="hotkeys-section">Онлайн-обучение</div>
      <div class="hotkeys-list">
        <div class="hotkey-row"><kbd>C</kbd><span>взять первую детекцию</span></div>
        <div class="hotkey-row"><kbd>1</kbd><span>отметить ✓ верно</span></div>
        <div class="hotkey-row"><kbd>2</kbd><span>отметить ✗ мусор</span></div>
        <div class="hotkey-row"><kbd>3</kbd><span>переклассифицировать</span></div>
      </div>

      <div class="hotkeys-section">Пресеты</div>
      <div class="hotkeys-list">
        <div class="hotkey-row"><kbd>P</kbd><span>люди</span></div>
        <div class="hotkey-row"><kbd>T</kbd><span>транспорт</span></div>
        <div class="hotkey-row"><kbd>A</kbd><span>животные</span></div>
        <div class="hotkey-row"><kbd>Y</kbd><span>трекинг вкл/выкл</span></div>
      </div>

      <div class="hotkeys-section">Прочее</div>
      <div class="hotkeys-list">
        <div class="hotkey-row"><kbd>H</kbd><span>health-check</span></div>
        <div class="hotkey-row"><kbd>?</kbd><span>справка</span></div>
        <div class="hotkey-row"><kbd>Esc</kbd><span>закрыть справку</span></div>
      </div>
    </div>
    """


def build_ui() -> gr.Blocks:
    ui = CFG.ui

    with gr.Blocks(
        title="YOLOE-26 — Detection & Segmentation",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.violet,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        ).set(
            body_background_fill="#0a0b10",
            body_background_fill_dark="#0a0b10",
            body_text_color="#e4e6eb",
            body_text_color_dark="#e4e6eb",
            background_fill_primary="#0f1117",
            background_fill_primary_dark="#0f1117",
            background_fill_secondary="#14161f",
            background_fill_secondary_dark="#14161f",
            border_color_primary="#232633",
            border_color_primary_dark="#232633",
            block_background_fill="#0f1117",
            block_background_fill_dark="#0f1117",
            block_border_color="#232633",
            block_border_color_dark="#232633",
            block_label_background_fill="#0a0b10",
            block_label_background_fill_dark="#0a0b10",
            block_label_text_color="#8b91a3",
            block_label_text_color_dark="#8b91a3",
            block_title_text_color="#e4e6eb",
            block_title_text_color_dark="#e4e6eb",
            input_background_fill="#14161f",
            input_background_fill_dark="#14161f",
            input_border_color="#232633",
            input_border_color_dark="#232633",
            input_border_color_focus="#6366f1",
            input_border_color_focus_dark="#6366f1",
            button_primary_background_fill="linear-gradient(135deg, #6366f1, #8b5cf6)",
            button_primary_background_fill_dark="linear-gradient(135deg, #6366f1, #8b5cf6)",
            button_primary_background_fill_hover="linear-gradient(135deg, #7c7ff5, #9f75f8)",
            button_primary_background_fill_hover_dark="linear-gradient(135deg, #7c7ff5, #9f75f8)",
            button_primary_text_color="#ffffff",
            button_primary_text_color_dark="#ffffff",
            button_secondary_background_fill="#14161f",
            button_secondary_background_fill_dark="#14161f",
            button_secondary_text_color="#e4e6eb",
            button_secondary_text_color_dark="#e4e6eb",
            button_secondary_border_color="#232633",
            button_secondary_border_color_dark="#232633",
        ),
        css=CUSTOM_CSS,
        js=HOTKEYS_JS,
        fill_height=True,
    ) as demo:

        # ============ HERO ============
        gr.HTML(f"""
        <div class="hero">
          <h1>YOLOE-26 · Detection &amp; Segmentation</h1>
          <p>$ yoloe run --source=ip-camera --mode=realtime --online-learning=true</p>
          <div class="hero-badges">
            <span class="badge">📦 {CFG.model.weights}</span>
            <span class="badge">{_device_badge()}</span>
            <span class="badge">🧠 open-vocab</span>
            <span class="badge">🎯 bytetrack</span>
            <span class="badge">⚙️ async-pipeline</span>
            <span class="badge">🔄 online-learning</span>
            <span class="badge">⌨️ нажми <kbd>?</kbd></span>
            <span class="badge">v{__version__}</span>
          </div>
        </div>
        """)

        # ============ MAIN GRID: контент + sidebar ============
        with gr.Row(equal_height=False):

            # ---------- ЛЕВАЯ ЧАСТЬ ----------
            with gr.Column(scale=5, min_width=600):

                with gr.Row(equal_height=True):
                    raw_video = gr.Image(
                        label="SOURCE · сырое видео", type="numpy", interactive=False,
                        height=ui.image_height, elem_classes="image-frame",
                        show_download_button=False,
                    )
                    output_image = gr.Image(
                        label="OUTPUT · результат YOLOE", type="numpy", interactive=False,
                        height=ui.image_height, elem_classes="image-frame",
                        show_download_button=True,
                    )

                # метрики-дашборд (значения обновляются через невидимые поля ниже)
                metrics_html = gr.HTML(value=f"""
                <div class="metrics-grid">
                  <div class="metric-card">
                    <div class="label">Objects</div>
                    <div class="value accent" id="m-obj">0</div>
                  </div>
                  <div class="metric-card">
                    <div class="label">Classes</div>
                    <div class="value" id="m-cls" style="font-size:13px;padding-top:6px;">—</div>
                  </div>
                  <div class="metric-card">
                    <div class="label">FPS</div>
                    <div class="value" id="m-fps">0<span class="unit">/s</span></div>
                  </div>
                  <div class="metric-card">
                    <div class="label">Device</div>
                    <div class="value" style="font-size:14px;padding-top:5px;">{engine.device.upper()}</div>
                  </div>
                </div>
                <div class="metric-card" style="margin-top:0;">
                  <div class="label">Status</div>
                  <div class="value" style="font-size:12px;font-weight:500;padding-top:2px;" id="m-status">ожидание</div>
                </div>
                """)

                # невидимые поля — источник данных
                with gr.Row(visible=False):
                    metric_objects = gr.Number(label="_obj", value=0)
                    metric_classes = gr.Textbox(label="_cls", value="—")
                    metric_fps = gr.Number(label="_fps", value=0.0)
                    metric_status = gr.Textbox(label="_status", value="ожидание")

                # ---------- ПАРАМЕТРЫ ----------
                with gr.Accordion("⚙️ Параметры детекции", open=True):
                    prompt_input = gr.Textbox(
                        label="подсказки через запятую",
                        value=", ".join(CFG.model.default_prompt),
                        placeholder="person, car, bus, dog, ...",
                        lines=1,
                    )
                    with gr.Row():
                        preset_people = gr.Button("👤 люди", size="sm")
                        preset_transport = gr.Button("🚗 транспорт", size="sm")
                        preset_animals = gr.Button("🐾 животные", size="sm")
                        preset_all = gr.Button("✨ всё", size="sm")

                    with gr.Row():
                        conf_slider = gr.Slider(0.0, 1.0, ui.default_conf, 0.01, label="confidence")
                        iou_slider = gr.Slider(0.1, 1.0, ui.default_iou, 0.05, label="IoU (NMS)")

                    with gr.Row():
                        imgsz_slider = gr.Slider(320, 1280, ui.default_imgsz, 32, label="imgsz")
                        min_area_slider = gr.Slider(0, 50_000, ui.default_min_area, 100, label="min area px²")

                    tracking_checkbox = gr.Checkbox(
                        label="🎯 трекинг (ByteTrack · ID между кадрами)", value=False,
                    )

                # ---------- ROI ----------
                with gr.Accordion("🎯 ROI-зона", open=False):
                    roi_input = gr.Textbox(
                        label="точки JSON [[x,y],[x,y],...]",
                        value="",
                        placeholder='[[100,100],[500,100],[500,400],[100,400]]',
                        lines=2,
                    )
                    gr.Markdown(
                        "Пусто — вся сцена. Координаты в пикселях исходного кадра. "
                        "Учитываются объекты, центр которых внутри полигона."
                    )

                # ---------- ИСТОЧНИК ----------
                with gr.Accordion("🎥 Источник", open=True):
                    source_type = gr.Radio(
                        ["IP-камера", "Веб-камера", "Видеофайл"],
                        value="IP-камера", label="тип источника",
                    )
                    camera_url = gr.Textbox(
                        label="URL / путь / индекс",
                        placeholder="http://IP:8080/video  |  rtsp://...  |  0  |  /path/to/video.mp4",
                    )
                    with gr.Row():
                        connect_btn = gr.Button("🔌 подключить", variant="primary", scale=2)
                        disconnect_btn = gr.Button("⛔ отключить", scale=1)

                    camera_status = gr.Textbox(
                        label="состояние", value="не подключено",
                        interactive=False, lines=1,
                    )
                    with gr.Row():
                        show_raw_checkbox = gr.Checkbox(label="показывать сырое", value=True)
                        refresh_slider = gr.Slider(0.1, 2.0, ui.default_timer, 0.1, label="частота, сек")

                # ---------- ЗАПИСЬ / ЭКСПОРТ ----------
                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div class="panel-title">📹 Запись</div>')
                        with gr.Row():
                            rec_path = gr.Textbox(label="файл", value="output.mp4", scale=2)
                            rec_start_btn = gr.Button("▶", variant="primary", scale=0, min_width=44)
                            rec_stop_btn = gr.Button("⏹", scale=0, min_width=44)
                        rec_status = gr.Textbox(
                            label="статус", value="не активна",
                            interactive=False, lines=1,
                        )

                    with gr.Column():
                        gr.HTML('<div class="panel-title">📤 Экспорт детекций</div>')
                        with gr.Row():
                            log_start_btn = gr.Button("▶ JSONL+CSV", variant="primary", scale=1)
                            log_stop_btn = gr.Button("⏹ стоп", scale=1)
                        log_status = gr.Textbox(
                            label="статус", value="не активен",
                            interactive=False, lines=1,
                        )

                # ---------- ЗАГРУЗКА КАРТИНКИ ----------
                with gr.Accordion("🖼 Обработка картинки", open=False):
                    input_image = gr.Image(
                        label="загрузить изображение", type="numpy",
                        sources=["upload", "clipboard"], height=220,
                        elem_classes="image-frame",
                    )
                    process_img_btn = gr.Button("🖼 обработать", variant="primary")
                    with gr.Accordion("📄 JSON с детекциями", open=False):
                        json_output = gr.Code(label="результат", language="json", lines=12, interactive=False)

                # ---------- ОНЛАЙН-ОБУЧЕНИЕ ----------
                with gr.Accordion("🧠 Онлайн-обучение", open=False):
                    gr.Markdown(
                        "**L1 · Per-class пороги** — адаптируется по обратной связи ✅/❌  \n"
                        "**L2 · Few-shot refinement** — подмешивает «хорошие» кропы в embedding  \n"
                        "**L3 · Экспорт кропов** — для ручного дообучения (YOLO classification)"
                    )

                    feedback_json = gr.Textbox(
                        label="выбранная детекция (JSON)",
                        placeholder='{"id": null, "class": "car", "confidence": 0.7, "bbox": [x1,y1,x2,y2], "area": 1200}',
                        lines=2,
                    )
                    with gr.Row():
                        pick_btn = gr.Button("📋 взять первую", size="sm")
                        fb_tp = gr.Button("✅ верно", variant="primary", size="sm")
                        fb_fp = gr.Button("❌ мусор", size="sm")

                    with gr.Row():
                        relabel_cls = gr.Textbox(label="класс", value="", scale=2)
                        fb_relabel = gr.Button("🏷 переклассифицировать", size="sm", scale=1)

                    fb_status = gr.Textbox(label="ответ", interactive=False, lines=1)

                    with gr.Row():
                        ol_stats_btn = gr.Button("📊 статистика", size="sm")
                        ol_reset_btn = gr.Button("🗑 сбросить", size="sm")
                        ol_export_btn = gr.Button("📤 куда кропы", size="sm")
                    ol_out = gr.Code(label="состояние", language="json", lines=10, interactive=False)

                # ---------- HEALTH ----------
                with gr.Accordion("🩺 Health-check", open=False):
                    health_btn = gr.Button("обновить", size="sm")
                    health_out = gr.Code(label="состояние", language="json", lines=8, interactive=False)

            # ---------- ПРАВАЯ ЧАСТЬ: SIDEBAR С ХОТКЕЯМИ ----------
            with gr.Column(scale=1, min_width=260):
                gr.HTML(_hotkeys_sidebar_html())

        # ---------- STATUS BAR ----------
        gr.HTML(f"""
        <div class="status-bar">
          <div class="sb-item">
            <span class="sb-dot" id="sb-dot"></span>
            <span class="k">source</span>
            <span class="v" id="sb-source">offline</span>
          </div>
          <div class="sb-item">
            <span class="k">device</span>
            <span class="v">{engine.device.upper()}</span>
          </div>
          <div class="sb-item">
            <span class="k">model</span>
            <span class="v">{CFG.model.weights}</span>
          </div>
          <div class="sb-item">
            <span class="k">imgsz</span>
            <span class="v" id="sb-imgsz">—</span>
          </div>
          <div class="sb-item" style="margin-left:auto;">
            <span class="k">v{__version__}</span>
          </div>
        </div>
        """)

        timer = gr.Timer(value=ui.default_timer, active=False)

        # ---------- ПРЕСЕТЫ ----------
        preset_people.click(fn=lambda: "person", outputs=[prompt_input])
        preset_transport.click(fn=lambda: "car, bus, truck, bicycle, motorcycle", outputs=[prompt_input])
        preset_animals.click(fn=lambda: "dog, cat, bird, horse", outputs=[prompt_input])
        preset_all.click(
            fn=lambda: "person, car, bus, truck, bicycle, motorcycle, dog, cat",
            outputs=[prompt_input],
        )

        # ---------- ОБРАБОТКА КАРТИНКИ ----------
        def _process_image_ui(image, prompt, conf, iou, imgsz, min_area):
            raw, annotated, status, js = process_image(image, prompt, conf, iou, imgsz, min_area)
            try:
                payload = json.loads(js)
                count = payload.get("count", 0)
                classes = ", ".join(sorted({o["class"] for o in payload.get("objects", [])})) or "—"
            except Exception:
                count, classes = 0, "—"
            return raw, annotated, status, js, count, classes

        process_img_btn.click(
            fn=_process_image_ui,
            inputs=[input_image, prompt_input, conf_slider, iou_slider, imgsz_slider, min_area_slider],
            outputs=[raw_video, output_image, metric_status, json_output, metric_objects, metric_classes],
        )

        # ---------- CONNECT / DISCONNECT ----------
        def _on_connect(source_type_, url_):
            ok, msg = open_source(source_type_, url_)
            icon = "🟢" if ok else "🔴"
            return f"{icon} {msg}", gr.Timer(active=state.running)

        def _on_disconnect():
            _, msg = close_source()
            return f"⚪ {msg}", gr.Timer(active=False)

        connect_btn.click(fn=_on_connect, inputs=[source_type, camera_url], outputs=[camera_status, timer])
        disconnect_btn.click(fn=_on_disconnect, inputs=[], outputs=[camera_status, timer])

        # ---------- TIMER SPEED ----------
        def _update_timer(v: float):
            timer.value = v
            return gr.update()

        refresh_slider.change(fn=_update_timer, inputs=[refresh_slider], outputs=[refresh_slider])

        # ---------- ГЛАВНЫЙ ТИК ----------
        timer.tick(
            fn=process_stream,
            inputs=[
                prompt_input, conf_slider, iou_slider, imgsz_slider,
                min_area_slider, show_raw_checkbox, tracking_checkbox,
                roi_input,
            ],
            outputs=[
                raw_video, output_image, metric_status, json_output,
                metric_objects, metric_classes, metric_fps,
            ],
            concurrency_limit=1,
            queue=False,
        )

        # ---------- ЗАПИСЬ ----------
        def _rec_start(path: str) -> str:
            reader = state.reader
            frame = reader.read() if reader is not None else None
            if frame is None:
                return "нет кадра (подключите источник)"
            return recorder.start(path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        rec_start_btn.click(fn=_rec_start, inputs=[rec_path], outputs=[rec_status])
        rec_stop_btn.click(fn=recorder.stop, inputs=[], outputs=[rec_status])

        # ---------- ЭКСПОРТ ----------
        log_start_btn.click(fn=det_logger.start, inputs=[], outputs=[log_status])
        log_stop_btn.click(fn=det_logger.stop, inputs=[], outputs=[log_status])

        # ---------- ОНЛАЙН-ОБУЧЕНИЕ ----------
        pick_btn.click(fn=_pick_first_detection, inputs=[], outputs=[feedback_json])
        fb_tp.click(
            fn=lambda js: _fb_dispatch(js, "tp", None),
            inputs=[feedback_json], outputs=[fb_status],
        )
        fb_fp.click(
            fn=lambda js: _fb_dispatch(js, "fp", None),
            inputs=[feedback_json], outputs=[fb_status],
        )
        fb_relabel.click(
            fn=lambda js, nc: _fb_dispatch(js, "relabel", nc),
            inputs=[feedback_json, relabel_cls], outputs=[fb_status],
        )

        ol_stats_btn.click(
            fn=lambda: ONLINE_LEARNER.stats() if ONLINE_LEARNER else "{}",
            inputs=[], outputs=[ol_out],
        )
        ol_reset_btn.click(
            fn=lambda: ONLINE_LEARNER.reset() if ONLINE_LEARNER else "off",
            inputs=[], outputs=[ol_out],
        )
        ol_export_btn.click(
            fn=lambda: ONLINE_LEARNER.export_dataset() if ONLINE_LEARNER else "off",
            inputs=[], outputs=[ol_out],
        )

        health_btn.click(fn=health_check, inputs=[], outputs=[health_out])

    return demo


# =============================================================================
# 18. MAIN
# =============================================================================

def main() -> None:
    log.info("starting on %s:%d", CFG.server.host, CFG.server.port)

    auth = None
    if CFG.server.auth_user and CFG.server.auth_pass:
        auth = (CFG.server.auth_user, CFG.server.auth_pass)
        log.info("basic-auth enabled for user %r", CFG.server.auth_user)

    demo = build_ui()
    demo.launch(
        server_name=CFG.server.host,
        server_port=CFG.server.port,
        share=CFG.server.share,
        auth=auth,
        show_error=True,
    )


if __name__ == "__main__":
    main()
