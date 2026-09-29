"""
Entrypoint. Собирает DI-контейнер и запускает сервер.
"""
from __future__ import annotations

import atexit
import logging
import signal
from typing import Any

from config import CFG
from domain import DetectionEngine, StateStore, VideoRecorder
from services import ServiceContainer
from ui.layout import Handlers, build_ui


log = logging.getLogger("yoloe.main")


def _bootstrap() -> ServiceContainer:
    engine = DetectionEngine(CFG.model)
    state = StateStore(CFG.camera)
    recorder = VideoRecorder()
    return ServiceContainer(engine, state, recorder)


def _make_handlers(sc: ServiceContainer) -> Handlers:
    return Handlers(
        process_image=sc.process_image,
        process_stream=sc.process_stream,
        on_connect=sc.on_connect,
        on_disconnect=sc.on_disconnect,
        on_update_timer=sc.on_update_timer,
        on_start_recording=sc.on_start_recording,
        on_stop_recording=sc.on_stop_recording,
    )


def _setup_shutdown(sc: ServiceContainer) -> None:
    def _shutdown(*_: Any) -> None:
        log.info("Shutting down...")
        sc.state.detach()
        sc.recorder.release()
        log.info("Cleanup complete.")

    atexit.register(_shutdown)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *a: (_shutdown(), exit(0)))
        except (ValueError, OSError):
            pass


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log.info("Bootstrapping services...")
    sc = _bootstrap()
    _setup_shutdown(sc)

    log.info("Building UI...")
    handlers = _make_handlers(sc)
    demo = build_ui(handlers)

    log.info("Starting server on %s:%d", CFG.server.host, CFG.server.port)
    demo.launch(
        server_name=CFG.server.host,
        server_port=CFG.server.port,
        share=CFG.server.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
