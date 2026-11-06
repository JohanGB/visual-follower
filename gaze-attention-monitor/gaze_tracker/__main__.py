"""Punto de entrada: python -m gaze_tracker"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from dataclasses import replace

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from .config import Config
from .ui import GazeApp


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="gaze_tracker", description=__doc__)
    parser.add_argument("--recalibrate", action="store_true", help="ignora la calibración guardada")
    parser.add_argument("--validate", action="store_true", help="mide el error antes de iniciar")
    parser.add_argument("--camera", default="0", help="índice de cámara o ruta de un video")
    parser.add_argument("--fps", type=float, default=15.0, help="FPS en seguimiento (1-30)")
    return parser.parse_args()


def main() -> int:
    """Crea la aplicación Qt y corre el bucle de eventos."""
    args = _parse()
    os.environ.setdefault("GLOG_minloglevel", "2")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    camera = int(args.camera) if args.camera.isdigit() else args.camera
    cfg = replace(Config(), camera=camera, fps_active=min(max(args.fps, 1.0), 30.0))

    # En Linux, cv2 apunta Qt a sus propios plugins y rompería PyQt6.
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # Python solo atiende Ctrl+C si el bucle de Qt le cede control de vez en cuando.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    heartbeat = QTimer()
    heartbeat.start(500)
    heartbeat.timeout.connect(lambda: None)

    gaze_app = GazeApp(app, cfg, recalibrate=args.recalibrate, validate=args.validate)
    gaze_app.start()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
