"""Repite el ajuste sobre la última sesión guardada, sin volver a calibrar.

Uso: python tools/replay.py [ruta de last_session.npz]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaze_tracker.calibration import report_text  # noqa: E402
from gaze_tracker.config import Config  # noqa: E402
from gaze_tracker.display import ScreenInfo  # noqa: E402
from gaze_tracker.personal import fit_session  # noqa: E402


def main() -> None:
    cfg = Config()
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else cfg.session_path
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in ("t", "pts", "tag", "t_tgt", "xy_tgt")}
        screen = tuple(int(v) for v in saved["screen"])
        size = tuple(int(v) for v in saved["size"])
        millimeters = saved["screen_mm"]
    info = None if np.isnan(millimeters[0]) else ScreenInfo(*map(float, millimeters), "EDID")
    model, report = fit_session(data, screen, cfg, size, info)
    print(report_text(report) if report else "Sin puntos de validación suficientes.")
    print(model.params)


if __name__ == "__main__":
    main()
