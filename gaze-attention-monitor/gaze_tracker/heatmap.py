"""Mapa de calor de fijaciones: acumulación O(1) e imagen PNG a resolución real."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_BG = np.array([0x19, 0x1A, 0x1A], np.float32)  # BGR
_GRID = np.array([0x2C, 0x2D, 0x2D], np.float32)
_INK = (0xE8, 0xE6, 0xE3)
_MUTED = (0xA0, 0x9C, 0x98)
# Rampa secuencial de un solo tono: de ámbar profundo a ámbar muy claro (BGR).
_RAMP = np.array(
    [[0x0C, 0x2A, 0x6E], [0x0B, 0x4E, 0xB4], [0x1E, 0x8C, 0xF0], [0xA8, 0xDC, 0xFD]],
    np.float32,
)


class FixationHeatmap:
    """Rejilla de tiempo de fijación. Su memoria no crece con la sesión."""

    def __init__(self, cols: int, screen: tuple[int, int], max_speed: float) -> None:
        self.screen = screen
        self._cols = cols
        self._rows = max(1, round(cols * screen[1] / screen[0]))
        self._grid = np.zeros((self._rows, cols), np.float32)
        self._max_speed = max_speed
        self._last: tuple[float, float, float] | None = None
        self.fixation_s = 0.0
        self.session_s = 0.0

    def add(self, x: float, y: float, t: float) -> None:
        """Suma una muestra si es fijación: lenta y dentro de la pantalla."""
        last, self._last = self._last, (x, y, t)
        if last is None:
            return
        dt = t - last[2]
        if not 0.0 < dt < 0.5:
            return
        self.session_s += dt
        if not (0.0 <= x < 1.0 and 0.0 <= y < 1.0):
            return
        if (x - last[0]) ** 2 + (y - last[1]) ** 2 > (self._max_speed * dt) ** 2:
            return
        self._grid[int(y * self._rows), int(x * self._cols)] += dt
        self.fixation_s += dt

    def break_sequence(self) -> None:
        """Evita medir velocidad a través de un hueco (rostro perdido)."""
        self._last = None

    def render(self, path: Path, blur_fraction: float = 0.03) -> Path | None:
        """Escribe el PNG. El desenfoque refleja la incertidumbre de la mirada."""
        if self.fixation_s <= 0.0:
            return None
        w, h = self.screen
        sigma = max(blur_fraction, 0.015) * self._cols
        density = cv2.GaussianBlur(self._grid, (0, 0), sigma)
        density = cv2.resize(density, (w, h), interpolation=cv2.INTER_CUBIC).clip(0.0)
        level = np.clip(density / max(float(np.percentile(density, 99.8)), 1e-9), 0.0, 1.0)

        canvas = np.empty((h, w, 3), np.float32)
        canvas[:] = _BG
        step = max(1, round(w / 12))
        canvas[:, step::step] = _GRID
        canvas[step::step, :] = _GRID

        stops = np.linspace(0.0, 1.0, _RAMP.shape[0])
        color = np.stack([np.interp(level, stops, _RAMP[:, c]) for c in range(3)], axis=-1)
        ramp_in = np.clip((level - 0.02) / 0.7, 0.0, 1.0)
        alpha = (ramp_in * ramp_in * (3.0 - 2.0 * ramp_in))[..., None]  # sin bordes duros
        image = (canvas * (1.0 - alpha) + color * alpha).astype(np.uint8)

        self._draw_caption(image, w, h, stops)
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            return None
        path.write_bytes(encoded.tobytes())  # write_bytes admite rutas con acentos
        return path

    def _draw_caption(self, image: np.ndarray, w: int, h: int, stops: np.ndarray) -> None:
        """Título, tiempos y leyenda de la escala de color."""
        scale = max(w / 1920.0, 0.5)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thick = max(1, round(scale * 1.4))
        x0, y0 = round(32 * scale), round(52 * scale)
        cv2.putText(image, "Mapa de calor de fijaciones", (x0, y0), font, 0.9 * scale,
                    _INK, thick + 1, cv2.LINE_AA)
        share = 100.0 * self.fixation_s / max(self.session_s, 1e-9)
        text = f"{self.session_s / 60:.1f} min registrados | {share:.0f}% fijando la mirada | {w}x{h}"
        cv2.putText(image, text, (x0, y0 + round(34 * scale)), font, 0.58 * scale,
                    _MUTED, thick, cv2.LINE_AA)

        bar_w, bar_h = round(220 * scale), round(10 * scale)
        bx, by = w - bar_w - x0, y0 - bar_h
        ramp = np.linspace(0.0, 1.0, bar_w)
        bar = np.stack([np.interp(ramp, stops, _RAMP[:, c]) for c in range(3)], axis=-1)
        image[by:by + bar_h, bx:bx + bar_w] = bar[None].astype(np.uint8)
        y_label = by + bar_h + round(24 * scale)
        cv2.putText(image, "menor tiempo", (bx, y_label), font, 0.5 * scale, _MUTED, thick, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize("mayor tiempo", font, 0.5 * scale, thick)
        cv2.putText(image, "mayor tiempo", (bx + bar_w - tw, y_label), font, 0.5 * scale,
                    _MUTED, thick, cv2.LINE_AA)
