"""Tamaño físico del monitor, leído del EDID que Windows guarda en el registro."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

log = logging.getLogger(__name__)

_EDID_HEADER = bytes((0, 255, 255, 255, 255, 255, 255, 0))
_MIN_MM = 100
_PRIMARY = 0x4
_DEVICE_INTERFACE = 0x1


@dataclass(frozen=True, slots=True)
class ScreenInfo:
    """Medidas físicas de la pantalla principal."""

    width_mm: float
    height_mm: float
    source: str  # "EDID" o "estimado"

    @property
    def inches(self) -> float:
        """Diagonal en pulgadas."""
        return (self.width_mm**2 + self.height_mm**2) ** 0.5 / 25.4


def parse_edid(edid: bytes) -> tuple[int, int] | None:
    """Extrae ancho y alto en mm. Devuelve None si el bloque no es válido."""
    if len(edid) < 128 or edid[:8] != _EDID_HEADER or sum(edid[:128]) % 256 != 0:
        return None
    # El descriptor detallado da milímetros; el bloque básico, solo centímetros.
    width = edid[66] | ((edid[68] & 0xF0) << 4)
    height = edid[67] | ((edid[68] & 0x0F) << 8)
    if width < _MIN_MM or height < _MIN_MM:
        width, height = edid[21] * 10, edid[22] * 10
    return (width, height) if width >= _MIN_MM and height >= _MIN_MM else None


def registry_key(device_id: str) -> str | None:
    r"""Convierte \\?\DISPLAY#GSM5BBF#5&3a&0&UID#{guid} en su clave de registro."""
    parts = device_id.split("#")
    if len(parts) < 3 or not parts[1] or not parts[2]:
        return None
    return rf"SYSTEM\CurrentControlSet\Enum\DISPLAY\{parts[1]}\{parts[2]}\Device Parameters"


def fit_orientation(size: tuple[int, int], pixels: tuple[int, int]) -> ScreenInfo | None:
    """Alinea las medidas con la orientación y descarta las incoherentes."""
    width, height = size
    if (width > height) != (pixels[0] > pixels[1]):
        width, height = height, width
    if abs(width / height - pixels[0] / pixels[1]) > 0.2:
        return None
    return ScreenInfo(float(width), float(height), "EDID")


def _primary_device_id() -> str | None:
    import ctypes
    from ctypes import wintypes

    class DisplayDevice(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("DeviceName", wintypes.WCHAR * 32),
            ("DeviceString", wintypes.WCHAR * 128),
            ("StateFlags", wintypes.DWORD),
            ("DeviceID", wintypes.WCHAR * 128),
            ("DeviceKey", wintypes.WCHAR * 128),
        ]

    user32 = ctypes.windll.user32
    adapter = DisplayDevice(cb=ctypes.sizeof(DisplayDevice))
    index = 0
    while user32.EnumDisplayDevicesW(None, index, ctypes.byref(adapter), 0):
        if adapter.StateFlags & _PRIMARY:
            monitor = DisplayDevice(cb=ctypes.sizeof(DisplayDevice))
            if user32.EnumDisplayDevicesW(
                adapter.DeviceName, 0, ctypes.byref(monitor), _DEVICE_INTERFACE
            ):
                return monitor.DeviceID
            return None
        index += 1
    return None


def _read_edid(key: str) -> bytes | None:
    import winreg

    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
        return bytes(winreg.QueryValueEx(handle, "EDID")[0])


def detect_screen(pixels: tuple[int, int]) -> ScreenInfo | None:
    """Detecta la pantalla principal. Nunca falla: sin datos devuelve None."""
    if sys.platform != "win32":
        return None
    try:
        device_id = _primary_device_id()
        key = registry_key(device_id) if device_id else None
        edid = _read_edid(key) if key else None
        size = parse_edid(edid) if edid else None
        return fit_orientation(size, pixels) if size else None
    except (OSError, ValueError) as exc:
        log.warning("No se pudo leer el EDID: %s", exc)
        return None
