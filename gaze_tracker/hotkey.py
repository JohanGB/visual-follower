"""Atajo global Ctrl+Shift+H, sin hooks de teclado.

En Windows se usa RegisterHotKey: el sistema solo avisa al pulsar esa
combinación, de modo que no se observa el resto del teclado y el hilo queda
bloqueado sin consumir CPU. En otros sistemas se recurre a pynput.
"""

from __future__ import annotations

import sys
import threading
from typing import Callable

_MOD_CONTROL = 0x0002
_MOD_SHIFT = 0x0004
_MOD_NOREPEAT = 0x4000
_VK_H = 0x48
_WM_HOTKEY = 0x0312
_WM_QUIT = 0x0012
_HOTKEY_ID = 0x4748


class GlobalHotkey:
    """Ejecuta `callback` en un hilo propio; conéctalo a una señal de Qt."""

    def __init__(self, callback: Callable[[], None]) -> None:
        self._callback = callback
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._listener = None
        self._ready = threading.Event()
        self.error: str | None = None

    def start(self) -> bool:
        """Registra el atajo; devuelve False si no se pudo."""
        if sys.platform == "win32":
            self._thread = threading.Thread(target=self._run_win32, daemon=True)
            self._thread.start()
            self._ready.wait(2.0)
            return self.error is None
        return self._start_pynput()

    def _run_win32(self) -> None:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self._thread_id = kernel32.GetCurrentThreadId()
        mods = _MOD_CONTROL | _MOD_SHIFT | _MOD_NOREPEAT
        if not user32.RegisterHotKey(None, _HOTKEY_ID, mods, _VK_H):
            self.error = "Ctrl+Shift+H ya está en uso por otra aplicación."
            self._ready.set()
            return
        self._ready.set()
        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == _WM_HOTKEY and msg.wParam == _HOTKEY_ID:
                    self._callback()
        finally:
            user32.UnregisterHotKey(None, _HOTKEY_ID)

    def _start_pynput(self) -> bool:
        try:
            from pynput import keyboard

            self._listener = keyboard.GlobalHotKeys({"<ctrl>+<shift>+h": self._callback})
            self._listener.start()
            return True
        except Exception as exc:  # pynput falla sin servidor gráfico compatible
            self.error = f"Atajo global no disponible: {exc}"
            return False

    def stop(self) -> None:
        """Libera el atajo."""
        if sys.platform == "win32" and self._thread_id:
            import ctypes

            ctypes.windll.user32.PostThreadMessageW(self._thread_id, _WM_QUIT, 0, 0)
        if self._listener is not None:
            self._listener.stop()
