"""Minimal menubar app using rumps."""
import threading
import logging
import rumps

from .config import TranslatorConfig, Mode, MicMode
from .engine import TranslationEngine
from .streams import AudioStreamManager

log = logging.getLogger(__name__)


class TranslatorMenubar(rumps.App):
    def __init__(self):
        super().__init__("TR", icon=None, quit_button=None)

        self.config = TranslatorConfig(mode=Mode.HYBRID)
        self.engine = TranslationEngine(self.config)
        self.streams: AudioStreamManager | None = None
        self._is_running = False

        # Wire callbacks
        self.engine.on_subtitle = self._on_subtitle
        self.engine.on_status = self._on_status

        self._last_subtitle = ""

        # Menu items
        self.menu = [
            rumps.MenuItem("Start", callback=self._toggle),
            None,  # separator
            rumps.MenuItem("Mode: hybrid", callback=None),
            rumps.MenuItem("  Local", callback=lambda _: self._set_mode(Mode.LOCAL)),
            rumps.MenuItem("  Cloud", callback=lambda _: self._set_mode(Mode.CLOUD)),
            rumps.MenuItem("  Hybrid", callback=lambda _: self._set_mode(Mode.HYBRID)),
            None,
            rumps.MenuItem("Mic: translated", callback=None),
            rumps.MenuItem("  Send Translated", callback=lambda _: self._set_mic(MicMode.TRANSLATED)),
            rumps.MenuItem("  Send Original", callback=lambda _: self._set_mic(MicMode.ORIGINAL)),
            rumps.MenuItem("  Send Both", callback=lambda _: self._set_mic(MicMode.BOTH)),
            None,
            rumps.MenuItem("Last: ---"),
            None,
            rumps.MenuItem("Quit", callback=self._quit),
        ]

    def _on_subtitle(self, direction, original, translated):
        arrow = "EN→RU" if direction == "incoming" else "RU→EN"
        self._last_subtitle = f"{arrow}: {translated[:50]}"
        try:
            self.menu["Last: ---"].title = f"Last: {self._last_subtitle}"
        except Exception:
            pass

    def _on_status(self, msg):
        log.info(f"[menubar] {msg}")

    def _toggle(self, sender):
        if self._is_running:
            self._stop()
            sender.title = "Start"
            self.title = "TR"
        else:
            self._start()
            sender.title = "Stop"
            self.title = "TR ●"

    def _start(self):
        def _run():
            try:
                self.engine.load_models()
                self.streams = AudioStreamManager(self.config, self.engine)
                self.streams.start()
                self._is_running = True
            except Exception as e:
                log.error(f"Start failed: {e}")
                rumps.notification("Translator", "Error", str(e))

        threading.Thread(target=_run, daemon=True).start()

    def _stop(self):
        if self.streams:
            self.streams.stop()
        self._is_running = False

    def _set_mode(self, mode: Mode):
        self.config.mode = mode
        try:
            self.menu["Mode: hybrid"].title = f"Mode: {mode.value}"
        except Exception:
            pass
        if self._is_running:
            self._stop()
            self._start()

    def _set_mic(self, mic_mode: MicMode):
        self.config.mic_mode = mic_mode
        try:
            self.menu["Mic: translated"].title = f"Mic: {mic_mode.value}"
        except Exception:
            pass

    def _quit(self, _):
        self._stop()
        rumps.quit_application()


def run_menubar():
    TranslatorMenubar().run()


if __name__ == "__main__":
    run_menubar()
