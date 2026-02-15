"""CLI controller for the live translator."""
import argparse
import logging
import signal
import sys
import time

from .config import TranslatorConfig, Mode, MicMode, LOGS_DIR
from .audio_devices import print_routing_status, check_blackhole_installed, setup_instructions
from .engine import TranslationEngine
from .streams import AudioStreamManager

log = logging.getLogger("translator")


def setup_logging(log_file: str, verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt,
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler(sys.stdout),
                        ])


class TranslatorApp:
    def __init__(self, config: TranslatorConfig):
        self.config = config
        self.engine = TranslationEngine(config)
        self.streams: AudioStreamManager | None = None

        # Wire up subtitle callback
        self.engine.on_subtitle = self._on_subtitle
        self.engine.on_status = self._on_status

    def _on_subtitle(self, direction: str, original: str, translated: str):
        arrow = "EN→RU" if direction == "incoming" else "RU→EN"
        print(f"\n  [{arrow}] {original}")
        print(f"  [{arrow}] → {translated}")

    def _on_status(self, msg: str):
        print(f"  [STATUS] {msg}")

    def start(self):
        print("=" * 50)
        print("  Live Translator RU↔EN")
        print(f"  Mode: {self.config.mode.value}")
        print(f"  Mic mode: {self.config.mic_mode.value}")
        print("=" * 50)

        self.engine.load_models()

        self.streams = AudioStreamManager(self.config, self.engine)
        self.streams.start()

        print("\n  Translator is running. Press Ctrl+C to stop.\n")

    def stop(self):
        if self.streams:
            self.streams.stop()
        print("\n  Translator stopped.")

    def run(self):
        self.start()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()


def main():
    parser = argparse.ArgumentParser(description="Live bidirectional RU↔EN translator")
    sub = parser.add_subparsers(dest="command", help="Command")

    # start
    start_p = sub.add_parser("start", help="Start the translator")
    start_p.add_argument("--mode", choices=["local", "cloud", "hybrid"], default="hybrid")
    start_p.add_argument("--mic-mode", choices=["translated", "original", "both"], default="translated")
    start_p.add_argument("--whisper-model", default="small", help="Whisper model size")
    start_p.add_argument("--input-device", default="MacBook Pro", help="Physical mic name")
    start_p.add_argument("--output-device", default="MacBook Pro", help="Headphones/speaker name")
    start_p.add_argument("--verbose", "-v", action="store_true")

    # status
    sub.add_parser("status", help="Show device status")

    # route
    route_p = sub.add_parser("route", help="Audio routing")
    route_p.add_argument("action", choices=["setup", "check"], default="check", nargs="?")

    # mode
    mode_p = sub.add_parser("mode", help="Switch mode")
    mode_p.add_argument("new_mode", choices=["local", "cloud", "hybrid"])

    # test
    sub.add_parser("test", help="Run a quick translation test")

    # devices
    sub.add_parser("devices", help="List audio devices")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "devices":
        print_routing_status()
        return

    if args.command == "status":
        print_routing_status()
        bh = check_blackhole_installed()
        print(f"  BlackHole installed: {'YES' if bh else 'NO'}")
        return

    if args.command == "route":
        if args.action == "setup":
            print(setup_instructions())
        else:
            print_routing_status()
        return

    if args.command == "test":
        _run_test()
        return

    if args.command == "start":
        config = TranslatorConfig(
            mode=Mode(args.mode),
            mic_mode=MicMode(args.mic_mode),
            whisper_model=args.whisper_model,
            input_device=args.input_device,
            output_device=args.output_device,
        )
        setup_logging(config.log_file, verbose=args.verbose)
        app = TranslatorApp(config)

        def sig_handler(signum, frame):
            app.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, sig_handler)
        signal.signal(signal.SIGTERM, sig_handler)

        app.run()


def _run_test():
    """Quick test: translate a text snippet and synthesize."""
    print("Running quick translation test...")
    print()

    config = TranslatorConfig(mode=Mode.HYBRID)
    engine = TranslationEngine(config)
    engine.load_models()

    # Test RU→EN
    test_ru = "Привет, как дела? Это тест системы перевода."
    translated_en = engine._translate_text(test_ru, "ru", "en")
    print(f"  RU→EN: '{test_ru}' → '{translated_en}'")

    # Test EN→RU
    test_en = "Hello, how are you? This is a translation system test."
    translated_ru = engine._translate_text(test_en, "en", "ru")
    print(f"  EN→RU: '{test_en}' → '{translated_ru}'")

    # Test TTS
    print("\n  Testing TTS (you should hear Russian audio)...")
    tts_audio = engine._tts.synthesize(translated_ru, lang="ru")
    if tts_audio is not None:
        import sounddevice as sd
        sd.play(tts_audio, samplerate=engine._tts.sample_rate)
        sd.wait()
        print("  TTS test complete.")
    else:
        print("  TTS failed - no audio generated.")

    print("\n  Test complete!")


if __name__ == "__main__":
    main()
