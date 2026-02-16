"""GUI for Live Translator RU↔EN."""
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import logging
import time
import sys
from pathlib import Path

from .config import TranslatorConfig, Mode, MicMode, LOGS_DIR
from .audio_devices import find_device, check_blackhole_installed
from .engine import TranslationEngine
from .streams import AudioStreamManager

log = logging.getLogger("translator")


class TranslatorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Живой Переводчик RU↔EN")
        self.root.geometry("540x900")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e2e")

        # State
        self.config = TranslatorConfig(mode=Mode.HYBRID)
        self.engine: TranslationEngine | None = None
        self.streams: AudioStreamManager | None = None
        self.is_running = False
        self.is_loading = False
        self.voice_profile_exists = False

        # Style
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self._setup_styles()

        # Build UI
        self._build_header()
        self._build_controls()
        self._build_device_info()
        self._build_subtitle_log()
        self._build_status_bar()

        # Setup logging to GUI
        self._setup_gui_logging()

        # Initial device check
        self.root.after(500, self._check_devices)

    def _setup_styles(self):
        bg = "#1e1e2e"
        fg = "#cdd6f4"
        accent = "#89b4fa"
        green = "#a6e3a1"
        red = "#f38ba8"
        surface = "#313244"

        self.colors = {
            "bg": bg, "fg": fg, "accent": accent,
            "green": green, "red": red, "surface": surface,
        }

        self.style.configure("TFrame", background=bg)
        self.style.configure("TLabel", background=bg, foreground=fg, font=("SF Pro Display", 13))
        self.style.configure("Header.TLabel", background=bg, foreground=fg, font=("SF Pro Display", 20, "bold"))
        self.style.configure("Status.TLabel", background=surface, foreground=fg, font=("SF Mono", 11))
        self.style.configure("Small.TLabel", background=bg, foreground="#6c7086", font=("SF Pro Display", 11))

        # Buttons
        self.style.configure("Start.TButton", font=("SF Pro Display", 14, "bold"), padding=(20, 10))
        self.style.configure("Mode.TButton", font=("SF Pro Display", 11), padding=(10, 5))

    def _build_header(self):
        frame = ttk.Frame(self.root)
        frame.pack(fill="x", padx=20, pady=(15, 5))

        ttk.Label(frame, text="Живой Переводчик", style="Header.TLabel").pack(side="left")

        self.status_dot = tk.Canvas(frame, width=14, height=14, bg=self.colors["bg"], highlightthickness=0)
        self.status_dot.pack(side="right", padx=(0, 5))
        self._draw_dot("gray")

    def _draw_dot(self, color):
        self.status_dot.delete("all")
        self.status_dot.create_oval(2, 2, 12, 12, fill=color, outline="")

    def _build_controls(self):
        frame = ttk.Frame(self.root)
        frame.pack(fill="x", padx=20, pady=10)

        # Start/Stop button
        self.start_btn = tk.Button(
            frame, text="СТАРТ", font=("SF Pro Display", 15, "bold"),
            bg=self.colors["green"], fg="#1e1e2e", activebackground="#7dd99e",
            relief="flat", width=12, height=1, cursor="hand2",
            command=self._toggle,
        )
        self.start_btn.pack(side="left", padx=(0, 10))

        # Mode selector
        mode_frame = ttk.Frame(frame)
        mode_frame.pack(side="left", fill="y")

        ttk.Label(mode_frame, text="Режим:", style="Small.TLabel").pack(anchor="w")
        self.mode_var = tk.StringVar(value="hybrid")
        mode_menu = ttk.Combobox(
            mode_frame, textvariable=self.mode_var,
            values=["local", "cloud", "hybrid"],
            state="readonly", width=10,
        )
        mode_menu.pack(anchor="w")
        mode_menu.bind("<<ComboboxSelected>>", self._on_mode_change)

        # Mic mode selector
        mic_frame = ttk.Frame(frame)
        mic_frame.pack(side="left", fill="y", padx=(15, 0))

        ttk.Label(mic_frame, text="Микрофон:", style="Small.TLabel").pack(anchor="w")
        self.mic_var = tk.StringVar(value="translated")
        mic_menu = ttk.Combobox(
            mic_frame, textvariable=self.mic_var,
            values=["перевод", "оригинал", "оба"],
            state="readonly", width=10,
        )
        mic_menu.pack(anchor="w")
        mic_menu.bind("<<ComboboxSelected>>", self._on_mic_change)

        # Direction buttons (manual switching)
        dir_frame = ttk.Frame(self.root)
        dir_frame.pack(fill="x", padx=20, pady=(5, 0))

        ttk.Label(dir_frame, text="Направление перевода:", style="Small.TLabel").pack(anchor="w")
        
        btn_frame = ttk.Frame(dir_frame)
        btn_frame.pack(fill="x", pady=(3, 0))

        self.dir_var = tk.StringVar(value="outgoing")
        
        self.outgoing_btn = tk.Button(
            btn_frame, text="Я → СОБЕСЕДНИК (RU→EN)",
            font=("SF Pro Display", 12, "bold"),
            bg="#a6e3a1", fg="#1e1e2e", activebackground="#7dd99e",
            relief="flat", width=22, height=1, cursor="hand2",
            command=lambda: self._set_direction("outgoing"),
        )
        self.outgoing_btn.pack(side="left", padx=(0, 5))

        self.incoming_btn = tk.Button(
            btn_frame, text="СОБЕСЕДНИК → Я (EN→RU)",
            font=("SF Pro Display", 12, "bold"),
            bg="#585b70", fg="#cdd6f4", activebackground="#6c7086",
            relief="flat", width=22, height=1, cursor="hand2",
            command=lambda: self._set_direction("incoming"),
        )
        self.incoming_btn.pack(side="left")

        # Auto-detect speaker toggle
        auto_frame = ttk.Frame(self.root)
        auto_frame.pack(fill="x", padx=20, pady=(5, 0))

        self.auto_detect_var = tk.BooleanVar(value=False)
        self.auto_detect_btn = tk.Button(
            auto_frame, text="АВТООПРЕДЕЛЕНИЕ: ВЫКЛ",
            font=("SF Pro Display", 11, "bold"),
            bg="#585b70", fg="#cdd6f4", activebackground="#6c7086",
            relief="flat", width=44, height=1, cursor="hand2",
            command=self._toggle_auto_detect,
        )
        self.auto_detect_btn.pack(fill="x")
        
        # Status label for speaker ID
        self.speaker_status = ttk.Label(
            auto_frame, text="Голосовой профиль: не загружен", 
            style="Small.TLabel"
        )
        self.speaker_status.pack(anchor="w", pady=(2, 0))

        # Speaker ID settings
        speaker_frame = ttk.Frame(self.root)
        speaker_frame.pack(fill="x", padx=20, pady=(8, 0))

        ttk.Label(speaker_frame, text="Распознавание голоса:", style="Small.TLabel").pack(anchor="w")
        
        hf_frame = ttk.Frame(speaker_frame)
        hf_frame.pack(fill="x", pady=(3, 0))

        self.hf_token_var = tk.StringVar(value="")
        ttk.Label(hf_frame, text="HF Токен:", style="Small.TLabel").pack(side="left")
        self.hf_entry = tk.Entry(
            hf_frame, textvariable=self.hf_token_var,
            width=25, bg="#313244", fg="#cdd6f4",
            insertbackground="#cdd6f4", relief="flat"
        )
        self.hf_entry.pack(side="left", padx=(5, 5))
        
        self.enroll_btn = tk.Button(
            hf_frame, text="ЗАПИСЬ ГОЛОСА",
            font=("SF Pro Display", 10, "bold"),
            bg="#89b4fa", fg="#1e1e2e", activebackground="#b4d0fe",
            relief="flat", width=12, cursor="hand2",
            command=self._start_enrollment,
        )
        self.enroll_btn.pack(side="left")

        # Preview button - hear what your partner hears
        preview_frame = ttk.Frame(self.root)
        preview_frame.pack(fill="x", padx=20, pady=(5, 0))

        self.preview_btn = tk.Button(
            preview_frame, text="ПРЕСЛУШИВАНИЕ ВЫКЛ",
            font=("SF Pro Display", 12, "bold"),
            bg="#585b70", fg="#cdd6f4", activebackground="#6c7086",
            relief="flat", width=30, height=1, cursor="hand2",
            command=self._toggle_preview,
        )
        self.preview_btn.pack(fill="x")
        self.preview_on = False

    def _build_device_info(self):
        frame = ttk.Frame(self.root)
        frame.pack(fill="x", padx=20, pady=(5, 5))

        # Separator
        sep = tk.Frame(frame, height=1, bg="#45475a")
        sep.pack(fill="x", pady=(0, 8))

        self.device_labels = {}
        devices = [
            ("mic", "Микрофон"),
            ("speaker", "Динамик"),
            ("blackhole", "Виртуальный мик (BlackHole)"),
        ]
        for key, label in devices:
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=f"{label}:", style="Small.TLabel", width=25).pack(side="left")
            lbl = ttk.Label(row, text="проверка...", style="Small.TLabel")
            lbl.pack(side="left")
            self.device_labels[key] = lbl

    def _build_subtitle_log(self):
        frame = ttk.Frame(self.root)
        frame.pack(fill="both", expand=True, padx=20, pady=(5, 5))

        ttk.Label(frame, text="Журнал переводов:", style="Small.TLabel").pack(anchor="w")

        self.log_text = scrolledtext.ScrolledText(
            frame, height=12, wrap="word",
            bg="#313244", fg="#cdd6f4",
            font=("SF Mono", 11),
            insertbackground="#cdd6f4",
            relief="flat", borderwidth=0,
            state="disabled",
        )
        self.log_text.pack(fill="both", expand=True, pady=(3, 0))

        # Tags for coloring
        self.log_text.tag_configure("incoming", foreground="#89b4fa")
        self.log_text.tag_configure("outgoing", foreground="#a6e3a1")
        self.log_text.tag_configure("system", foreground="#6c7086")
        self.log_text.tag_configure("error", foreground="#f38ba8")

    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg="#313244", height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self.status_label = tk.Label(
            bar, text="Готов", bg="#313244", fg="#6c7086",
            font=("SF Mono", 11), anchor="w",
        )
        self.status_label.pack(side="left", padx=10)

        self.latency_label = tk.Label(
            bar, text="", bg="#313244", fg="#6c7086",
            font=("SF Mono", 11), anchor="e",
        )
        self.latency_label.pack(side="right", padx=10)

    def _setup_gui_logging(self):
        fmt = "%(asctime)s [%(levelname)s] %(message)s"
        datefmt = "%H:%M:%S"
        logging.basicConfig(
            level=logging.INFO, format=fmt, datefmt=datefmt,
            handlers=[
                logging.FileHandler(str(LOGS_DIR / "translator.log")),
            ],
        )

    def _log(self, msg: str, tag: str = "system"):
        self.log_text.configure(state="normal")
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {msg}\n", tag)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_status(self, text: str):
        self.status_label.configure(text=text)

    def _check_devices(self):
        import sounddevice as sd

        mic = find_device(self.config.input_device, kind="input")
        spk = find_device(self.config.output_device, kind="output")
        bh = find_device("BlackHole", kind="output")

        if mic is not None:
            name = sd.query_devices(mic)["name"]
            self.device_labels["mic"].configure(text=name, foreground=self.colors["green"])
        else:
            self.device_labels["mic"].configure(text="НЕ НАЙДЕН", foreground=self.colors["red"])

        if spk is not None:
            name = sd.query_devices(spk)["name"]
            self.device_labels["speaker"].configure(text=name, foreground=self.colors["green"])
        else:
            self.device_labels["speaker"].configure(text="НЕ НАЙДЕН", foreground=self.colors["red"])

        if bh is not None:
            name = sd.query_devices(bh)["name"]
            self.device_labels["blackhole"].configure(text=name, foreground=self.colors["green"])
        else:
            self.device_labels["blackhole"].configure(text="НЕ НАЙДЕН", foreground=self.colors["red"])

        # Check voice profile
        from pathlib import Path
        profile_path = Path(__file__).parent.parent / "config" / "voice_profile.npy"
        if profile_path.exists():
            self.speaker_status.configure(text="Голосовой профиль: загружен ✓", foreground=self.colors["green"])
            self.voice_profile_exists = True
        else:
            self.speaker_status.configure(text="Голосовой профиль: не записан", foreground="#f9e2af")
            self.voice_profile_exists = False

        self._log("Устройства проверены.")

    def _toggle(self):
        if self.is_loading:
            return
        if self.is_running:
            self._stop()
        else:
            self._start()

    def _start(self):
        self.is_loading = True
        self.start_btn.configure(text="ЗАГРУЗКА...", bg="#f9e2af", state="disabled")
        self._draw_dot("#f9e2af")
        self._set_status("Загрузка моделей...")
        self._log("Запуск переводчика...")

        def _run():
            try:
                self.config.mode = Mode(self.mode_var.get())
                # Map Russian mic mode to enum
                mic_map = {"перевод": "translated", "оригинал": "original", "оба": "both"}
                mic_val = mic_map.get(self.mic_var.get(), "translated")
                self.config.mic_mode = MicMode(mic_val)

                self.engine = TranslationEngine(self.config)
                self.engine.on_subtitle = self._on_subtitle
                self.engine.on_status = lambda msg: self.root.after(0, self._log, msg)

                self.engine.load_models()

                self.streams = AudioStreamManager(self.config, self.engine)
                # Pass auto-detect state if enabled
                if self.auto_detect_var.get():
                    self.streams.set_auto_detect(True)
                self.streams.start()

                self.is_running = True
                self.is_loading = False
                self.root.after(0, self._update_ui_running)

            except Exception as e:
                self.is_loading = False
                self.root.after(0, self._on_error, str(e))

        threading.Thread(target=_run, daemon=True).start()

    def _stop(self):
        self._log("Остановка переводчика...")
        self._set_status("Остановка...")

        if self.streams:
            self.streams.stop()
            self.streams = None

        self.engine = None
        self.is_running = False

        self.start_btn.configure(text="СТАРТ", bg=self.colors["green"], state="normal")
        self._draw_dot("gray")
        self._set_status("Остановлен")
        self._log("Переводчик остановлен.")

    def _update_ui_running(self):
        self.start_btn.configure(text="СТОП", bg=self.colors["red"], state="normal")
        self._draw_dot(self.colors["green"])
        self._set_status(f"Работает — режим: {self.config.mode.value}")
        self._log("Переводчик запущен. Говорите в микрофон.", "system")

    def _on_error(self, msg: str):
        self.start_btn.configure(text="СТАРТ", bg=self.colors["green"], state="normal")
        self._draw_dot(self.colors["red"])
        self._set_status(f"Ошибка: {msg[:50]}")
        self._log(f"ОШИБКА: {msg}", "error")

    def _on_subtitle(self, direction: str, original: str, translated: str):
        if direction == "incoming":
            tag = "incoming"
            arrow = "EN→RU"
        else:
            tag = "outgoing"
            arrow = "RU→EN"

        def _update():
            self._log(f"[{arrow}] {original}", tag)
            self._log(f"[{arrow}] → {translated}", tag)

        self.root.after(0, _update)

    def _toggle_preview(self):
        self.preview_on = not self.preview_on
        if self.preview_on:
            self.preview_btn.configure(
                text="ПРЕСЛУШИВАНИЕ ВКЛ",
                bg="#f9e2af", fg="#1e1e2e",
            )
            if self.streams:
                self.streams.preview_enabled = True
            self._log("Преслушивание включено", "system")
        else:
            self.preview_btn.configure(
                text="ПРЕСЛУШИВАНИЕ ВЫКЛ",
                bg="#585b70", fg="#cdd6f4",
            )
            if self.streams:
                self.streams.preview_enabled = False
            self._log("Преслушивание выключено", "system")

    def _on_mode_change(self, event=None):
        mode = self.mode_var.get()
        self.config.mode = Mode(mode)
        self._log(f"Режим изменён на: {mode}")
        if self.is_running:
            self._log("Требуется перезапуск для применения изменений.", "system")

    def _on_mic_change(self, event=None):
        mic_mode = self.mic_var.get()
        # Map Russian to English
        mic_map = {"перевод": "translated", "оригинал": "original", "оба": "both"}
        mic_val = mic_map.get(mic_mode, "translated")
        self.config.mic_mode = MicMode(mic_val)
        self._log(f"Режим микрофона: {mic_mode}")

    def _set_direction(self, direction: str):
        """Set translation direction manually."""
        self.dir_var.set(direction)
        
        if direction == "outgoing":
            # I speak Russian -> translate to English for partner
            self.outgoing_btn.configure(bg="#a6e3a1", fg="#1e1e2e")
            self.incoming_btn.configure(bg="#585b70", fg="#cdd6f4")
            self._log("Направление: Я → СОБЕСЕДНИК (RU→EN)", "outgoing")
            
            # Update engine if running
            if self.streams:
                self.streams.set_direction("outgoing")
        else:
            # Partner speaks English -> translate to Russian for me
            self.outgoing_btn.configure(bg="#585b70", fg="#cdd6f4")
            self.incoming_btn.configure(bg="#89b4fa", fg="#1e1e2e")
            self._log("Направление: СОБЕСЕДНИК → Я (EN→RU)", "incoming")
            
            # Update engine if running
            if self.streams:
                self.streams.set_direction("incoming")

    def _toggle_auto_detect(self):
        """Toggle automatic speaker detection."""
        if not hasattr(self, 'voice_profile_exists') or not self.voice_profile_exists:
            self._log("Сначала запишите голосовой профиль!", "error")
            self.auto_detect_var.set(False)
            return
        
        self.auto_detect_var.set(not self.auto_detect_var.get())
        
        if self.auto_detect_var.get():
            self.auto_detect_btn.configure(
                text="АВТООПРЕДЕЛЕНИЕ: ВКЛ",
                bg="#a6e3a1", fg="#1e1e2e"
            )
            self._log("Автоопределение говорящего включено", "outgoing")
            # Disable manual direction buttons
            self.outgoing_btn.configure(state="disabled")
            self.incoming_btn.configure(state="disabled")
            # Enable in streams
            if self.streams:
                self.streams.set_auto_detect(True)
        else:
            self.auto_detect_btn.configure(
                text="АВТООПРЕДЕЛЕНИЕ: ВЫКЛ",
                bg="#585b70", fg="#cdd6f4"
            )
            self._log("Автоопределение говорящего выключено", "system")
            # Enable manual direction buttons
            self.outgoing_btn.configure(state="normal")
            self.incoming_btn.configure(state="normal")
            # Disable in streams
            if self.streams:
                self.streams.set_auto_detect(False)

    def _start_enrollment(self):
        """Start voice enrollment directly in the app."""
        import threading
        import numpy as np
        import sounddevice as sd
        
        token = self.hf_token_var.get().strip()
        
        # Save token to config if provided
        if token:
            config_path = Path(__file__).parent.parent / "config" / "hf_token.txt"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(token)
            self._log("Hugging Face токен сохранён.", "system")
        
        # Disable enrollment button during recording
        self.enroll_btn.configure(state="disabled", text="ЗАПИСЬ...")
        self._log("Запись голоса 30 секунд... Говорите!", "system")
        
        def do_enrollment():
            try:
                # Use the same microphone as in config
                from .audio_devices import find_device
                mic_idx = find_device(self.config.input_device, kind="input")
                if mic_idx is None:
                    self.root.after(0, lambda: self._log("Микрофон не найден!", "error"))
                    self.root.after(0, lambda: self.enroll_btn.configure(state="normal", text="ЗАПИСЬ ГОЛОСА"))
                    return
                
                # Get device sample rate
                device_sr = int(sd.query_devices(mic_idx)["default_samplerate"])
                
                # Record 30 seconds at device sample rate, then resample
                duration = 30
                self.root.after(0, lambda: self._log(f"Запись с микрофона [{mic_idx}]...", "system"))
                
                audio = sd.rec(int(duration * device_sr), samplerate=device_sr, 
                              channels=1, dtype=np.float32, device=mic_idx)
                sd.wait()
                audio = audio.flatten()
                
                # Resample to 16kHz if needed
                if device_sr != 16000:
                    ratio = 16000 / device_sr
                    num_samples = int(len(audio) * ratio)
                    old_indices = np.arange(len(audio))
                    new_indices = np.linspace(0, len(audio) - 1, num_samples)
                    audio = np.interp(new_indices, old_indices, audio).astype(np.float32)
                
                # Check level
                rms = np.sqrt(np.mean(audio ** 2))
                self.root.after(0, lambda: self._log(f"Уровень звука: {rms:.4f}", "system"))
                
                if rms < 0.005:
                    self.root.after(0, lambda: self._log("Слишком тихо! Говорите громче.", "error"))
                    self.root.after(0, lambda: self.enroll_btn.configure(state="normal", text="ЗАПИСЬ ГОЛОСА"))
                    return
                
                # Create voice profile
                self.root.after(0, lambda: self._log("Создание профиля...", "system"))
                
                from .speaker_id import SpeakerIdentifier
                sid = SpeakerIdentifier(use_advanced=True)
                sid.enroll(audio, sample_rate=16000)
                
                self.root.after(0, lambda: self._log("✓ Голосовой профиль создан!", "outgoing"))
                self.root.after(0, lambda: self.enroll_btn.configure(state="normal", text="ЗАПИСЬ ГОЛОСА"))
                
            except Exception as e:
                self.root.after(0, lambda: self._log(f"Ошибка записи: {e}", "error"))
                self.root.after(0, lambda: self.enroll_btn.configure(state="normal", text="ЗАПИСЬ ГОЛОСА"))
        
        # Run in background thread
        threading.Thread(target=do_enrollment, daemon=True).start()


    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        if self.is_running:
            self._stop()
        self.root.destroy()


def main():
    app = TranslatorGUI()
    app.run()


if __name__ == "__main__":
    main()
