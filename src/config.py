"""Configuration for the live translator."""
import os
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


class Mode(Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    HYBRID = "hybrid"


class MicMode(Enum):
    TRANSLATED = "translated"
    ORIGINAL = "original"
    BOTH = "both"


SAMPLE_RATE = 48000
CHANNELS = 1
BLOCK_SIZE = 1024
VAD_FRAME_MS = 30  # 10, 20 or 30 ms for webrtcvad
SILENCE_THRESHOLD_S = 0.6  # seconds of silence to trigger end-of-speech
MIN_SPEECH_S = 0.15  # minimum speech duration to process (short words like numbers)


@dataclass
class TranslatorConfig:
    mode: Mode = Mode.LOCAL
    mic_mode: MicMode = MicMode.TRANSLATED
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS

    # Device names (will be resolved to indices at runtime)
    # Uses substring matching — works with both English and Russian macOS locale
    input_device: str = "MacBook Pro"  # physical mic ("Микрофон MacBook Pro" or "MacBook Pro Microphone")
    output_device: str = "MacBook Pro"  # speakers/headphones ("Динамики MacBook Pro" or "MacBook Pro Speakers")
    browser_capture_device: str = "BlackHole 2ch"  # captures browser output
    virtual_mic_device: str = "BlackHole 2ch"  # what browser sees as mic

    # LOCAL mode model paths
    whisper_model: str = "small"  # tiny, base, small, medium, large-v3
    whisper_model_path: str = str(MODELS_DIR / "whisper-small-ct2")
    piper_voice_ru: str = str(MODELS_DIR / "piper-ru" / "ru_RU-irina-medium.onnx")
    piper_voice_en: str = str(MODELS_DIR / "piper-en" / "en_US-amy-medium.onnx")
    nllb_model_path: str = str(MODELS_DIR / "nllb-200-distilled-600M-ct2")

    # CLOUD mode
    cloud_provider: str = "google"  # placeholder

    # Audio
    push_to_talk: bool = False
    push_to_talk_key: str = "ctrl"

    # TTS chunking - split translation into small chunks for incremental playback
    # Lower = more responsive but more TTS calls, higher = less responsive but more efficient
    tts_chunk_words: int = 3  # words per TTS chunk

    # Logging
    log_file: str = str(LOGS_DIR / "translator.log")
    show_subtitles: bool = True
