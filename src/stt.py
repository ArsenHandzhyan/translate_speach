"""Speech-to-Text engines."""
import logging
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)


class WhisperSTT:
    """Local STT using faster-whisper (CTranslate2 backend, optimized for Apple Silicon)."""

    def __init__(self, model_size_or_path: str = "small", device: str = "cpu",
                 compute_type: str = "int8"):
        from faster_whisper import WhisperModel
        log.info(f"Loading Whisper model: {model_size_or_path} on {device}/{compute_type}")
        self.model = WhisperModel(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
        )
        log.info("Whisper model loaded.")

    def transcribe(self, audio_float32: np.ndarray, language: str = None) -> str:
        """
        Transcribe audio.
        audio_float32: mono float32 array, 16kHz sample rate.
        language: 'en', 'ru', or None for auto-detect.
        Returns transcribed text.
        """
        segments, info = self.model.transcribe(
            audio_float32,
            language=language,
            beam_size=3,
            best_of=3,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        text = " ".join(seg.text.strip() for seg in segments)
        if text:
            log.info(f"STT [{info.language}]: {text}")
        return text

    def detect_language(self, audio_float32: np.ndarray) -> str:
        """Detect language of audio. Returns language code."""
        _, info = self.model.transcribe(audio_float32, beam_size=1)
        return info.language
