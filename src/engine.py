"""Main translation engine - orchestrates STT → MT → TTS pipeline."""
import logging
import re
import threading
import time
import numpy as np
from typing import Callable

from .config import TranslatorConfig, Mode, MicMode
from .stt import WhisperSTT
from .mt import NLLBTranslator, SimpleCloudTranslator
from .tts import MacOSSayTTS
from .vad import VoiceActivityDetector

log = logging.getLogger(__name__)


def split_into_chunks(text: str, max_words: int = 3) -> list[str]:
    """
    Split text into small chunks for incremental TTS.
    Each chunk contains up to max_words words.
    Preserves punctuation at the end of chunks.
    """
    if not text.strip():
        return []

    # Split by sentence boundaries first, then by word count
    sentences = re.split(r'([.!?]+)', text)
    chunks = []

    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        # Get punctuation if present
        punct = sentences[i + 1] if i + 1 < len(sentences) else ""

        if not sentence:
            i += 2 if punct else 1
            continue

        words = sentence.split()

        # Split long sentences into word chunks
        for j in range(0, len(words), max_words):
            chunk_words = words[j:j + max_words]
            chunk = " ".join(chunk_words)

            # Add punctuation only to the last chunk of the sentence
            if j + max_words >= len(words):
                chunk += punct

            chunks.append(chunk)

        i += 2 if punct else 1

    return chunks


class TranslationEngine:
    """Bidirectional speech-to-speech translation engine."""

    def __init__(self, config: TranslatorConfig):
        self.config = config
        self._running = False
        self._lock = threading.Lock()

        # Callbacks
        self.on_subtitle: Callable[[str, str, str], None] | None = None  # (direction, original, translated)
        self.on_status: Callable[[str], None] | None = None

        # Components (lazy-loaded)
        self._stt: WhisperSTT | None = None
        self._mt_local: NLLBTranslator | None = None
        self._mt_cloud: SimpleCloudTranslator | None = None
        self._tts: MacOSSayTTS | None = None
        self._vad_incoming: VoiceActivityDetector | None = None
        self._vad_outgoing: VoiceActivityDetector | None = None

    def _status(self, msg: str):
        log.info(msg)
        if self.on_status:
            self.on_status(msg)

    def load_models(self):
        """Load all models for the current mode."""
        self._status("Loading models...")

        # VAD always needed
        self._vad_incoming = VoiceActivityDetector(aggressiveness=2, sample_rate=16000)
        self._vad_outgoing = VoiceActivityDetector(aggressiveness=2, sample_rate=16000)

        mode = self.config.mode

        if mode in (Mode.LOCAL, Mode.HYBRID):
            self._status("Loading Whisper STT...")
            self._stt = WhisperSTT(
                model_size_or_path=self.config.whisper_model,
                device="cpu",
                compute_type="int8",
            )

            # Try loading NLLB, fall back to cloud MT
            from pathlib import Path
            if Path(self.config.nllb_model_path).exists():
                self._status("Loading NLLB MT...")
                try:
                    self._mt_local = NLLBTranslator(self.config.nllb_model_path)
                except Exception as e:
                    log.warning(f"Failed to load NLLB: {e}. Will use cloud MT.")
                    self._mt_local = None

            self._status("Loading TTS...")
            self._tts = MacOSSayTTS()

        if mode in (Mode.CLOUD, Mode.HYBRID):
            self._mt_cloud = SimpleCloudTranslator()
            if mode == Mode.CLOUD:
                # Cloud mode still needs STT and TTS locally for now
                if self._stt is None:
                    self._stt = WhisperSTT(
                        model_size_or_path=self.config.whisper_model,
                        device="cpu",
                        compute_type="int8",
                    )
                if self._tts is None:
                    self._tts = MacOSSayTTS()

        self._status("Models loaded.")

    def translate_speech(self, audio_int16: np.ndarray, src_lang: str, tgt_lang: str) -> tuple[np.ndarray | None, str, str]:
        """
        Full pipeline: audio_int16 (16kHz mono) → STT → MT → TTS → audio_float32.
        Returns (tts_audio_float32, original_text, translated_text).
        """
        if self._stt is None:
            return None, "", ""

        # STT
        audio_f32 = audio_int16.astype(np.float32) / 32768.0
        text = self._stt.transcribe(audio_f32, language=src_lang)
        if not text.strip():
            return None, "", ""

        log.info(f"STT [{src_lang}]: {text}")

        # MT
        translated = self._translate_text(text, src_lang, tgt_lang)
        if not translated.strip():
            return None, text, ""

        log.info(f"MT  [{src_lang}→{tgt_lang}]: '{text}' → '{translated}'")

        # TTS - synthesize full text at once
        tts_audio = self._tts.synthesize(translated, lang=tgt_lang)

        return tts_audio, text, translated

    def translate_speech_chunked(self, audio_int16: np.ndarray, src_lang: str, tgt_lang: str) -> list[tuple[np.ndarray | None, str, str]]:
        """
        Full pipeline with chunked TTS: audio → STT → MT → split into chunks → TTS per chunk.
        Returns list of (tts_audio_float32, original_chunk, translated_chunk) for incremental playback.
        """
        if self._stt is None:
            return []

        # STT
        audio_f32 = audio_int16.astype(np.float32) / 32768.0
        text = self._stt.transcribe(audio_f32, language=src_lang)
        if not text.strip():
            return []

        # MT - use cloud MT directly to avoid duplicate logging from hybrid mode
        translated = self._translate_text_once(text, src_lang, tgt_lang)
        if not translated.strip():
            return []

        log.info(f"STT [{src_lang}]: {text}")
        log.info(f"MT  [{src_lang}→{tgt_lang}]: '{text}' → '{translated}'")

        # Split translated text into small chunks
        chunk_size = getattr(self.config, 'tts_chunk_words', 3)
        chunks = split_into_chunks(translated, max_words=chunk_size)
        log.info(f"TTS chunks: {chunks}")

        results = []
        for chunk in chunks:
            # TTS per chunk for incremental playback
            tts_audio = self._tts.synthesize(chunk, lang=tgt_lang)
            results.append((tts_audio, text, chunk))

        return results

    def _translate_text_once(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate text once without duplicate logging."""
        mode = self.config.mode

        # Try cloud first if available (avoids duplicate logs from hybrid fallback)
        if self._mt_cloud:
            try:
                result = self._mt_cloud.translate(text, src_lang, tgt_lang)
                if result and result != text:
                    return result
            except Exception:
                pass

        # Fallback to local MT
        if self._mt_local:
            return self._mt_local.translate(text, src_lang, tgt_lang)

        # Ultimate fallback
        if self._mt_cloud is None:
            self._mt_cloud = SimpleCloudTranslator()
        return self._mt_cloud.translate(text, src_lang, tgt_lang)

    def _translate_text(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate text using available MT engine."""
        mode = self.config.mode

        if mode == Mode.LOCAL and self._mt_local:
            return self._mt_local.translate(text, src_lang, tgt_lang)
        elif mode == Mode.CLOUD and self._mt_cloud:
            return self._mt_cloud.translate(text, src_lang, tgt_lang)
        elif mode == Mode.HYBRID:
            # Try cloud first, fall back to local
            if self._mt_cloud:
                try:
                    result = self._mt_cloud.translate(text, src_lang, tgt_lang)
                    if result and result != text:
                        return result
                except Exception:
                    pass
            if self._mt_local:
                return self._mt_local.translate(text, src_lang, tgt_lang)

        # Ultimate fallback: cloud translator
        if self._mt_cloud is None:
            self._mt_cloud = SimpleCloudTranslator()
        return self._mt_cloud.translate(text, src_lang, tgt_lang)

    def process_incoming_chunk(self, audio_int16_16k: np.ndarray) -> list[tuple[np.ndarray | None, str, str]]:
        """
        Process incoming audio chunk (from browser/remote speaker, EN).
        Returns list of (tts_audio, original_text, translated_text) for complete speech segments.
        """
        results = []
        segments = self._vad_incoming.process_chunk(audio_int16_16k)
        for segment in segments:
            result = self.translate_speech(segment, src_lang="en", tgt_lang="ru")
            results.append(result)
            if self.on_subtitle and result[1]:
                self.on_subtitle("incoming", result[1], result[2])
        return results

    def process_outgoing_chunk(self, audio_int16_16k: np.ndarray) -> list[tuple[np.ndarray | None, str, str]]:
        """
        Process outgoing audio chunk (from my microphone, RU).
        Returns list of (tts_audio, original_text, translated_text) for complete speech segments.
        """
        results = []
        segments = self._vad_outgoing.process_chunk(audio_int16_16k)
        for segment in segments:
            result = self.translate_speech(segment, src_lang="ru", tgt_lang="en")
            results.append(result)
            if self.on_subtitle and result[1]:
                self.on_subtitle("outgoing", result[1], result[2])
        return results
