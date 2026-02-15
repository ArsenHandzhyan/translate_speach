"""Text-to-Speech engines."""
import logging
import subprocess
import tempfile
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)


class PiperTTS:
    """Local TTS using Piper."""

    def __init__(self, model_path: str, sample_rate: int = 22050):
        self.model_path = Path(model_path)
        self.config_path = self.model_path.with_suffix(".onnx.json")
        self.sample_rate = sample_rate
        self._piper_available = self._check_piper()

    def _check_piper(self) -> bool:
        try:
            subprocess.run(["piper", "--version"], capture_output=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            log.warning("Piper binary not found. Will use macOS 'say' as fallback.")
            return False

    def synthesize(self, text: str) -> np.ndarray | None:
        """Synthesize speech. Returns float32 audio array or None."""
        if not text.strip():
            return None

        if self._piper_available and self.model_path.exists():
            return self._piper_synth(text)
        else:
            return self._macos_say_synth(text)

    def _piper_synth(self, text: str) -> np.ndarray | None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name

            proc = subprocess.run(
                [
                    "piper",
                    "--model", str(self.model_path),
                    "--output_file", tmp_path,
                ],
                input=text,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode != 0:
                log.error(f"Piper error: {proc.stderr}")
                return self._macos_say_synth(text)

            import wave
            with wave.open(tmp_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                self.sample_rate = wf.getframerate()

            Path(tmp_path).unlink(missing_ok=True)
            return audio
        except Exception as e:
            log.error(f"Piper synthesis failed: {e}")
            return self._macos_say_synth(text)

    def _macos_say_synth(self, text: str) -> np.ndarray | None:
        """Fallback: use macOS 'say' command."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
                tmp_path = f.name

            subprocess.run(
                ["say", "-o", tmp_path, text],
                capture_output=True,
                timeout=30,
            )

            # Convert AIFF to raw PCM via ffmpeg
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name

            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", tmp_path,
                    "-ar", "22050", "-ac", "1", "-f", "wav", wav_path,
                ],
                capture_output=True,
                timeout=30,
            )

            import wave
            with wave.open(wav_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                self.sample_rate = wf.getframerate()

            Path(tmp_path).unlink(missing_ok=True)
            Path(wav_path).unlink(missing_ok=True)
            return audio
        except Exception as e:
            log.error(f"macOS say synthesis failed: {e}")
            return None


class MacOSSayTTS:
    """Simple TTS using macOS built-in 'say' command."""

    def __init__(self, voice_ru: str = "Milena", voice_en: str = "Samantha"):
        self.voices = {"ru": voice_ru, "en": voice_en}
        self.sample_rate = 22050

    def synthesize(self, text: str, lang: str = "ru") -> np.ndarray | None:
        if not text.strip():
            return None

        voice = self.voices.get(lang, self.voices["en"])
        try:
            with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
                tmp_aiff = f.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_wav = f.name

            subprocess.run(
                ["say", "-v", voice, "-o", tmp_aiff, "--", text],
                capture_output=True, timeout=30,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_aiff,
                 "-ar", str(self.sample_rate), "-ac", "1", "-f", "wav", tmp_wav],
                capture_output=True, timeout=30,
            )

            import wave
            with wave.open(tmp_wav, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

            Path(tmp_aiff).unlink(missing_ok=True)
            Path(tmp_wav).unlink(missing_ok=True)
            return audio
        except Exception as e:
            log.error(f"macOS say TTS failed: {e}")
            return None
