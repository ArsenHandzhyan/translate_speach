"""Audio stream management - continuous recording with processing queue."""
import logging
import threading
import queue
import time
import numpy as np
import sounddevice as sd
from scipy import signal as scipy_signal

from .config import TranslatorConfig, MicMode, SAMPLE_RATE
from .audio_devices import find_device
from .engine import TranslationEngine
from .vad import VoiceActivityDetector

log = logging.getLogger(__name__)


def resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return audio
    num_samples = int(len(audio) * to_sr / from_sr)
    return scipy_signal.resample(audio, num_samples).astype(audio.dtype)


class AudioStreamManager:
    """
    Manages audio streams for bidirectional translation.

    Architecture (per direction):
      1. Recorder thread: reads mic continuously, runs VAD, puts COMPLETE
         speech segments into a processing queue. Never blocks.
      2. Pipeline thread: takes segments from queue, runs STT→MT→TTS,
         puts TTS audio into playback queue.
      3. Playback thread: plays TTS audio sequentially to output device.

    Anti-echo: mic recorder is muted whenever ANY TTS plays through speakers
    (both incoming translation playback and outgoing preview).
    """

    def __init__(self, config: TranslatorConfig, engine: TranslationEngine):
        self.config = config
        self.engine = engine
        self._running = False
        self._streams: list = []
        self._threads: list[threading.Thread] = []
        self.preview_enabled = False
        # Mute mic when TTS plays through speakers (anti-echo)
        self._mute_recorder = False
        self._mute_lock = threading.Lock()
        self._mute_count = 0  # reference counting for nested mutes
        # Manual direction control
        self._manual_direction = "outgoing"  # "outgoing" or "incoming"
        self._direction_lock = threading.Lock()
        # Auto-detect speaker
        self._auto_detect_enabled = False
        self._speaker_id = None

    def _mute(self):
        """Mute the mic recorder (reference counted for nested calls)."""
        with self._mute_lock:
            self._mute_count += 1
            self._mute_recorder = True

    def _unmute(self):
        """Unmute the mic recorder (only when all mute sources release)."""
        with self._mute_lock:
            self._mute_count = max(0, self._mute_count - 1)
            if self._mute_count == 0:
                self._mute_recorder = False

    def set_direction(self, direction: str):
        """Set translation direction manually.
        
        Args:
            direction: "outgoing" (me→partner, RU→EN) or "incoming" (partner→me, EN→RU)
        """
        with self._direction_lock:
            self._manual_direction = direction
            log.info(f"Direction set to: {direction}")

    def get_direction(self) -> str:
        """Get current translation direction."""
        with self._direction_lock:
            return self._manual_direction

    def set_auto_detect(self, enabled: bool):
        """Enable or disable automatic speaker detection."""
        self._auto_detect_enabled = enabled
        if enabled and self._speaker_id is None:
            from .speaker_id import SpeakerIdentifier
            self._speaker_id = SpeakerIdentifier(use_advanced=True)
            if not self._speaker_id.is_enrolled():
                log.warning("Auto-detect enabled but no voice profile!")
                self._auto_detect_enabled = False
                return
            # Pre-load advanced model if available
            if self._speaker_id._hf_token:
                log.info("Pre-loading pyannote.audio model...")
                self._speaker_id._load_advanced_model()
                if self._speaker_id._use_advanced:
                    log.info("pyannote.audio loaded successfully!")
                else:
                    log.warning("Using MFCC mode (pyannote failed to load)")
        log.info(f"Auto speaker detection: {'enabled' if enabled else 'disabled'}")

    def is_auto_detect_enabled(self) -> bool:
        return self._auto_detect_enabled

    def start(self):
        self._running = True
        log.info("Starting audio streams...")

        mic_idx = find_device(self.config.input_device, kind="input")
        headphone_idx = find_device(self.config.output_device, kind="output")
        blackhole_out_idx = find_device("BlackHole 2ch", kind="output")

        if mic_idx is None:
            raise RuntimeError(f"Microphone not found: {self.config.input_device}")
        if headphone_idx is None:
            raise RuntimeError(f"Output device not found: {self.config.output_device}")
        if blackhole_out_idx is None:
            raise RuntimeError("BlackHole 2ch not found. Install: brew install blackhole-2ch")

        log.info(f"Mic: [{mic_idx}] {sd.query_devices(mic_idx)['name']}")
        log.info(f"Output: [{headphone_idx}] {sd.query_devices(headphone_idx)['name']}")
        log.info(f"BlackHole OUT: [{blackhole_out_idx}] {sd.query_devices(blackhole_out_idx)['name']}")

        self._headphone_idx = headphone_idx

        # --- OUTGOING pipeline ---
        self._start_pipeline(
            capture_idx=mic_idx,
            output_idx=blackhole_out_idx,
            direction="outgoing",
            src_lang="ru",
            tgt_lang="en",
        )

        # --- INCOMING pipeline (if BlackHole 16ch available) ---
        bh16_idx = find_device("BlackHole 16ch", kind="input")
        if bh16_idx is not None:
            log.info(f"BlackHole 16ch found — enabling incoming translation")
            self._start_pipeline(
                capture_idx=bh16_idx,
                output_idx=headphone_idx,
                direction="incoming",
                src_lang="en",
                tgt_lang="ru",
            )
        else:
            log.info("BlackHole 16ch not found. Incoming translation disabled.")

        log.info("Audio streams started.")

    def _start_pipeline(self, capture_idx: int, output_idx: int,
                        direction: str, src_lang: str, tgt_lang: str):
        segment_q = queue.Queue(maxsize=100)
        playback_q = queue.Queue(maxsize=100)

        t_rec = threading.Thread(
            target=self._recorder_loop,
            args=(capture_idx, segment_q, direction),
            daemon=True, name=f"{direction}-recorder",
        )
        t_proc = threading.Thread(
            target=self._pipeline_loop,
            args=(segment_q, playback_q, direction, src_lang, tgt_lang),
            daemon=True, name=f"{direction}-pipeline",
        )
        t_play = threading.Thread(
            target=self._playback_loop,
            args=(output_idx, playback_q, direction),
            daemon=True, name=f"{direction}-playback",
        )

        for t in (t_rec, t_proc, t_play):
            t.start()
            self._threads.append(t)

    def stop(self):
        self._running = False
        for s in self._streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self._streams.clear()
        for t in self._threads:
            t.join(timeout=3)
        self._threads.clear()
        log.info("Audio streams stopped.")

    def _recorder_loop(self, device_idx: int, segment_queue: queue.Queue,
                       direction: str = "outgoing"):
        """
        Continuously reads audio, runs VAD, puts COMPLETE speech segments
        into the queue. Never blocks on downstream processing.
        For outgoing: muted when TTS plays through speakers (anti-echo).
        For incoming: never muted (reads from BlackHole, not mic).
        """
        device_sr = int(sd.query_devices(device_idx)["default_samplerate"])
        chunk_duration = 0.03
        chunk_size = int(device_sr * chunk_duration)
        device_name = sd.query_devices(device_idx)["name"]

        vad = VoiceActivityDetector(aggressiveness=1, sample_rate=16000)

        vad_accum = np.array([], dtype=np.float32)
        vad_accum_target = int(device_sr * 0.1)

        # Only outgoing recorder should be muted (it reads from physical mic)
        should_check_mute = (direction == "outgoing")

        log.info(f"Recorder: [{device_idx}] {device_name} sr={device_sr}")

        try:
            stream = sd.InputStream(
                device=device_idx, samplerate=device_sr,
                channels=1, dtype="float32", blocksize=chunk_size,
            )
            stream.start()
            self._streams.append(stream)

            while self._running:
                data, overflowed = stream.read(chunk_size)
                if overflowed:
                    log.debug("Overflow: %s", device_name)

                # Mute outgoing mic when TTS plays through speakers
                if should_check_mute and self._mute_recorder:
                    vad_accum = np.array([], dtype=np.float32)
                    vad.reset()
                    continue

                mono = data[:, 0] if data.ndim > 1 else data.flatten()
                vad_accum = np.concatenate([vad_accum, mono])

                if len(vad_accum) >= vad_accum_target:
                    audio_16k = resample(vad_accum, device_sr, 16000)
                    audio_int16 = (audio_16k * 32768).clip(-32768, 32767).astype(np.int16)
                    vad_accum = np.array([], dtype=np.float32)

                    segments = vad.process_chunk(audio_int16)
                    for seg in segments:
                        # Speaker ID check for outgoing (mic) when auto-detect is enabled
                        if direction == "outgoing" and self._auto_detect_enabled and self._speaker_id:
                            is_me, confidence = self._speaker_id.identify(seg.astype(np.float32) / 32768.0, 16000)
                            if not is_me:
                                log.debug(f"Ignoring segment - not my voice (confidence: {confidence:.2f})")
                                continue
                            log.debug(f"Accepting segment - my voice (confidence: {confidence:.2f})")
                        
                        try:
                            segment_queue.put_nowait(seg)
                        except queue.Full:
                            log.warning("Segment queue full, dropping segment")

        except Exception as e:
            log.error(f"Recorder error: {e}", exc_info=True)
        finally:
            log.info(f"Recorder ended: {device_name}")

    def _pipeline_loop(self, segment_queue: queue.Queue, playback_queue: queue.Queue,
                       direction: str, src_lang: str, tgt_lang: str):
        log.info(f"Pipeline started: {direction} ({src_lang}→{tgt_lang})")

        while self._running:
            try:
                segment = segment_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                # Use chunked translation for incremental TTS playback
                results = self.engine.translate_speech_chunked(
                    segment, src_lang=src_lang, tgt_lang=tgt_lang
                )

                if results:
                    # Show full translation in subtitle (last chunk has complete text)
                    full_translated = " ".join(r[2] for r in results if r[2])
                    full_original = results[0][1] if results else ""

                    if full_original.strip():
                        if self.engine.on_subtitle:
                            self.engine.on_subtitle(direction, full_original, full_translated)

                    # Queue each chunk for sequential playback
                    for tts_audio, orig, chunk in results:
                        if tts_audio is not None:
                            try:
                                playback_queue.put(tts_audio, timeout=10)
                            except queue.Full:
                                log.warning("Playback queue full")

            except Exception as e:
                log.error(f"Pipeline error: {e}", exc_info=True)

        log.info(f"Pipeline ended: {direction}")

    def _playback_loop(self, device_idx: int, playback_queue: queue.Queue,
                       direction: str = "outgoing"):
        """
        Plays TTS audio sequentially.
        For outgoing: sends to BlackHole (+ preview to speakers if enabled).
        For incoming: sends to speakers (mutes mic during playback to prevent echo).
        """
        device_sr = int(sd.query_devices(device_idx)["default_samplerate"])
        device_name = sd.query_devices(device_idx)["name"]
        log.info(f"Playback: [{device_idx}] {device_name} sr={device_sr}")

        # Incoming plays to speakers — must mute mic during playback
        plays_to_speakers = (direction == "incoming")

        # For outgoing preview: open a separate stream to speakers
        preview_stream = None
        preview_sr = None
        if direction == "outgoing" and hasattr(self, '_headphone_idx'):
            try:
                preview_sr = int(sd.query_devices(self._headphone_idx)["default_samplerate"])
                preview_stream = sd.OutputStream(
                    device=self._headphone_idx, samplerate=preview_sr,
                    channels=1, dtype="float32",
                )
                preview_stream.start()
                self._streams.append(preview_stream)
                log.info(f"Preview stream ready on speakers [{self._headphone_idx}]")
            except Exception as e:
                log.warning(f"Could not open preview stream: {e}")
                preview_stream = None

        try:
            out_stream = sd.OutputStream(
                device=device_idx, samplerate=device_sr,
                channels=1, dtype="float32",
            )
            out_stream.start()
            self._streams.append(out_stream)

            while self._running:
                try:
                    tts_audio = playback_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                tts_sr = self.engine._tts.sample_rate

                # Determine if we need to mute mic
                do_preview = self.preview_enabled and preview_stream is not None
                need_mute = plays_to_speakers or do_preview

                if need_mute:
                    self._mute()

                try:
                    resampled = resample(tts_audio, tts_sr, device_sr)
                    try:
                        out_stream.write(resampled.reshape(-1, 1))
                    except Exception as e:
                        log.error(f"Playback write error: {e}")

                    # Preview: also play to speakers
                    if do_preview:
                        try:
                            preview_resampled = resample(tts_audio, tts_sr, preview_sr)
                            preview_stream.write(preview_resampled.reshape(-1, 1))
                        except Exception as e:
                            log.debug(f"Preview write error: {e}")

                    # Wait for reverb to decay
                    if need_mute:
                        time.sleep(0.4)
                finally:
                    if need_mute:
                        self._unmute()

        except Exception as e:
            log.error(f"Playback error: {e}", exc_info=True)
        finally:
            log.info(f"Playback ended: {device_name}")
