"""Voice Activity Detection using webrtcvad with smart merging."""
import collections
import time
import numpy as np
import webrtcvad

from .config import VAD_FRAME_MS, SILENCE_THRESHOLD_S, MIN_SPEECH_S


# If a new speech segment starts within this time after the previous one ended,
# merge them together. This prevents Whisper from getting tiny fragments.
MERGE_GAP_S = 1.0
# Maximum merged segment length (seconds). Force-emit if exceeded.
MAX_SEGMENT_S = 10.0


class VoiceActivityDetector:
    """Segments audio into speech chunks using WebRTC VAD.

    Uses smart merging: short segments that come in quick succession
    are merged into a single longer segment for better STT accuracy.

    State machine:
    - IDLE: waiting for speech onset
    - SPEAKING: accumulating speech, waiting for silence
    - MERGING: silence detected, but waiting to see if speech resumes quickly
    """

    def __init__(self, aggressiveness: int = 1, sample_rate: int = 16000):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration_ms = VAD_FRAME_MS
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)

        self.silence_frames = int(SILENCE_THRESHOLD_S * 1000 / self.frame_duration_ms)
        self.min_speech_frames = int(MIN_SPEECH_S * 1000 / self.frame_duration_ms)
        self._onset_window = max(3, self.silence_frames // 4)

        # Merge gap in frames
        self._merge_gap_frames = int(MERGE_GAP_S * 1000 / self.frame_duration_ms)
        # Max segment length in frames
        self._max_segment_frames = int(MAX_SEGMENT_S * 1000 / self.frame_duration_ms)

        self._ring = collections.deque(maxlen=self._onset_window)
        self._silence_ring = collections.deque(maxlen=self.silence_frames)
        self._triggered = False
        self._speech_buffer: list[bytes] = []
        self._speech_frame_count = 0

        # Merging state
        self._pending_segment: bytes | None = None  # completed segment waiting for merge
        self._pending_frame_count = 0
        self._silence_after_pending = 0  # frames of silence since pending was set

    def reset(self):
        self._ring.clear()
        self._silence_ring.clear()
        self._triggered = False
        self._speech_buffer = []
        self._speech_frame_count = 0
        self._pending_segment = None
        self._pending_frame_count = 0
        self._silence_after_pending = 0

    def process_chunk(self, audio_int16: np.ndarray) -> list[np.ndarray]:
        """
        Feed audio (int16, mono, at self.sample_rate).
        Returns list of complete speech segments (may be empty).
        """
        segments = []
        raw = audio_int16.tobytes()

        offset = 0
        frame_bytes = self.frame_size * 2

        while offset + frame_bytes <= len(raw):
            frame = raw[offset:offset + frame_bytes]
            offset += frame_bytes

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            # Track merge gap timeout
            if self._pending_segment is not None and not self._triggered:
                self._silence_after_pending += 1
                if self._silence_after_pending >= self._merge_gap_frames:
                    # Gap too long — emit the pending segment
                    if self._pending_frame_count >= self.min_speech_frames:
                        seg = np.frombuffer(self._pending_segment, dtype=np.int16)
                        segments.append(seg)
                    self._pending_segment = None
                    self._pending_frame_count = 0
                    self._silence_after_pending = 0

            if not self._triggered:
                # IDLE: looking for speech onset
                self._ring.append((frame, is_speech))
                num_voiced = sum(1 for _, s in self._ring if s)
                if num_voiced > 0.5 * len(self._ring) and len(self._ring) >= 2:
                    self._triggered = True
                    if self._pending_segment is not None:
                        # Merge: prepend pending segment + silence gap
                        self._speech_buffer = []
                        # Add the pending audio
                        pending_frames = len(self._pending_segment) // frame_bytes
                        for i in range(pending_frames):
                            s = i * frame_bytes
                            self._speech_buffer.append(
                                self._pending_segment[s:s + frame_bytes]
                            )
                        # Add silence frames that were in between
                        silence_frame = b'\x00' * frame_bytes
                        for _ in range(self._silence_after_pending):
                            self._speech_buffer.append(silence_frame)
                        # Add onset ring buffer
                        for f, _ in self._ring:
                            self._speech_buffer.append(f)
                        self._speech_frame_count = (
                            self._pending_frame_count
                            + self._silence_after_pending
                            + len(self._ring)
                        )
                        self._pending_segment = None
                        self._pending_frame_count = 0
                        self._silence_after_pending = 0
                    else:
                        self._speech_buffer = [f for f, _ in self._ring]
                        self._speech_frame_count = len(self._ring)
                    self._ring.clear()
                    self._silence_ring.clear()
            else:
                # SPEAKING: accumulating speech
                self._speech_buffer.append(frame)
                self._speech_frame_count += 1
                self._silence_ring.append(is_speech)

                # Force-emit if segment is too long
                if self._speech_frame_count >= self._max_segment_frames:
                    audio_bytes = b"".join(self._speech_buffer)
                    seg = np.frombuffer(audio_bytes, dtype=np.int16)
                    segments.append(seg)
                    self._triggered = False
                    self._speech_buffer = []
                    self._speech_frame_count = 0
                    self._ring.clear()
                    self._silence_ring.clear()
                    self._pending_segment = None
                    continue

                # Check if enough silence to end segment
                if len(self._silence_ring) >= self.silence_frames:
                    num_unvoiced = sum(1 for s in self._silence_ring if not s)
                    if num_unvoiced > 0.8 * self.silence_frames:
                        # End of speech — don't emit yet, hold for merging
                        audio_bytes = b"".join(self._speech_buffer)
                        if self._pending_segment is not None:
                            # Already have pending — this shouldn't happen
                            # but just in case, emit pending first
                            if self._pending_frame_count >= self.min_speech_frames:
                                seg = np.frombuffer(self._pending_segment, dtype=np.int16)
                                segments.append(seg)

                        self._pending_segment = audio_bytes
                        self._pending_frame_count = self._speech_frame_count
                        self._silence_after_pending = 0

                        self._triggered = False
                        self._speech_buffer = []
                        self._speech_frame_count = 0
                        self._ring.clear()
                        self._silence_ring.clear()

        return segments
