"""Speaker identification using pyannote.audio embeddings."""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

log = logging.getLogger(__name__)

# Profile storage
PROFILE_PATH = Path(__file__).parent.parent / "config" / "voice_profile.npy"


class SpeakerIdentifier:
    """Identify speakers using voice embeddings."""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.my_embedding: Optional[np.ndarray] = None
        self.model = None
        self._load_profile()
        
    def _load_profile(self):
        """Load enrolled voice profile if exists."""
        if PROFILE_PATH.exists():
            self.my_embedding = np.load(PROFILE_PATH)
            log.info(f"Loaded voice profile from {PROFILE_PATH}")
        else:
            log.warning("No voice profile found. Run enrollment first.")
    
    def load_model(self):
        """Lazy load pyannote model."""
        if self.model is None:
            log.info("Loading speaker embedding model...")
            from pyannote.audio import Model
            self.model = Model.from_pretrained(
                "pyannote/embedding",
                use_auth_token=False
            )
            self.model.eval()
            log.info("Model loaded.")
    
    def extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract voice embedding from audio.
        
        Args:
            audio: Audio array (float32, mono)
            sample_rate: Sample rate (default 16000)
            
        Returns:
            512-dimensional embedding vector
        """
        self.load_model()
        
        # Ensure correct format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Convert to torch tensor
        waveform = torch.from_numpy(audio).unsqueeze(0)
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(waveform)
        
        return embedding.numpy().flatten()
    
    def enroll(self, audio: np.ndarray, sample_rate: int = 16000):
        """Enroll user's voice profile.
        
        Args:
            audio: 30-60 seconds of user's speech
            sample_rate: Sample rate
        """
        log.info("Creating voice profile...")
        self.my_embedding = self.extract_embedding(audio, sample_rate)
        
        # Save profile
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(PROFILE_PATH, self.my_embedding)
        log.info(f"Voice profile saved to {PROFILE_PATH}")
    
    def identify(self, audio: np.ndarray, sample_rate: int = 16000) -> tuple[bool, float]:
        """Identify if the speaker is you.
        
        Args:
            audio: Audio segment to identify
            sample_rate: Sample rate
            
        Returns:
            (is_me, confidence) tuple
            - is_me: True if identified as you
            - confidence: Similarity score (0-1)
        """
        if self.my_embedding is None:
            log.error("No voice profile. Run enrollment first.")
            return False, 0.0
        
        # Extract embedding
        embedding = self.extract_embedding(audio, sample_rate)
        
        # Calculate cosine similarity
        similarity = self._cosine_similarity(embedding, self.my_embedding)
        
        is_me = similarity > self.threshold
        
        log.debug(f"Speaker similarity: {similarity:.3f}, is_me: {is_me}")
        
        return is_me, similarity
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def is_enrolled(self) -> bool:
        """Check if voice profile exists."""
        return self.my_embedding is not None


class SpeakerAwarePipeline:
    """Translation pipeline with automatic speaker detection."""
    
    def __init__(self, engine, speaker_id: SpeakerIdentifier):
        self.engine = engine
        self.speaker_id = speaker_id
        
    def process(self, audio_int16: np.ndarray) -> tuple[np.ndarray | None, str, str, str]:
        """Process audio with speaker identification.
        
        Args:
            audio_int16: Audio data (16kHz, int16)
            
        Returns:
            (tts_audio, original, translated, direction)
            - direction: "outgoing" (me) or "incoming" (partner)
        """
        # Convert to float32 for speaker ID
        audio_f32 = audio_int16.astype(np.float32) / 32768.0
        
        # Identify speaker
        if self.speaker_id.is_enrolled():
            is_me, confidence = self.speaker_id.identify(audio_f32)
            log.info(f"Speaker identified: {'me' if is_me else 'other'} (confidence: {confidence:.3f})")
        else:
            # No profile - assume outgoing (backward compatibility)
            log.warning("No voice profile, assuming outgoing")
            is_me = True
        
        # Determine translation direction
        if is_me:
            # I speak Russian -> translate to English for partner
            src_lang, tgt_lang = "ru", "en"
            direction = "outgoing"
        else:
            # Partner speaks English -> translate to Russian for me
            src_lang, tgt_lang = "en", "ru"
            direction = "incoming"
        
        # Translate
        results = self.engine.translate_speech_chunked(
            audio_int16,
            src_lang=src_lang,
            tgt_lang=tgt_lang
        )
        
        if not results:
            return None, "", "", direction
        
        # Combine audio chunks
        audio_chunks = [r[0] for r in results if r[0] is not None]
        if audio_chunks:
            combined_audio = np.concatenate(audio_chunks)
        else:
            combined_audio = None
        
        # Get text
        original = results[0][1] if results else ""
        translated = " ".join(r[2] for r in results if r[2])
        
        return combined_audio, original, translated, direction
