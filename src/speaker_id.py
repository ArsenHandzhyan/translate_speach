"""Speaker identification using MFCC or pyannote.audio (with HF token)."""
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Profile storage
PROFILE_PATH = Path(__file__).parent.parent / "config" / "voice_profile.npy"
HF_TOKEN_PATH = Path(__file__).parent.parent / "config" / "hf_token.txt"


def get_hf_token() -> Optional[str]:
    """Get Hugging Face token if saved."""
    if HF_TOKEN_PATH.exists():
        return HF_TOKEN_PATH.read_text().strip()
    return None


class SpeakerIdentifier:
    """Identify speakers using voice embeddings.
    
    Supports two backends:
    - MFCC (basic, no dependencies) - less accurate
    - pyannote.audio (advanced, requires HF token) - more accurate
    """
    
    def __init__(self, threshold: float = 0.65, use_advanced: bool = False):
        self.threshold = threshold
        self.my_embedding: Optional[np.ndarray] = None
        self._model = None
        self._use_advanced = use_advanced
        self._hf_token = get_hf_token()
        self._load_profile()
        
    def _load_profile(self):
        """Load enrolled voice profile if exists."""
        if PROFILE_PATH.exists():
            self.my_embedding = np.load(PROFILE_PATH)
            log.info(f"Loaded voice profile from {PROFILE_PATH}")
        else:
            log.warning("No voice profile found. Run enrollment first.")
    
    def _load_advanced_model(self):
        """Load pyannote.audio model if HF token available."""
        if self._model is None:
            if not self._hf_token:
                log.warning("No Hugging Face token. Using MFCC (basic) mode.")
                self._use_advanced = False
                return
            
            try:
                log.info("Loading pyannote.audio model (advanced mode)...")
                from pyannote.audio import Model, Inference
                self._model = Model.from_pretrained(
                    "pyannote/embedding",
                    token=self._hf_token
                )
                self._inference = Inference(
                    self._model,
                    window="whole"
                )
                self._use_advanced = True
                log.info("pyannote.audio model loaded successfully!")
            except Exception as e:
                log.error(f"Failed to load pyannote.audio: {e}")
                log.warning("Falling back to MFCC (basic) mode.")
                self._use_advanced = False
    
    def extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract voice embedding from audio.
        
        Args:
            audio: Audio array (float32, mono)
            sample_rate: Sample rate (default 16000)
            
        Returns:
            Feature vector (78-dim for MFCC, 512-dim for pyannote)
        """
        # Ensure correct format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Resample if needed (using numpy to avoid scipy mutex issues)
        if sample_rate != 16000:
            ratio = 16000 / sample_rate
            num_samples = int(len(audio) * ratio)
            old_indices = np.arange(len(audio))
            new_indices = np.linspace(0, len(audio) - 1, num_samples)
            audio = np.interp(new_indices, old_indices, audio).astype(np.float32)
        
        # Try advanced mode if enabled
        if self._use_advanced:
            self._load_advanced_model()
            
            if self._use_advanced and self._model is not None:
                return self._extract_pyannote(audio)
        
        # Fallback to MFCC
        return self._extract_mfcc(audio)
    
    def _extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features (basic mode)."""
        import librosa
        
        # Get MFCCs (13 coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
        
        # Get delta and delta-delta for better speaker characterization
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Combine features
        features = np.vstack([mfcc, delta, delta2])
        
        # Compute statistics (mean and std over time)
        mean = np.mean(features, axis=1)
        std = np.std(features, axis=1)
        
        return np.concatenate([mean, std])
    
    def _extract_pyannote(self, audio: np.ndarray) -> np.ndarray:
        """Extract pyannote.audio embedding (advanced mode)."""
        try:
            embedding = self._inference(audio[np.newaxis, :])
            return embedding.flatten()
        except Exception as e:
            log.error(f"pyannote extraction failed: {e}")
            return self._extract_mfcc(audio)
    
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
