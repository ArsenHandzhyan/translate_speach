"""Speaker-aware translator - integrates Speaker ID with existing pipeline."""
import logging
from typing import Callable

import numpy as np

from .config import TranslatorConfig
from .engine import TranslationEngine
from .speaker_id import SpeakerIdentifier, SpeakerAwarePipeline
from .streams import AudioStreamManager, resample

log = logging.getLogger(__name__)


class SpeakerAwareAudioManager(AudioStreamManager):
    """Audio manager with automatic speaker identification."""
    
    def __init__(self, config: TranslatorConfig, engine: TranslationEngine):
        super().__init__(config, engine)
        self.speaker_id = SpeakerIdentifier()
        self.speaker_pipeline = SpeakerAwarePipeline(engine, self.speaker_id)
        
    def _pipeline_loop(self, segment_queue, playback_queue, direction, src_lang, tgt_lang):
        """Override pipeline to use speaker-aware processing."""
        log.info(f"Speaker-aware pipeline started: {direction}")
        
        while self._running:
            try:
                segment = segment_queue.get(timeout=0.5)
            except:
                continue
            
            try:
                # Use speaker-aware pipeline for automatic direction detection
                tts_audio, orig, translated, detected_direction = self.speaker_pipeline.process(segment)
                
                if orig.strip():
                    log.info(f"Detected speaker: {detected_direction}")
                    
                    if self.engine.on_subtitle:
                        self.engine.on_subtitle(detected_direction, orig, translated)
                    
                    # Route audio based on detected speaker
                    if detected_direction == "outgoing":
                        # My speech -> send to BlackHole (partner hears)
                        if tts_audio is not None:
                            try:
                                playback_queue.put(tts_audio, timeout=10)
                            except:
                                log.warning("Playback queue full")
                    else:
                        # Partner speech -> send to headphones (I hear)
                        if tts_audio is not None:
                            try:
                                # Use separate queue for incoming to route to headphones
                                self._play_to_headphones(tts_audio)
                            except:
                                log.warning("Headphones playback failed")
                                
            except Exception as e:
                log.error(f"Pipeline error: {e}", exc_info=True)
    
    def _play_to_headphones(self, audio: np.ndarray):
        """Play audio to headphones (for incoming translation)."""
        # This would need separate output stream to headphones
        # For now, use the same playback mechanism
        pass


def create_speaker_aware_translator(
    config: TranslatorConfig,
    on_subtitle: Callable[[str, str, str], None] = None
):
    """Create translator with speaker identification.
    
    Usage:
        from src.speaker_aware_translator import create_speaker_aware_translator
        
        translator = create_speaker_aware_translator(config)
        translator.start()
    """
    from .engine import TranslationEngine
    
    # Check if voice profile exists
    speaker_id = SpeakerIdentifier()
    if not speaker_id.is_enrolled():
        log.warning("=" * 60)
        log.warning("NO VOICE PROFILE FOUND!")
        log.warning("Run enrollment first: python -m src.enrollment")
        log.warning("=" * 60)
    
    # Create engine
    engine = TranslationEngine(config)
    engine.load_models()
    
    if on_subtitle:
        engine.on_subtitle = on_subtitle
    
    # Create speaker-aware audio manager
    manager = SpeakerAwareAudioManager(config, engine)
    
    return manager
