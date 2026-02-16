"""Voice enrollment script for speaker identification."""
import logging
import sys
import time

import numpy as np
import sounddevice as sd

from .speaker_id import SpeakerIdentifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# Recording settings
SAMPLE_RATE = 16000
DURATION = 30  # seconds
CHUNK_SIZE = 1024


def record_audio(duration: int = DURATION, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record audio from microphone.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate
        
    Returns:
        Recorded audio as numpy array (float32)
    """
    log.info(f"Recording for {duration} seconds...")
    log.info("Please speak continuously in your natural voice.")
    log.info("You can read any text or just talk about your day.")
    print("\n" + "="*50)
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    
    print("🎤 RECORDING STARTED - SPEAK NOW")
    print("="*50 + "\n")
    
    # Record audio
    recording = []
    
    def callback(indata, frames, time_info, status):
        if status:
            log.warning(f"Audio status: {status}")
        recording.append(indata.copy())
    
    # Start recording
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32,
        blocksize=CHUNK_SIZE,
        callback=callback
    ):
        # Progress bar
        for i in range(duration):
            time.sleep(1)
            progress = (i + 1) / duration * 100
            bar = "█" * int(progress // 5) + "░" * (20 - int(progress // 5))
            print(f"\r[{bar}] {progress:.0f}% ({i+1}/{duration}s)", end="", flush=True)
    
    print("\n\n" + "="*50)
    print("✅ RECORDING COMPLETE")
    print("="*50 + "\n")
    
    # Concatenate chunks
    audio = np.concatenate(recording, axis=0).flatten()
    
    log.info(f"Recorded {len(audio) / sample_rate:.1f} seconds of audio")
    
    return audio


def main():
    """Run voice enrollment."""
    print("\n" + "="*60)
    print("  VOICE ENROLLMENT FOR LIVE TRANSLATOR")
    print("="*60 + "\n")
    
    print("This will create a voice profile for speaker identification.")
    print("The system will learn to recognize YOUR voice.")
    print("\nRequirements:")
    print("- Speak for 30 seconds")
    print("- Use your natural speaking voice")
    print("- Speak clearly and at normal volume")
    print("- Minimize background noise")
    print()
    
    # Confirm
    response = input("Ready to start? (yes/no): ").strip().lower()
    if response not in ('yes', 'y'):
        print("Enrollment cancelled.")
        return
    
    try:
        # Record audio
        audio = record_audio(duration=DURATION)
        
        # Check audio quality
        rms = np.sqrt(np.mean(audio ** 2))
        log.info(f"Audio RMS level: {rms:.4f}")
        
        if rms < 0.005:
            log.error("Audio level too low! Please check your microphone and try again.")
            log.error("Tips: speak closer to microphone, increase input volume in System Settings")
            return
        
        # Create speaker identifier and enroll
        print("\nCreating voice profile...")
        print("(This may take a moment - loading ML model)\n")
        
        speaker_id = SpeakerIdentifier(use_advanced=True)
        speaker_id.enroll(audio, sample_rate=SAMPLE_RATE)
        
        print("\n" + "="*60)
        print("  ✅ ENROLLMENT SUCCESSFUL!")
        print("="*60)
        print("\nYour voice profile has been saved.")
        print("You can now run the translator with automatic speaker detection.")
        print("\nTo start translation:")
        print("  python run.py")
        print()
        
    except KeyboardInterrupt:
        print("\n\nEnrollment cancelled by user.")
        sys.exit(0)
    except Exception as e:
        log.error(f"Enrollment failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
