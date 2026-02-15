#!/usr/bin/env python3
"""
Android Root Audio Translator for Pixel 8 Pro
Captures system audio with root and translates via WebSocket
"""

import asyncio
import base64
import io
import json
import logging
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np
import websockets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# Configuration
SERVER_URL = "wss://translate-speach.onrender.com/ws"  # Your Render server
AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
CHUNK_DURATION = 0.5  # seconds


class AudioCapture:
    """Capture system audio using root access."""
    
    def __init__(self):
        self.is_capturing = False
        self.process = None
        
    def find_audio_devices(self):
        """Find available audio devices on Android."""
        try:
            result = subprocess.run(
                ['su', '-c', 'ls /dev/snd/'],
                capture_output=True,
                text=True,
                timeout=5
            )
            devices = result.stdout.strip().split('\n')
            log.info(f"Found audio devices: {devices}")
            return devices
        except Exception as e:
            log.error(f"Failed to find audio devices: {e}")
            return []
    
    def capture_output_audio(self, callback):
        """Capture outgoing audio (your speech to microphone)."""
        # On Pixel 8 Pro, this captures from the microphone input
        cmd = [
            'su', '-c',
            'cat /dev/snd/pcmC0D0c 2>/dev/null || '
            'tinycap /sdcard/temp_out.wav -r 16000 -c 1 -b 16 -D 0 -d 0 2>/dev/null & '
            'sleep 0.5; cat /sdcard/temp_out.wav; rm /sdcard/temp_out.wav'
        ]
        
        return self._capture_stream(cmd, callback, "output")
    
    def capture_input_audio(self, callback):
        """Capture incoming audio (speaker output - other person's speech)."""
        # This captures from the speaker output (requires specific device)
        cmd = [
            'su', '-c',
            'cat /dev/snd/pcmC0D0p 2>/dev/null || '
            'tinymix 0 1 2>/dev/null; tinycap /sdcard/temp_in.wav -r 16000 -c 1 -b 16 -D 0 -d 1 2>/dev/null & '
            'sleep 0.5; cat /sdcard/temp_in.wav; rm /sdcard/temp_in.wav'
        ]
        
        return self._capture_stream(cmd, callback, "input")
    
    def _capture_stream(self, cmd, callback, stream_type):
        """Generic audio capture stream."""
        def capture_thread():
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                log.info(f"Started {stream_type} audio capture")
                
                # Read audio data in chunks
                while self.is_capturing:
                    data = self.process.stdout.read(int(AUDIO_RATE * 2 * CHUNK_DURATION))
                    if data:
                        callback(data, stream_type)
                        
            except Exception as e:
                log.error(f"Capture error: {e}")
            finally:
                self.stop()
        
        self.is_capturing = True
        thread = threading.Thread(target=capture_thread, daemon=True)
        thread.start()
        return thread
    
    def stop(self):
        """Stop audio capture."""
        self.is_capturing = False
        if self.process:
            self.process.terminate()
            self.process = None


class VADProcessor:
    """Simple Voice Activity Detection."""
    
    def __init__(self, threshold=500):
        self.threshold = threshold
        self.is_speaking = False
        self.buffer = []
        self.silence_frames = 0
        
    def process(self, audio_bytes):
        """Process audio chunk and detect speech."""
        try:
            # Convert to numpy
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Calculate RMS
            rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
            
            # Detect speech
            if rms > self.threshold:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.buffer = []
                    log.info("Speech started")
                
                self.buffer.append(audio)
                self.silence_frames = 0
                
            elif self.is_speaking:
                self.buffer.append(audio)
                self.silence_frames += 1
                
                # End of speech after 1 second of silence
                if self.silence_frames > int(1.0 / CHUNK_DURATION):
                    self.is_speaking = False
                    speech_data = np.concatenate(self.buffer)
                    self.buffer = []
                    log.info("Speech ended")
                    return speech_data
                    
        except Exception as e:
            log.error(f"VAD error: {e}")
        
        return None


class TranslatorClient:
    """WebSocket client for translation server."""
    
    def __init__(self, server_url):
        self.server_url = server_url
        self.ws = None
        self.connected = False
        
    async def connect(self):
        """Connect to translation server."""
        try:
            self.ws = await websockets.connect(self.server_url)
            self.connected = True
            log.info(f"Connected to {self.server_url}")
            return True
        except Exception as e:
            log.error(f"Connection failed: {e}")
            return False
    
    async def translate(self, audio_data, source_lang, target_lang):
        """Send audio for translation."""
        if not self.connected:
            log.error("Not connected to server")
            return None
        
        try:
            # Convert to base64
            audio_base64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
            
            # Send to server
            message = {
                'type': 'audio',
                'data': audio_base64,
                'sourceLang': source_lang,
                'targetLang': target_lang
            }
            
            await self.ws.send(json.dumps(message))
            
            # Wait for response
            response = await asyncio.wait_for(self.ws.recv(), timeout=30)
            data = json.loads(response)
            
            if data.get('type') == 'translation':
                return data
            elif data.get('type') == 'error':
                log.error(f"Server error: {data.get('message')}")
                
        except Exception as e:
            log.error(f"Translation error: {e}")
        
        return None
    
    async def close(self):
        """Close connection."""
        if self.ws:
            await self.ws.close()
            self.connected = False


class AudioPlayer:
    """Play translated audio using Android's audio system."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def play(self, audio_base64, sample_rate=22050):
        """Play audio from base64."""
        try:
            # Decode audio
            audio_bytes = base64.b64decode(audio_base64)
            
            # Save to temp file
            temp_file = Path(self.temp_dir) / "temp_audio.wav"
            
            with wave.open(str(temp_file), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)
            
            # Play using Android media player
            subprocess.run(
                ['su', '-c', f'am start -a android.intent.action.VIEW -d file://{temp_file} -t audio/wav'],
                capture_output=True,
                timeout=10
            )
            
            # Alternative: use tinyplay
            subprocess.run(
                ['su', '-c', f'tinyplay {temp_file}'],
                capture_output=True,
                timeout=30
            )
            
        except Exception as e:
            log.error(f"Playback error: {e}")


class LiveTranslator:
    """Main translator application."""
    
    def __init__(self, server_url):
        self.server_url = server_url
        self.client = TranslatorClient(server_url)
        self.capture = AudioCapture()
        self.vad_out = VADProcessor()  # For your speech
        self.vad_in = VADProcessor()   # For partner's speech
        self.player = AudioPlayer()
        self.running = False
        
    async def start(self):
        """Start the translator."""
        log.info("Starting Live Translator...")
        
        # Check root access
        if not self._check_root():
            log.error("Root access required!")
            return
        
        # Connect to server
        if not await self.client.connect():
            log.error("Failed to connect to server")
            return
        
        self.running = True
        
        # Start audio capture threads
        self.capture.capture_output_audio(self._on_output_audio)
        self.capture.capture_input_audio(self._on_input_audio)
        
        log.info("Translator running. Press Ctrl+C to stop.")
        
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            log.info("Stopping...")
        finally:
            await self.stop()
    
    def _check_root(self):
        """Check if we have root access."""
        try:
            result = subprocess.run(
                ['su', '-c', 'id'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return 'uid=0' in result.stdout
        except:
            return False
    
    def _on_output_audio(self, data, stream_type):
        """Handle captured output audio (your speech)."""
        speech = self.vad_out.process(data)
        if speech is not None:
            # Your speech detected - translate RU->EN
            asyncio.create_task(self._translate_speech(speech, 'ru', 'en'))
    
    def _on_input_audio(self, data, stream_type):
        """Handle captured input audio (partner's speech)."""
        speech = self.vad_in.process(data)
        if speech is not None:
            # Partner's speech detected - translate EN->RU
            asyncio.create_task(self._translate_speech(speech, 'en', 'ru'))
    
    async def _translate_speech(self, audio_data, source_lang, target_lang):
        """Translate speech and play result."""
        log.info(f"Translating {source_lang}->{target_lang}...")
        
        result = await self.client.translate(audio_data, source_lang, target_lang)
        
        if result:
            original = result.get('original', '')
            translated = result.get('text', '')
            audio = result.get('audio')
            
            log.info(f"Original: {original}")
            log.info(f"Translated: {translated}")
            
            if audio:
                self.player.play(audio)
    
    async def stop(self):
        """Stop the translator."""
        self.running = False
        self.capture.stop()
        await self.client.close()
        log.info("Translator stopped")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Android Root Audio Translator')
    parser.add_argument('--server', '-s', default=SERVER_URL,
                        help='WebSocket server URL')
    
    args = parser.parse_args()
    
    translator = LiveTranslator(args.server)
    
    try:
        asyncio.run(translator.start())
    except Exception as e:
        log.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
