"""FastAPI Web Server for Live Translator with WebSocket support."""
import asyncio
import base64
import io
import json
import logging
import os
import tempfile
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import TranslatorConfig, Mode
from .engine import TranslationEngine

log = logging.getLogger(__name__)

# Global engine instance
engine: Optional[TranslationEngine] = None
config: Optional[TranslatorConfig] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global engine, config
    
    log.info("Loading translation engine...")
    config = TranslatorConfig()
    config.mode = Mode.CLOUD  # Use cloud MT for server
    config.show_subtitles = False

    # Use smaller Whisper model for server (saves memory on Render free tier)
    config.whisper_model = os.environ.get("WHISPER_MODEL", "tiny")
    log.info(f"Using Whisper model: {config.whisper_model}")

    engine = TranslationEngine(config)
    engine.load_models()
    
    log.info("Server ready!")
    yield
    
    log.info("Shutting down...")
    engine = None


app = FastAPI(
    title="Live Translator",
    description="RU↔EN Speech-to-Speech Translation Server",
    version="0.1.0",
    lifespan=lifespan
)

# Serve static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")


@app.get("/")
async def root():
    """Serve the main web interface."""
    return FileResponse("web/index.html")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "engine_loaded": engine is not None
    }


class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        log.info(f"Client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        log.info(f"Client disconnected. Total: {len(self.active_connections)}")
    
    async def send_message(self, websocket: WebSocket, message: dict):
        try:
            await websocket.send_json(message)
        except Exception as e:
            log.error(f"Error sending message: {e}")


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time translation."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            msg_type = message.get('type')
            
            if msg_type == 'audio':
                await handle_audio_message(websocket, message)
            elif msg_type == 'ping':
                await manager.send_message(websocket, {'type': 'pong'})
            else:
                log.warning(f"Unknown message type: {msg_type}")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        log.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def handle_audio_message(websocket: WebSocket, message: dict):
    """Process audio message and send back translation."""
    try:
        # Get audio data
        audio_base64 = message.get('data', '')
        source_lang = message.get('sourceLang', 'ru')
        target_lang = message.get('targetLang', 'en')
        
        if not audio_base64:
            await manager.send_message(websocket, {
                'type': 'error',
                'message': 'No audio data received'
            })
            return
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_base64)
        mime_type = message.get('mimeType', 'audio/webm')

        # Convert to numpy array (16kHz, mono, int16)
        audio_int16 = await convert_audio(audio_bytes, mime_type)
        
        if audio_int16 is None or len(audio_int16) == 0:
            await manager.send_message(websocket, {
                'type': 'error',
                'message': 'Could not process audio'
            })
            return
        
        # Send transcription progress
        await manager.send_message(websocket, {
            'type': 'progress',
            'text': 'Transcribing...'
        })
        
        # Process through translation engine
        results = engine.translate_speech_chunked(
            audio_int16,
            src_lang=source_lang,
            tgt_lang=target_lang
        )
        
        if not results:
            await manager.send_message(websocket, {
                'type': 'error',
                'message': 'No speech detected'
            })
            return
        
        # Get full text from results
        original_text = results[0][1] if results else ""
        translated_chunks = [r[2] for r in results if r[2]]
        translated_text = ' '.join(translated_chunks)
        
        # Send transcription
        await manager.send_message(websocket, {
            'type': 'transcription',
            'text': original_text
        })
        
        # Combine all TTS audio chunks
        tts_audio_parts = [r[0] for r in results if r[0] is not None]
        
        if tts_audio_parts:
            # Combine audio chunks
            combined_audio = np.concatenate(tts_audio_parts)
            
            # Convert to WAV bytes
            audio_base64 = audio_to_base64(combined_audio, engine._tts.sample_rate)
            
            # Send translation with audio
            await manager.send_message(websocket, {
                'type': 'translation',
                'original': original_text,
                'text': translated_text,
                'audio': audio_base64
            })
        else:
            # Send translation without audio
            await manager.send_message(websocket, {
                'type': 'translation',
                'original': original_text,
                'text': translated_text
            })
            
    except Exception as e:
        log.error(f"Error processing audio: {e}", exc_info=True)
        await manager.send_message(websocket, {
            'type': 'error',
            'message': str(e)
        })


async def convert_audio(audio_bytes: bytes, mime_type: str = 'audio/webm') -> Optional[np.ndarray]:
    """Convert browser audio to 16kHz int16 numpy array.

    Supports WebM, OGG, MP4 and other formats via ffmpeg/pydub.
    """
    # Determine file extension from MIME type
    mime_to_ext = {
        'audio/webm': '.webm',
        'audio/webm;codecs=opus': '.webm',
        'audio/ogg': '.ogg',
        'audio/ogg;codecs=opus': '.ogg',
        'audio/mp4': '.mp4',
        'audio/mpeg': '.mp3',
        'audio/wav': '.wav',
    }
    ext = mime_to_ext.get(mime_type, '.webm')
    log.info(f"Audio conversion: mime={mime_type}, ext={ext}, size={len(audio_bytes)} bytes")

    try:
        # Save to temporary file with correct extension
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(audio_bytes)
            input_path = f.name

        wav_path = input_path.rsplit('.', 1)[0] + '.wav'

        # Try ffmpeg first
        try:
            import subprocess
            subprocess.run([
                'ffmpeg', '-y', '-i', input_path,
                '-ar', '16000', '-ac', '1', '-f', 'wav', wav_path
            ], capture_output=True, timeout=30, check=True)

            with wave.open(wav_path, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16)

            Path(input_path).unlink(missing_ok=True)
            Path(wav_path).unlink(missing_ok=True)
            return audio

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            log.warning(f"ffmpeg failed for {ext}: {e}")
            # Fallback: pydub with auto-detection (no explicit format)
            try:
                from pydub import AudioSegment

                audio = AudioSegment.from_file(input_path)
                audio = audio.set_frame_rate(16000).set_channels(1)
                samples = np.array(audio.get_array_of_samples())

                Path(input_path).unlink(missing_ok=True)
                return samples.astype(np.int16)

            except Exception as pydub_err:
                log.error(f"pydub fallback also failed: {pydub_err}")
                Path(input_path).unlink(missing_ok=True)
                return None

    except Exception as e:
        log.error(f"Audio conversion error: {e}")
        return None


def audio_to_base64(audio: np.ndarray, sample_rate: int) -> str:
    """Convert numpy audio array to base64 WAV."""
    # Ensure int16 format
    if audio.dtype != np.int16:
        audio = (audio * 32767).astype(np.int16)
    
    # Create WAV in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    
    # Convert to base64
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def main():
    """Run the web server."""
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    uvicorn.run(
        "src.web_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
