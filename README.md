# Live Translator

Real-time speech-to-speech translation system supporting RU↔EN bidirectional translation.

## Features

- **Real-time Translation**: Speech-to-speech translation with minimal latency
- **Bidirectional**: Supports both RU→EN and EN→RU translation
- **Web Interface**: Access from any device via browser
- **Android Root Support**: System-level audio capture on rooted devices
- **Chunked TTS**: Incremental text-to-speech for better responsiveness
- **Cloud Deployment**: Ready for Render.com deployment

## Architecture

```
┌─────────────────┐     WebSocket      ┌─────────────────┐
│  Android/       │ ←────────────────→ │  Render Server  │
│  Web Browser    │                    │  (FastAPI)      │
└─────────────────┘                    └────────┬────────┘
       │                                        │
       │ Capture Audio                          │ STT → MT → TTS
       │                                        │
       └────────────────────────────────────────┘
                            ↓
                    ┌───────────────┐
                    │  Whisper STT  │
                    │  Google MT    │
                    │  macOS TTS    │
                    └───────────────┘
```

## Quick Start

### Web Interface

1. Open https://translate-speach.onrender.com in your browser
2. Click "Connect"
3. Select translation mode:
   - **Me → Partner**: You speak Russian, partner hears English
   - **Partner → Me**: Partner speaks English, you hear Russian
4. Hold "Speak" button and talk

### Local Development

```bash
# Clone repository
git clone https://github.com/ArsenHandzhyan/translate_speach.git
cd translate_speach

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Run web server
python -m uvicorn src.web_server:app --host 0.0.0.0 --port 8000

# Or run desktop app (macOS only)
python run.py
```

## Deployment

### Render.com

1. Fork this repository
2. Create new Web Service on Render
3. Select "Docker" runtime
4. Deploy automatically from GitHub

Environment variables (optional):
- `MODE=cloud` - Use cloud translation (default)
- `LOG_LEVEL=info` - Logging level

## Android Root Setup (Pixel 8 Pro)

### Prerequisites

- Android 11+ with root access (Magisk 24+)
- Termux from F-Droid
- Internet connection

### Installation

```bash
# In Termux
cd ~
git clone https://github.com/ArsenHandzhyan/translate_speach.git
cd translate_speach/android
bash setup.sh
```

### Usage

```bash
cd /sdcard/translator
./run.sh
```

The script will:
1. Capture microphone audio (your speech) → translate RU→EN
2. Capture speaker audio (partner's speech) → translate EN→RU
3. Play translated audio automatically

### Audio Device Configuration

If audio capture doesn't work, find correct devices:

```bash
su -c "ls /dev/snd/"
su -c "tinymix contents"
```

Edit `translator.py` to use correct device IDs.

## Project Structure

```
translate_speach/
├── src/
│   ├── __init__.py
│   ├── web_server.py      # FastAPI web server
│   ├── engine.py          # Translation engine
│   ├── stt.py            # Speech-to-text (Whisper)
│   ├── mt.py             # Machine translation
│   ├── tts.py            # Text-to-speech
│   ├── vad.py            # Voice activity detection
│   └── streams.py        # Audio stream management (desktop)
├── web/
│   ├── index.html        # Web interface
│   └── static/
│       ├── style.css     # UI styles
│       └── app.js        # WebSocket client
├── android/
│   ├── translator.py     # Android root script
│   └── setup.sh          # Setup script
├── Dockerfile            # Render deployment
├── pyproject.toml        # Python dependencies
└── README.md            # This file
```

## Configuration

### Web Interface Modes

1. **Outgoing (Me → Partner)**
   - Source: Russian
   - Target: English
   - Use: You speak, partner hears translation

2. **Incoming (Partner → Me)**
   - Source: English
   - Target: Russian
   - Use: Partner speaks, you hear translation

### Server Configuration

Edit `src/config.py`:

```python
# Translation mode
mode = Mode.CLOUD  # Use Google Translate API

# Audio settings
SAMPLE_RATE = 48000
SILENCE_THRESHOLD_S = 0.6

# TTS chunking
tts_chunk_words = 3  # Words per TTS chunk
```

## API Endpoints

### HTTP

- `GET /` - Web interface
- `GET /health` - Health check

### WebSocket

- `WS /ws` - Real-time translation

Message format:
```json
{
  "type": "audio",
  "data": "base64_encoded_audio",
  "sourceLang": "ru",
  "targetLang": "en"
}
```

Response:
```json
{
  "type": "translation",
  "original": "Привет",
  "text": "Hello",
  "audio": "base64_encoded_wav"
}
```

## Troubleshooting

### Web Interface

**Problem**: Cannot connect to server
- Check server URL is correct
- Ensure HTTPS is used (required for microphone)
- Check browser console for errors

**Problem**: No audio playback
- Check "Auto-play" is enabled in settings
- Ensure browser allows audio playback
- Try clicking "Speak" button again

### Android Root

**Problem**: "Root access required" error
- Verify Magisk is installed and granted root
- Run `su -c "id"` in Termux to test

**Problem**: No audio capture
- Check audio devices: `su -c "ls /dev/snd/"`
- Verify tinycap/tinyplay are available
- Try different device IDs in translator.py

**Problem**: WebSocket connection fails
- Check internet connection
- Verify server URL in run.sh
- Check Render server is running

### Server Deployment

**Problem**: Build fails on Render
- Check all dependencies in Dockerfile
- Verify pyproject.toml is valid
- Check Render logs for specific errors

## Technologies

- **Backend**: Python 3.11, FastAPI, WebSocket
- **STT**: OpenAI Whisper (faster-whisper)
- **MT**: Google Translate API
- **TTS**: macOS `say` command (server), Web Audio API (browser)
- **Audio**: WebRTC, pydub, webrtcvad
- **Frontend**: Vanilla JavaScript, WebSocket

## License

MIT License - See LICENSE file for details

## Contributing

Pull requests welcome. Please ensure:
- Code follows existing style
- Tests pass (if applicable)
- Documentation is updated

## Support

For issues and feature requests, please use GitHub Issues.
