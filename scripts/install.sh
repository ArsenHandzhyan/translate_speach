#!/bin/bash
set -euo pipefail

# Live Translator RU↔EN — Installation Script for macOS (Apple Silicon)
# Usage: bash scripts/install.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"
MODELS_DIR="$PROJECT_DIR/models"

echo "========================================"
echo "  Live Translator RU↔EN Installer"
echo "========================================"
echo "  Project: $PROJECT_DIR"
echo ""

# --- Step 1: Check prerequisites ---
echo "[1/6] Checking prerequisites..."

if ! command -v brew &>/dev/null; then
    echo "ERROR: Homebrew not found. Install from https://brew.sh"
    exit 1
fi

if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python3 not found."
    exit 1
fi

echo "  Homebrew: $(brew --version | head -1)"
echo "  Python: $(python3 --version)"

# --- Step 2: Install system dependencies ---
echo ""
echo "[2/6] Installing system dependencies..."

brew list portaudio &>/dev/null || brew install portaudio
brew list ffmpeg &>/dev/null || brew install ffmpeg

# Check BlackHole
if system_profiler SPAudioDataType 2>/dev/null | grep -qi blackhole; then
    echo "  BlackHole: already installed"
else
    echo ""
    echo "  ⚠ BlackHole is NOT installed."
    echo "  Run in a separate terminal: brew install blackhole-2ch"
    echo "  Then REBOOT your Mac."
    echo "  After reboot, re-run this script."
    echo ""
fi

# --- Step 3: Create Python venv ---
echo ""
echo "[3/6] Setting up Python virtual environment..."

if [ -d "$VENV_DIR" ]; then
    echo "  Venv already exists at $VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
    echo "  Created venv at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel 2>&1 | tail -1

# --- Step 4: Install Python dependencies ---
echo ""
echo "[4/6] Installing Python packages..."

pip install \
    numpy \
    sounddevice \
    scipy \
    faster-whisper \
    ctranslate2 \
    sentencepiece \
    requests \
    webrtcvad-wheels \
    rumps \
    pynput \
    2>&1 | tail -5

# Install the project itself
pip install -e "$PROJECT_DIR" 2>&1 | tail -3

echo "  Python packages installed."

# --- Step 5: Download models ---
echo ""
echo "[5/6] Downloading models for LOCAL mode..."

mkdir -p "$MODELS_DIR"

# Whisper model (faster-whisper will auto-download on first use)
echo "  Whisper 'small' model will be auto-downloaded on first run."
echo "  (faster-whisper caches to ~/.cache/huggingface/hub/)"

# Test that faster-whisper can initialize
python3 -c "
from faster_whisper import WhisperModel
print('  Testing Whisper model download...')
model = WhisperModel('small', device='cpu', compute_type='int8')
print('  Whisper model ready.')
" 2>&1 || echo "  Warning: Whisper model download may happen on first run."

echo ""
echo "  Models setup complete."

# --- Step 6: Create convenience aliases ---
echo ""
echo "[6/6] Setup complete!"
echo ""

cat << 'INSTRUCTIONS'
========================================
  SETUP COMPLETE — Next Steps
========================================

1. If BlackHole is not installed yet:
   brew install blackhole-2ch
   Then REBOOT your Mac.

2. After reboot, set up Audio MIDI Setup:
   - Open: Audio MIDI Setup (Cmd+Space → "Audio MIDI")
   - Click [+] → "Create Multi-Output Device"
   - Check: your headphones/speakers (first!) + BlackHole 2ch
   - Rename to: "Translator Monitor"

3. Activate the environment:
   source .venv/bin/activate

4. Test the system:
   python -m src.cli test

5. Start the translator:
   python -m src.cli start --mode hybrid

6. In your browser call:
   - Microphone: BlackHole 2ch
   - Speaker: Translator Monitor (Multi-Output)

========================================
  Quick Commands
========================================
  python -m src.cli start          # Start (hybrid mode)
  python -m src.cli start --mode local  # Offline mode
  python -m src.cli stop           # Stop
  python -m src.cli devices        # List audio devices
  python -m src.cli route setup    # Show routing instructions
  python -m src.cli test           # Quick translation test
  python -m src.menubar            # Start menubar app

========================================
INSTRUCTIONS
