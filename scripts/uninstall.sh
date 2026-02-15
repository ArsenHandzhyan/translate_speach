#!/bin/bash
set -euo pipefail

# Live Translator — Complete Uninstall Script
# This removes everything the translator installed, safely.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================"
echo "  Live Translator — Uninstall"
echo "========================================"
echo ""

read -p "Remove Python venv? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$PROJECT_DIR/.venv"
    echo "  Removed .venv"
fi

read -p "Remove downloaded models? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$PROJECT_DIR/models"
    echo "  Removed models/"
fi

read -p "Remove logs? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$PROJECT_DIR/logs"
    echo "  Removed logs/"
fi

read -p "Remove faster-whisper cache (~/.cache/huggingface)? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf ~/.cache/huggingface/hub/models--Systran--faster-whisper-*
    echo "  Removed whisper model cache"
fi

echo ""
echo "To uninstall BlackHole:"
echo "  brew uninstall blackhole-2ch"
echo "  Then reboot."
echo ""
echo "To remove the Multi-Output Device:"
echo "  Open Audio MIDI Setup → select 'Translator Monitor' → click [-]"
echo ""
echo "Done. The project directory remains at: $PROJECT_DIR"
echo "Delete it manually if you want: rm -rf $PROJECT_DIR"
