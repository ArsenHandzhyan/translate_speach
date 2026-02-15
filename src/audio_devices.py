"""Audio device discovery and routing for macOS."""
import sounddevice as sd
import subprocess
import json
import logging

log = logging.getLogger(__name__)


def list_devices():
    """List all audio devices."""
    return sd.query_devices()


def find_device(name_substring: str, kind: str = None) -> int | None:
    """Find device index by name substring. kind='input' or 'output'."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if name_substring.lower() in dev["name"].lower():
            if kind == "input" and dev["max_input_channels"] == 0:
                continue
            if kind == "output" and dev["max_output_channels"] == 0:
                continue
            return i
    return None


def check_blackhole_installed() -> bool:
    """Check if BlackHole is available as an audio device."""
    return find_device("BlackHole") is not None


def get_device_sample_rate(device_index: int) -> float:
    """Get the default sample rate of a device."""
    dev = sd.query_devices(device_index)
    return dev["default_samplerate"]


def print_routing_status():
    """Print current audio routing status."""
    devices = sd.query_devices()
    print("\n=== Audio Devices ===")
    for i, d in enumerate(devices):
        direction = ""
        if d["max_input_channels"] > 0:
            direction += "IN"
        if d["max_output_channels"] > 0:
            direction += ("/" if direction else "") + "OUT"
        print(f"  [{i}] {d['name']} ({direction}) sr={d['default_samplerate']}")

    bh = find_device("BlackHole")
    if bh is not None:
        print(f"\n  BlackHole found at index {bh}")
    else:
        print("\n  WARNING: BlackHole NOT found. Install it first.")

    print()


def setup_instructions() -> str:
    """Return instructions for Audio MIDI Setup."""
    return """
=== Audio MIDI Setup Instructions ===

1. Open Audio MIDI Setup (Spotlight → "Audio MIDI Setup")

2. Create Multi-Output Device (for capturing browser audio):
   - Click "+" at bottom-left → "Create Multi-Output Device"
   - Check: your headphones/speakers (first!) + BlackHole 2ch
   - Rename to: "Translator Monitor"
   - This lets you hear browser audio AND route it to the translator

3. In your browser (Chrome):
   - Go to Settings → Privacy → Site Settings → Microphone
   - Or in a call: select "BlackHole 2ch" as microphone
   - For speakers: select "Translator Monitor" as output

4. In macOS System Settings → Sound:
   - Output: your headphones (for hearing translated audio)
   - Input: your real microphone (built-in or external)

The translator will:
   - Read from BlackHole (browser output) → translate EN→RU → play to headphones
   - Read from your mic → translate RU→EN → write to BlackHole → browser picks it up

NOTE: Since BlackHole 2ch is used for BOTH directions, we need BlackHole 16ch
for full duplex. With 2ch, we use time-division: the translator alternates
between reading browser audio and writing translated mic audio.

SIMPLER APPROACH (recommended for start):
- Use BlackHole 2ch as the "Translated Mic" for browser input only
- Capture browser output via a loopback (Multi-Output Device)
- Your physical mic → translator → BlackHole 2ch → browser
- Browser → Multi-Output (headphones + loopback) → translator → headphones TTS
"""
