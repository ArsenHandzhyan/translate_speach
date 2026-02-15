#!/bin/bash
# Setup script for Android Root Audio Translator on Pixel 8 Pro
# Run this in Termux with root access

set -e

echo "=========================================="
echo "Android Root Audio Translator Setup"
echo "=========================================="
echo ""

# Check if running in Termux
if [ -z "$TERMUX_VERSION" ]; then
    echo "Error: This script must be run in Termux!"
    exit 1
fi

# Check root access
echo "Checking root access..."
if ! su -c "id" > /dev/null 2>&1; then
    echo "Error: Root access required! Please grant root permissions."
    exit 1
fi
echo "Root access: OK"
echo ""

# Update packages
echo "Updating packages..."
pkg update -y
echo ""

# Install required packages
echo "Installing required packages..."
pkg install -y python ffmpeg libsndfile
pip install --upgrade pip
pip install numpy websockets

# Check for audio tools
echo ""
echo "Checking audio tools..."

# Check tinycap/tinyplay
if su -c "which tinycap" > /dev/null 2>&1; then
    echo "Found tinycap: OK"
else
    echo "Warning: tinycap not found. Will try alternative methods."
fi

if su -c "which tinyplay" > /dev/null 2>&1; then
    echo "Found tinyplay: OK"
else
    echo "Warning: tinyplay not found. Will try alternative methods."
fi

# Check audio devices
echo ""
echo "Checking audio devices..."
su -c "ls -la /dev/snd/" || echo "Warning: Cannot access /dev/snd/"
echo ""

# Create working directory
echo "Creating working directory..."
mkdir -p /sdcard/translator
cp translator.py /sdcard/translator/
chmod +x /sdcard/translator/translator.py
echo "Created: /sdcard/translator/"
echo ""

# Create launcher script
echo "Creating launcher script..."
cat > /sdcard/translator/run.sh << 'EOF'
#!/bin/bash
cd /sdcard/translator
python translator.py --server "wss://YOUR_RENDER_URL/ws"
EOF
chmod +x /sdcard/translator/run.sh
echo "Created: /sdcard/translator/run.sh"
echo ""

# Create test script
echo "Creating test script..."
cat > /sdcard/translator/test_audio.sh << 'EOF'
#!/bin/bash
echo "Testing audio capture..."
echo "Recording 3 seconds of microphone audio..."
su -c "tinycap /sdcard/translator/test_mic.wav -r 16000 -c 1 -b 16 -D 0 -d 0" &
PID=$!
sleep 3
kill $PID 2>/dev/null
wait $PID 2>/dev/null
echo "Playing back..."
su -c "tinyplay /sdcard/translator/test_mic.wav"
echo "Test complete!"
EOF
chmod +x /sdcard/translator/test_audio.sh
echo "Created: /sdcard/translator/test_audio.sh"
echo ""

# Instructions
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: Edit /sdcard/translator/run.sh"
echo "Replace YOUR_RENDER_URL with your actual Render URL"
echo ""
echo "To start translator:"
echo "  cd /sdcard/translator"
echo "  ./run.sh"
echo ""
echo "To test audio first:"
echo "  ./test_audio.sh"
echo ""
echo "To find correct audio device:"
echo "  su -c 'ls /dev/snd/'"
echo "  su -c 'tinymix contents'"
echo ""
