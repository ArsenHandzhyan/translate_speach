// WebRTC Client for Live Translator

class TranslatorClient {
    constructor() {
        this.ws = null;
        this.peerConnection = null;
        this.localStream = null;
        this.isConnected = false;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        
        this.initElements();
        this.bindEvents();
    }

    initElements() {
        this.connectBtn = document.getElementById('connect-btn');
        this.speakBtn = document.getElementById('speak-btn');
        this.statusDot = document.getElementById('status-dot');
        this.statusText = document.getElementById('status-text');
        this.progressText = document.getElementById('progress-text');
        this.originalText = document.getElementById('original-text');
        this.translatedText = document.getElementById('translated-text');
        this.historyList = document.getElementById('history-list');
        this.sourceLang = document.getElementById('source-lang');
        this.targetLang = document.getElementById('target-lang');
        this.autoPlay = document.getElementById('auto-play');
    }

    bindEvents() {
        this.connectBtn.addEventListener('click', () => this.toggleConnection());
        
        // Speak button - hold to record
        this.speakBtn.addEventListener('mousedown', () => this.startRecording());
        this.speakBtn.addEventListener('mouseup', () => this.stopRecording());
        this.speakBtn.addEventListener('mouseleave', () => this.stopRecording());
        
        // Touch events for mobile
        this.speakBtn.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.startRecording();
        });
        this.speakBtn.addEventListener('touchend', (e) => {
            e.preventDefault();
            this.stopRecording();
        });

        // Language swap
        this.sourceLang.addEventListener('change', () => this.updateTargetLang());
    }

    updateTargetLang() {
        const src = this.sourceLang.value;
        this.targetLang.value = src === 'ru' ? 'en' : 'ru';
    }

    async toggleConnection() {
        if (this.isConnected) {
            this.disconnect();
        } else {
            await this.connect();
        }
    }

    async connect() {
        try {
            this.setStatus('connecting', 'Connecting...');
            
            // Get WebSocket URL
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            // Connect WebSocket
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                this.isConnected = true;
                this.setStatus('connected', 'Connected');
                this.speakBtn.disabled = false;
                this.connectBtn.textContent = 'Disconnect';
                this.progressText.textContent = 'Ready to translate';
            };
            
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleServerMessage(data);
            };
            
            this.ws.onclose = () => {
                this.disconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.setStatus('disconnected', 'Error');
            };
            
        } catch (error) {
            console.error('Connection error:', error);
            this.setStatus('disconnected', 'Failed to connect');
        }
    }

    disconnect() {
        this.isConnected = false;
        
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        
        if (this.localStream) {
            this.localStream.getTracks().forEach(track => track.stop());
            this.localStream = null;
        }
        
        this.setStatus('disconnected', 'Disconnected');
        this.speakBtn.disabled = true;
        this.connectBtn.textContent = 'Connect';
        this.progressText.textContent = 'Ready';
    }

    async startRecording() {
        if (!this.isConnected || this.isRecording) return;
        
        try {
            this.isRecording = true;
            this.speakBtn.classList.add('recording');
            this.progressText.textContent = 'Listening...';
            this.audioChunks = [];
            
            // Get microphone access
            this.localStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
            
            // Use MediaRecorder to capture audio
            this.mediaRecorder = new MediaRecorder(this.localStream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                this.sendAudio();
            };
            
            this.mediaRecorder.start(100); // Collect data every 100ms
            
        } catch (error) {
            console.error('Recording error:', error);
            this.stopRecording();
            alert('Could not access microphone. Please allow microphone access.');
        }
    }

    stopRecording() {
        if (!this.isRecording) return;
        
        this.isRecording = false;
        this.speakBtn.classList.remove('recording');
        this.progressText.textContent = 'Processing...';
        
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }
        
        if (this.localStream) {
            this.localStream.getTracks().forEach(track => track.stop());
            this.localStream = null;
        }
    }

    async sendAudio() {
        if (this.audioChunks.length === 0) return;
        
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        const reader = new FileReader();
        
        reader.onloadend = () => {
            const base64Audio = reader.result.split(',')[1];
            
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    type: 'audio',
                    data: base64Audio,
                    sourceLang: this.sourceLang.value,
                    targetLang: this.targetLang.value
                }));
            }
        };
        
        reader.readAsDataURL(audioBlob);
    }

    handleServerMessage(data) {
        switch (data.type) {
            case 'transcription':
                this.originalText.textContent = data.text;
                break;
                
            case 'translation':
                this.translatedText.textContent = data.text;
                this.addToHistory(data.original, data.text);
                this.progressText.textContent = 'Ready';
                
                // Play audio if provided and auto-play is enabled
                if (data.audio && this.autoPlay.checked) {
                    this.playAudio(data.audio);
                }
                break;
                
            case 'error':
                console.error('Server error:', data.message);
                this.progressText.textContent = 'Error: ' + data.message;
                break;
                
            case 'ping':
                // Keep connection alive
                break;
        }
    }

    playAudio(base64Audio) {
        const audio = new Audio('data:audio/wav;base64,' + base64Audio);
        audio.play().catch(error => {
            console.error('Audio playback error:', error);
        });
    }

    addToHistory(original, translated) {
        const item = document.createElement('div');
        item.className = 'history-item';
        item.innerHTML = `
            <div class="history-original">${this.escapeHtml(original)}</div>
            <div class="history-translated">${this.escapeHtml(translated)}</div>
        `;
        this.historyList.insertBefore(item, this.historyList.firstChild);
        
        // Keep only last 10 items
        while (this.historyList.children.length > 10) {
            this.historyList.removeChild(this.historyList.lastChild);
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    setStatus(state, text) {
        this.statusDot.className = 'status-dot';
        if (state === 'connected') {
            this.statusDot.classList.add('connected');
        } else if (state === 'connecting') {
            this.statusDot.classList.add('connecting');
        }
        this.statusText.textContent = text;
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.translatorClient = new TranslatorClient();
});
