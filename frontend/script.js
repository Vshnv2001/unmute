const API_URL = "http://127.0.0.1:8000/api/translate";
const TRANSCRIBE_API_URL = "http://127.0.0.1:8000/api/transcribe";

// DOM Elements
const outputSection = document.getElementById('outputSection');
const glossDisplay = document.getElementById('glossDisplay');
const signPlayer = document.getElementById('signPlayer');
const placeholder = document.getElementById('placeholder');
const statusNotes = document.getElementById('statusNotes');
const playerLabel = document.getElementById('playerLabel');
const replayBtn = document.getElementById('replayBtn');

// Mode switching elements
const voiceModeTab = document.getElementById('voiceModeTab');
const videoModeTab = document.getElementById('videoModeTab');
const voiceInputSection = document.getElementById('voiceInputSection');
const videoInputSection = document.getElementById('videoInputSection');

// Voice mode elements
const micBtn = document.getElementById('micBtn');
const micIcon = document.getElementById('micIcon');
const stopIcon = document.getElementById('stopIcon');
const micStatus = document.getElementById('micStatus');
const audioWave = document.getElementById('audioWave');

// Video mode elements
const myPeerIdDisplay = document.getElementById('myPeerId');
const remotePeerIdInput = document.getElementById('remotePeerId');
const callBtn = document.getElementById('callBtn');
const videoMicBtn = document.getElementById('videoMicBtn');
const localVideo = document.getElementById('localVideo');
const remoteVideo = document.getElementById('remoteVideo');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const copyPeerIdBtn = document.getElementById('copyPeerId');

// Video integrated output elements
const videoTranscriptionDisplay = document.getElementById('videoTranscriptionDisplay');
const videoSignPlayer = document.getElementById('videoSignPlayer');
const videoPlayerPlaceholder = document.getElementById('videoPlayerPlaceholder');
const videoPlayerLabel = document.getElementById('videoPlayerLabel');

// Transcription review elements
const transcriptionReviewSection = document.getElementById('transcriptionReviewSection');
const transcriptionInput = document.getElementById('transcriptionInput');
const confirmTranscriptionBtn = document.getElementById('confirmTranscriptionBtn');
const retryRecordingBtn = document.getElementById('retryRecordingBtn');

let currentPlan = [];
let isPlaying = false;

// Audio recording state
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let currentRecordingStream = null;
let isVideoTranscription = false;


// PeerJS state
let peer = null;
let localStream = null;
let currentCall = null;

// Initialization
initPeer();

async function initPeer() {
    // Fetch ICE servers (STUN/TURN) from backend
    // This includes free TURN servers to handle NAT traversal issues
    let iceServers = [
        // Default STUN servers as fallback
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' },
        { urls: 'stun:stun2.l.google.com:19302' },
        { urls: 'stun:stun.stunprotocol.org:3478' },
    ];

    try {
        const response = await fetch('http://127.0.0.1:8000/api/webrtc/ice-servers');
        if (response.ok) {
            const data = await response.json();
            if (data.iceServers && data.iceServers.length > 0) {
                iceServers = data.iceServers;
                console.log('Fetched ICE servers from backend:', iceServers);
            }
        } else {
            console.warn('Failed to fetch ICE servers from backend, using defaults');
        }
    } catch (error) {
        console.warn('Error fetching ICE servers from backend:', error);
        console.log('Using default STUN servers (TURN may not work without backend)');
    }

    // Use public PeerJS server with proper STUN/TURN configuration
    // STUN servers help with NAT traversal, TURN servers relay traffic when direct connection fails
    peer = new Peer({
        config: {
            iceServers: iceServers
        }
    });

    peer.on('open', (id) => {
        console.log('My peer ID is: ' + id);
        myPeerIdDisplay.textContent = id;
    });

    peer.on('call', (call) => {
        console.log('Receiving call...');
        navigator.mediaDevices.getUserMedia({ video: true, audio: true })
            .then((stream) => {
                localStream = stream;
                localVideo.srcObject = stream;
                call.answer(stream);
                handleCall(call);
            })
            .catch((err) => {
                console.error('Error accessing media devices:', err);
                alert('Could not access camera/microphone. Please grant permissions.');
            });
    });

    peer.on('error', (err) => {
        console.error('PeerJS error:', err);
        // Handle specific error types more gracefully
        if (err.type === 'peer-unavailable') {
            console.warn('Peer unavailable:', err.message);
        } else if (err.type === 'network') {
            alert('Network error: Could not connect to PeerJS server. Please check your internet connection.');
        } else if (err.type === 'browser-incompatible') {
            alert('Your browser does not support WebRTC. Please use a modern browser like Chrome, Firefox, or Safari.');
        } else {
            console.error('PeerJS error:', err.type, err.message);
        }
    });
}

function handleCall(call) {
    currentCall = call;

    call.on('stream', (remoteStream) => {
        console.log('Received remote stream');
        remoteVideo.srcObject = remoteStream;
        videoPlaceholder.classList.add('hidden');
    });

    call.on('close', () => {
        console.log('Call closed');
        remoteVideo.srcObject = null;
        videoPlaceholder.classList.remove('hidden');
    });

    call.on('error', (err) => {
        console.error('Call error:', err);
        // ICE failures often indicate network/NAT issues
        if (err.message && err.message.includes('ICE')) {
            alert('Connection failed: Unable to establish peer-to-peer connection. This may be due to network restrictions. Try:\n\n1. Check your firewall settings\n2. Ensure both peers are on the same network or have proper NAT traversal\n3. Consider using a TURN server for production use');
        } else {
            alert('Call error: ' + (err.message || err.type || 'Unknown error'));
        }
    });

    // Monitor ICE connection state
    if (call.peerConnection) {
        call.peerConnection.oniceconnectionstatechange = () => {
            const state = call.peerConnection.iceConnectionState;
            console.log('ICE connection state:', state);

            if (state === 'failed' || state === 'disconnected') {
                console.warn('ICE connection failed or disconnected');
                // The error handler above will catch this, but we log it here for debugging
            }
        };

        call.peerConnection.onicecandidateerror = (event) => {
            console.error('ICE candidate error:', event);
            // Log but don't alert - this is often just a warning about unreachable servers
        };
    }
}

// Mode switching
voiceModeTab.addEventListener('click', () => {
    voiceModeTab.classList.add('active');
    voiceModeTab.classList.remove('text-slate-400');
    videoModeTab.classList.remove('active');
    videoModeTab.classList.add('text-slate-400');
    voiceInputSection.classList.remove('hidden');
    videoInputSection.classList.add('hidden');
});

videoModeTab.addEventListener('click', async () => {
    videoModeTab.classList.add('active');
    videoModeTab.classList.remove('text-slate-400');
    voiceModeTab.classList.remove('active');
    voiceModeTab.classList.add('text-slate-400');
    videoInputSection.classList.remove('hidden');
    voiceInputSection.classList.add('hidden');

    // Start local camera when switching to video mode
    if (!localStream) {
        try {
            localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
            localVideo.srcObject = localStream;
        } catch (err) {
            console.error('Error accessing media devices.', err);
            alert('Could not access camera/microphone.');
        }
    }
});

callBtn.addEventListener('click', () => {
    const remoteId = remotePeerIdInput.value.trim();
    if (!remoteId) {
        alert('Please enter a remote Peer ID');
        return;
    }

    if (!peer || !peer.open) {
        alert('PeerJS not ready. Please wait for connection.');
        return;
    }

    console.log('Calling ' + remoteId + '...');
    if (!localStream) {
        alert('Local stream not ready. Please ensure camera access is granted.');
        return;
    }

    try {
        const call = peer.call(remoteId, localStream);
        if (!call) {
            alert('Failed to initiate call. The peer may be unavailable.');
            return;
        }
        handleCall(call);
    } catch (err) {
        console.error('Error initiating call:', err);
        alert('Failed to start call: ' + (err.message || 'Unknown error'));
    }
});

videoMicBtn.addEventListener('click', async () => {
    if (!isRecording) {
        if (!localStream) {
            alert('Camera stream not active. Please switch to Video mode first.');
            return;
        }
        isVideoTranscription = true;

        // Clear previous video transcription text
        videoTranscriptionDisplay.textContent = "Transcribing...";

        await startRecording(localStream);
        videoMicBtn.innerHTML = `
            <svg class="w-5 h-5 animate-pulse" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="6" width="12" height="12" rx="2" />
            </svg>
            Stop
        `;
        videoMicBtn.classList.add('bg-red-600', 'hover:bg-red-500');
        videoMicBtn.classList.remove('bg-blue-600', 'hover:bg-blue-500');
    } else {
        stopRecording();
        videoMicBtn.innerHTML = `
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                    d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
            Transcribe
        `;
        videoMicBtn.classList.remove('bg-red-600', 'hover:bg-red-500');
        videoMicBtn.classList.add('bg-blue-600', 'hover:bg-blue-500');
    }
});

copyPeerIdBtn.addEventListener('click', () => {
    const id = myPeerIdDisplay.textContent;
    if (id && id !== 'Initializing...') {
        navigator.clipboard.writeText(id).then(() => {
            alert('Peer ID copied to clipboard!');
        });
    }
});

// Voice recording
micBtn.addEventListener('click', async () => {
    if (!isRecording) {
        await startRecording();
    } else {
        stopRecording();
    }
});

async function startRecording(providedStream = null) {
    try {
        // Reset UI
        transcriptionReviewSection.classList.add('hidden');
        outputSection.classList.add('hidden');

        let stream;
        if (providedStream) {
            // When using the video stream, create a new stream with only the audio tracks
            // This prevents MediaRecorder errors when mimeType is set to 'audio/webm'
            const audioTracks = providedStream.getAudioTracks();
            if (audioTracks.length === 0) {
                throw new Error('No audio track found in the video stream.');
            }
            stream = new MediaStream(audioTracks);
        } else {
            stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
        }

        // Detect supported MIME type
        const types = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/ogg;codecs=opus',
            'audio/mp4'
        ];
        let supportedType = '';
        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                supportedType = type;
                break;
            }
        }

        if (!supportedType) {
            throw new Error('No supported MediaRecorder MIME type found in this browser.');
        }

        mediaRecorder = new MediaRecorder(stream, {
            mimeType: supportedType
        });

        audioChunks = [];
        // Only track if we created the stream specifically for this (standalone mic)
        currentRecordingStream = providedStream ? null : stream;

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            // Stop all tracks IF we created the stream specifically for this
            if (currentRecordingStream) {
                currentRecordingStream.getTracks().forEach(track => track.stop());
                currentRecordingStream = null;
            }

            // Process the audio (transcription only)
            await transcribeAudio();
        };

        mediaRecorder.start(100); // Collect data every 100ms
        isRecording = true;

        // Update UI for standalone mic ONLY (video section has its own toggle)
        if (!providedStream) {
            micBtn.classList.add('recording');
            micIcon.classList.add('hidden');
            stopIcon.classList.remove('hidden');
            micStatus.textContent = 'Recording... Click to stop';
            audioWave.classList.remove('hidden');
        }

    } catch (error) {
        console.error('Error starting recording:', error);
        alert('Could not start recording: ' + error.message);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        isRecording = false;

        // Update UI for standalone mic
        micBtn.classList.remove('recording');
        micIcon.classList.remove('hidden');
        stopIcon.classList.add('hidden');
        micStatus.textContent = 'Processing...';
        audioWave.classList.add('hidden');
    }
}

async function transcribeAudio() {
    try {
        // Create audio blob
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });

        // Convert to base64
        const base64Audio = await blobToBase64(audioBlob);

        // Send to backend for transcription only
        const res = await fetch(TRANSCRIBE_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                audio_data: base64Audio,
                mime_type: 'audio/webm'
            })
        });

        if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            throw new Error(errorData.detail || "Transcription failed");
        }

        const data = await res.json();

        // Hide processing status
        micStatus.textContent = 'Click to start recording';

        // Show transcription
        if (data.transcription) {
            transcriptionInput.value = data.transcription;

            if (isVideoTranscription) {
                videoTranscriptionDisplay.textContent = data.transcription;
            } else {
                transcriptionReviewSection.classList.remove('hidden');
            }

            // If it's video mode, trigger translation automatically
            if (isVideoTranscription) {
                // We keep isVideoTranscription true so that renderResult knows where to go
                confirmTranscriptionBtn.click();
            }
        } else {
            alert('No speech detected. Please try again.');
            isVideoTranscription = false;
        }

    } catch (error) {
        console.error('Error transcribing audio:', error);
        alert('Error transcribing audio: ' + error.message);
        micStatus.textContent = 'Click to start recording';
    }
}

// Confirm transcription and translate
confirmTranscriptionBtn.addEventListener('click', async () => {
    const text = transcriptionInput.value.trim();
    if (!text) {
        alert('Please enter some text to translate');
        return;
    }

    const isForVideo = isVideoTranscription;
    setConfirmLoading(true);

    try {
        const res = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        if (!res.ok) throw new Error("Translation failed");

        const data = await res.json();
        renderResult(data, isForVideo);

    } catch (e) {
        alert("Translation Failed: " + e.message);
    } finally {
        setConfirmLoading(false);
        if (isForVideo) isVideoTranscription = false; // Reset after use
    }
});

// Retry recording
retryRecordingBtn.addEventListener('click', () => {
    transcriptionReviewSection.classList.add('hidden');
    transcriptionInput.value = '';
    outputSection.classList.add('hidden');
});

function setConfirmLoading(loading) {
    confirmTranscriptionBtn.disabled = loading;
    confirmTranscriptionBtn.innerHTML = loading
        ? `<svg class="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
             <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
             <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
           </svg>
           Translating...`
        : `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
           </svg>
           Confirm & Translate`;
    confirmTranscriptionBtn.classList.toggle('opacity-50', loading);
}

function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64 = reader.result.split(',')[1];
            resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

replayBtn.addEventListener('click', () => {
    if (currentPlan.length > 0 && !isPlaying) {
        playSequence(currentPlan);
    }
});

function renderResult(data, isForVideo = false) {
    if (!isForVideo) {
        outputSection.classList.remove('hidden');
        glossDisplay.innerHTML = "";
        data.gloss.forEach(token => {
            const badge = document.createElement('span');
            badge.className = "bg-teal-500/20 text-teal-300 px-2 py-1 rounded text-sm font-bold border border-teal-500/30";
            badge.textContent = token;
            glossDisplay.appendChild(badge);
        });

        let notes = "";
        if (data.unmatched.length > 0) notes += `Unmatched: ${data.unmatched.join(", ")}. `;
        if (data.notes) notes += data.notes;
        statusNotes.textContent = notes;
    }

    currentPlan = data.plan;
    playSequence(currentPlan, isForVideo);
}

async function playSequence(plan, isForVideo = false) {
    if (isPlaying) return;
    isPlaying = true;

    const currentPlayer = isForVideo ? videoSignPlayer : signPlayer;
    const currentPlaceholder = isForVideo ? videoPlayerPlaceholder : placeholder;
    const currentLabel = isForVideo ? videoPlayerLabel : playerLabel;

    currentPlaceholder.classList.add('hidden');
    currentPlayer.classList.remove('hidden');
    currentLabel.classList.remove('hidden');

    for (const item of plan) {
        if (item.type === 'sign' && item.assets.gif) {
            currentLabel.textContent = item.token;
            const baseUrl = "http://127.0.0.1:8000" + item.assets.gif;
            currentPlayer.src = baseUrl;
            await new Promise(r => setTimeout(r, 2000));
        } else {
            currentLabel.textContent = item.token + " (No Asset)";
            currentPlayer.src = "";
            await new Promise(r => setTimeout(r, 1500));
        }
    }

    isPlaying = false;
    currentLabel.textContent = "DONE";
    setTimeout(() => {
        currentLabel.classList.add('hidden');
    }, 1000);
}
