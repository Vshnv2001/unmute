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
const localVideo = document.getElementById('localVideo');
const remoteVideo = document.getElementById('remoteVideo');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const copyPeerIdBtn = document.getElementById('copyPeerId');

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

// PeerJS state
let peer = null;
let localStream = null;
let currentCall = null;

// Initialization
initPeer();

function initPeer() {
    peer = new Peer({
        host: '/',
        port: 8000,
        path: '/peerjs', // Note: This might need backend support if not using a public peer server
        debug: 3
    });

    // Fallback to public server if local fails (simpler for demo)
    peer = new Peer();

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
            });
    });

    peer.on('error', (err) => {
        console.error('PeerJS error:', err);
        alert('PeerJS error: ' + err.type);
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
        remoteVideo.srcObject = null;
        videoPlaceholder.classList.remove('hidden');
    });
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

    console.log('Calling ' + remoteId + '...');
    if (!localStream) {
        alert('Local stream not ready. Please ensure camera access is granted.');
        return;
    }

    const call = peer.call(remoteId, localStream);
    handleCall(call);
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

async function startRecording() {
    try {
        // Reset UI
        transcriptionReviewSection.classList.add('hidden');
        outputSection.classList.add('hidden');

        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            }
        });

        mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm;codecs=opus'
        });

        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());

            // Process the audio (transcription only)
            await transcribeAudio();
        };

        mediaRecorder.start(100); // Collect data every 100ms
        isRecording = true;

        // Update UI
        micBtn.classList.add('recording');
        micIcon.classList.add('hidden');
        stopIcon.classList.remove('hidden');
        micStatus.textContent = 'Recording... Click to stop';
        audioWave.classList.remove('hidden');

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please ensure you have granted permission.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        isRecording = false;

        // Update UI
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

        // Show transcription for review
        if (data.transcription) {
            transcriptionInput.value = data.transcription;
            transcriptionReviewSection.classList.remove('hidden');
        } else {
            alert('No speech detected. Please try again.');
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

    setConfirmLoading(true);

    try {
        const res = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        if (!res.ok) throw new Error("Translation failed");

        const data = await res.json();
        renderResult(data);

    } catch (e) {
        alert("Translation Failed: " + e.message);
    } finally {
        setConfirmLoading(false);
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

function renderResult(data) {
    outputSection.classList.remove('hidden');
    currentPlan = data.plan;

    // Render Gloss
    glossDisplay.innerHTML = "";
    data.gloss.forEach(token => {
        const badge = document.createElement('span');
        badge.className = "bg-teal-500/20 text-teal-300 px-2 py-1 rounded text-sm font-bold border border-teal-500/30";
        badge.textContent = token;
        glossDisplay.appendChild(badge);
    });

    // Notes
    let notes = "";
    if (data.unmatched.length > 0) notes += `Unmatched: ${data.unmatched.join(", ")}. `;
    if (data.notes) notes += data.notes;
    statusNotes.textContent = notes;

    // Start Playback
    playSequence(currentPlan);
}

async function playSequence(plan) {
    if (isPlaying) return;
    isPlaying = true;
    placeholder.classList.add('hidden');
    signPlayer.classList.remove('hidden');
    playerLabel.classList.remove('hidden');

    for (const item of plan) {
        if (item.type === 'sign' && item.assets.gif) {
            playerLabel.textContent = item.token;
            const baseUrl = "http://127.0.0.1:8000" + item.assets.gif;
            signPlayer.src = baseUrl;
            await new Promise(r => setTimeout(r, 2000));
        } else {
            playerLabel.textContent = item.token + " (No Asset)";
            signPlayer.src = "";
            await new Promise(r => setTimeout(r, 1500));
        }
    }

    isPlaying = false;
    playerLabel.textContent = "DONE";
    setTimeout(() => {
        playerLabel.classList.add('hidden');
    }, 1000);
}
