import { AvatarController } from './avatar.js';

const API_URL = "http://127.0.0.1:8000/api/translate";
const LANDMARKS_URL = "http://127.0.0.1:8000/api/sign";
const TRANSCRIBE_API_URL = "http://127.0.0.1:8000/api/transcribe";

// DOM Elements
const outputSection = document.getElementById('outputSection');
const outputSectionRight = document.getElementById('outputSectionRight');
const glossDisplay = document.getElementById('glossDisplay');
const statusNotes = document.getElementById('statusNotes');
const replayBtn = document.getElementById('replayBtn');
const signPlayer = document.getElementById('signPlayer');
const placeholder = document.getElementById('placeholder');
const playerLabel = document.getElementById('playerLabel');

// Initialize 3D Avatar
const avatar = new AvatarController('avatarContainer');

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
let currentGifCancel = null; // Function to cancel current GIF playback
let currentPlaybackId = 0; // Increment on each new request to cancel ongoing loops

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
        if (outputSectionRight) outputSectionRight.classList.add('hidden');

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

            // Process the audio (transcription and auto-translation)
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
        micStatus.textContent = 'Processing audio...';
        audioWave.classList.add('hidden');
    }
}

async function transcribeAudio() {
    try {
        // Update status to show processing
        micStatus.textContent = 'Transcribing & Translating...';

        // Create audio blob
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });

        // Convert to base64
        const base64Audio = await blobToBase64(audioBlob);

        // Send to backend with auto_translate enabled
        const res = await fetch(TRANSCRIBE_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                audio_data: base64Audio,
                mime_type: 'audio/webm',
                auto_translate: true  // Always enable auto-translate
            })
        });

        if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            throw new Error(errorData.detail || "Transcription failed");
        }

        const data = await res.json();

        // Hide processing status
        micStatus.textContent = 'Click to start recording';

        // Check if response contains full translation data
        if (data.plan && Array.isArray(data.plan) && data.gloss && Array.isArray(data.gloss)) {
            // Full translation received - skip review, show results directly
            transcriptionReviewSection.classList.add('hidden');
            outputSection.classList.remove('hidden');
            if (outputSectionRight) outputSectionRight.classList.remove('hidden');
            renderResult(data);
        } else if (data.transcription && data.transcription.trim()) {
            // Transcription only - show review step (fallback case)
            // This happens if auto-translate failed but transcription succeeded
            transcriptionInput.value = data.transcription;
            transcriptionReviewSection.classList.remove('hidden');
            outputSection.classList.add('hidden');
            if (outputSectionRight) outputSectionRight.classList.add('hidden');
        } else {
            // No transcription received or empty transcription
            alert('No speech detected. Please try again.');
            transcriptionReviewSection.classList.add('hidden');
            outputSection.classList.add('hidden');
            if (outputSectionRight) outputSectionRight.classList.add('hidden');
        }

    } catch (error) {
        console.error('Error transcribing audio:', error);
        alert('Error transcribing audio: ' + error.message);
        micStatus.textContent = 'Click to start recording';
        transcriptionReviewSection.classList.add('hidden');
        outputSection.classList.add('hidden');
        if (outputSectionRight) outputSectionRight.classList.add('hidden');
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

        // Hide transcription review section after successful translation
        transcriptionReviewSection.classList.add('hidden');
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
    if (outputSectionRight) outputSectionRight.classList.add('hidden');
});

function setConfirmLoading(loading) {
    confirmTranscriptionBtn.disabled = loading;
    confirmTranscriptionBtn.innerHTML = loading
        ? `<svg class="animate-spin" width="18" height="18" fill="none" viewBox="0 0 24 24">
             <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
             <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
           </svg>
           Translating...`
        : `<svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7"/>
           </svg>
           Confirm`;
    confirmTranscriptionBtn.style.opacity = loading ? '0.6' : '1';
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

function setLoading(loading) {
    translateBtn.disabled = loading;
    translateBtn.textContent = loading ? "Translating..." : "Translate to SGSL";
    translateBtn.style.opacity = loading ? '0.6' : '1';
}

function renderResult(data) {
    outputSection.classList.remove('hidden');
    if (outputSectionRight) outputSectionRight.classList.remove('hidden');
    
    // Resize avatar canvas now that container is visible
    avatar.resize();

    // Safely extract data with defaults
    currentPlan = data.plan || [];
    const gloss = data.gloss || [];
    const unmatched = data.unmatched || [];
    const notes = data.notes || "";

    // Render Gloss with new styling
    glossDisplay.innerHTML = "";
    if (gloss.length > 0) {
        gloss.forEach(token => {
            const badge = document.createElement('span');
            badge.className = "gloss-token";
            badge.textContent = token;
            glossDisplay.appendChild(badge);
        });
    } else {
        glossDisplay.innerHTML = '<span class="status-notes">No gloss tokens found</span>';
    }

    // Notes
    let notesText = "";
    if (unmatched.length > 0) notesText += `Unmatched: ${unmatched.join(", ")}. `;
    if (notes) notesText += notes;
    statusNotes.textContent = notesText || "";

    // Start Playback with 3D Avatar if plan exists
    if (currentPlan.length > 0) {
        // Increment playback ID to cancel any ongoing loops from previous requests
        currentPlaybackId++;
        playSequence(currentPlan);
    } else {
        statusNotes.textContent = (statusNotes.textContent || "") + " No signs to display.";
    }
}

// Play a single sign (GIF + MediaPipe skeleton)
async function playSingleSign(item, playbackId) {
    // Check if playback was cancelled
    if (playbackId !== currentPlaybackId) {
        return false;
    }

    if (!item.sign_name) {
        return false;
    }

    console.log(`Playing sign: ${item.sign_name}`);

    // Show GIF - add timestamp to force reload
    if (item.assets && item.assets.gif) {
        const gifUrl = `http://127.0.0.1:8000${item.assets.gif}?t=${Date.now()}`;
        signPlayer.src = gifUrl;
        signPlayer.classList.remove('hidden');
        placeholder.classList.add('hidden');
        playerLabel.textContent = item.token;
        playerLabel.classList.remove('hidden');
    }

    try {
        const resp = await fetch(`${LANDMARKS_URL}/${item.sign_name}/landmarks`);
        if (resp.ok) {
            const data = await resp.json();
            
            // Prefer pose_frames if available, fallback to hand_frames
            let frames = null;
            if (data.pose_frames && Array.isArray(data.pose_frames) && data.pose_frames.length > 0) {
                frames = data.pose_frames;
                console.log(`Playing pose skeleton: ${item.token} (${frames.length} frames)`);
            } else if (data.hand_frames && Array.isArray(data.hand_frames) && data.hand_frames.length > 0) {
                frames = data.hand_frames;
                console.log(`Playing hand skeleton: ${item.token} (${frames.length} frames)`);
            } else if (data.frames && Array.isArray(data.frames) && data.frames.length > 0) {
                // Fallback to old format if new format not available
                frames = data.frames;
                console.log(`Playing skeleton (legacy format): ${item.token} (${frames.length} frames)`);
            }
            
            if (!frames) {
                console.warn(`No frame data available for ${item.sign_name}`);
                await new Promise(r => setTimeout(r, 2000));
                signPlayer.classList.add('hidden');
                signPlayer.src = '';
                return false;
            }

            console.log(`Playing: ${item.token} (${frames.length} total frames)`);

            // Create a cancellable GIF promise
            let gifCancel = null;
            const gifPromise = new Promise((resolve) => {
                const timeout = setTimeout(() => {
                    resolve(false); // GIF completed normally
                }, 4000); // 4s for GIF
                
                gifCancel = () => {
                    clearTimeout(timeout);
                    resolve(true); // GIF was cancelled
                };
            });
            
            // Store cancel function so it can be called if skeleton goes blank
            currentGifCancel = gifCancel;

            // Play skeleton and check if it has valid frames
            let hasValidSkeleton = false;
            try {
                const skeletonPromise = avatar.playSequence(frames, 10);  // 10fps skeleton
                hasValidSkeleton = await skeletonPromise;
            } catch (e) {
                console.error(`Error playing skeleton for ${item.token}:`, e);
                hasValidSkeleton = false; // Treat errors as blank
            }

            // If skeleton went blank or errored, cancel the GIF immediately
            if (!hasValidSkeleton) {
                console.log(`Skeleton went blank or errored for ${item.token}, stopping GIF`);
                if (gifCancel) {
                    gifCancel();
                }
                // Stop the GIF immediately
                signPlayer.classList.add('hidden');
                signPlayer.src = '';
                // Clear the cancel function
                currentGifCancel = null;
                return false;
            }

            // Wait for GIF to complete (or be cancelled)
            const gifWasCancelled = await gifPromise;
            
            // Clear the cancel function
            currentGifCancel = null;

            if (!gifWasCancelled) {
                console.log(`Finished: ${item.token}`);
            }

            // Hide GIF after both complete
            signPlayer.classList.add('hidden');
            signPlayer.src = '';
            return true;
        } else {
            console.warn(`No 3D data for ${item.sign_name}`);
            await new Promise(r => setTimeout(r, 4000));
            signPlayer.classList.add('hidden');
            signPlayer.src = '';
            return false;
        }
    } catch (e) {
        console.error("Fetch error", e);
        return false;
    }
}

async function playSequence(plan) {
    if (isPlaying) return;
    isPlaying = true;
    placeholder.classList.add('hidden');
    signPlayer.classList.remove('hidden');
    playerLabel.classList.remove('hidden');
    
    // Clear any previous GIF cancellation
    currentGifCancel = null;

    // Debug: log the plan
    console.log('Plan received:', plan.map(p => p.sign_name || p.token));

    // Filter to unique signs only (no consecutive duplicates)
    const uniquePlan = [];
    let lastSignName = null;
    for (const item of plan) {
        if (item.type === 'sign' && item.sign_name !== lastSignName) {
            uniquePlan.push(item);
            lastSignName = item.sign_name;
        } else if (item.type !== 'sign') {
            uniquePlan.push(item);
            lastSignName = null;
        }
    }

    console.log('Unique plan:', uniquePlan.map(p => p.sign_name || p.token));

    // Detect single sign vs multiple signs
    const signs = uniquePlan.filter(item => item.type === 'sign' && item.sign_name);
    const isSingleSign = signs.length === 1;

    if (isSingleSign) {
        // Single sign mode: loop until a new request is sent
        const playbackId = currentPlaybackId;
        const singleSign = signs[0];
        
        console.log(`Single sign detected: ${singleSign.sign_name}, starting loop`);
        
        while (playbackId === currentPlaybackId) {
            // Check if cancelled before each iteration
            if (playbackId !== currentPlaybackId) {
                break;
            }
            
            await playSingleSign(singleSign, playbackId);
            
            // Brief pause between loops (only if still same playback session)
            if (playbackId === currentPlaybackId) {
                await new Promise(r => setTimeout(r, 300));
            }
        }
        
        console.log('Single sign loop ended (new request received)');
    } else {
        // Multiple signs mode: sequential playback (existing behavior)
        for (let i = 0; i < uniquePlan.length; i++) {
            const item = uniquePlan[i];

            if (item.type === 'sign' && item.sign_name) {
                console.log(`Starting sign ${i + 1}/${uniquePlan.length}: ${item.sign_name}`);
                
                await playSingleSign(item, currentPlaybackId);

                // Brief pause between words
                if (i < uniquePlan.length - 1) {
                    await new Promise(r => setTimeout(r, 300));
                }

            } else {
                console.log(`Skipping non-sign: ${item.token}`);
                await new Promise(r => setTimeout(r, 500));
            }
        }
    }

    // Final cleanup
    placeholder.classList.remove('hidden');
    playerLabel.classList.add('hidden');
    // Ensure GIF is stopped
    if (currentGifCancel) {
        currentGifCancel();
        currentGifCancel = null;
    }
    signPlayer.classList.add('hidden');
    signPlayer.src = '';
    isPlaying = false;
    console.log('Sequence complete');
}
