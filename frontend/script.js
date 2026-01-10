const BACKEND_URL = window.location.port === '3000' ? `http://${window.location.hostname}:8000` : '';
const API_URL = `${BACKEND_URL}/api/translate`;
const TRANSCRIBE_URL = `${BACKEND_URL}/api/transcribe`;

console.log("Unmute Frontend Initialized. Backend URL:", BACKEND_URL || "Relative");

// DOM Elements
const portalSection = document.getElementById('portalSection');
const lobbySection = document.getElementById('lobbySection');
const meetingSection = document.getElementById('meetingSection');

const displayNameInput = document.getElementById('displayName');
const enterBtn = document.getElementById('enterBtn');
const userGreeting = document.getElementById('userGreeting');
const displayRoomId = document.getElementById('displayRoomId');
const videoGrid = document.getElementById('videoGrid');
const localVideo = document.getElementById('localVideo');
const transcriptionText = document.getElementById('transcriptionText');
const speakerName = document.getElementById('speakerName');
const interpreterGif = document.getElementById('interpreterGif');
const interpreterPlaceholder = document.getElementById('interpreterPlaceholder');
const transcriptionHistory = document.getElementById('transcriptionHistory');

// App State
let currentUser = localStorage.getItem('displayName');
let currentRoomId = null;
let localStream = null;
let myPeer = null;
let peers = {};
let socket = null;
let transcriptionInterval = null;

// --- Portal Entry Logic ---

enterBtn.onclick = () => {
    const name = displayNameInput.value.trim();
    if (name) {
        currentUser = name;
        localStorage.setItem('displayName', name);
        showLobby();
    } else {
        alert("Please enter a name to continue");
    }
};

document.getElementById('exitBtn').onclick = () => {
    localStorage.removeItem('displayName');
    location.reload();
};

function showLobby() {
    portalSection.classList.add('hidden');
    lobbySection.classList.remove('hidden');
    userGreeting.textContent = currentUser;
}

// --- Lobby Logic ---

document.getElementById('createMeetingBtn').onclick = async () => {
    try {
        const res = await fetch(`${BACKEND_URL}/api/rooms/create`, {
            method: 'POST'
        });
        if (res.ok) {
            const data = await res.json();
            startMeeting(data.room_id);
        } else {
            console.error("Failed to create room");
        }
    } catch (e) {
        console.error("Error creating meeting:", e);
    }
};

document.getElementById('joinBtn').onclick = async () => {
    const roomId = document.getElementById('joinRoomId').value.trim().toUpperCase();
    if (!roomId) return;

    try {
        const res = await fetch(`${BACKEND_URL}/api/rooms/join/${roomId}`);
        if (res.ok) {
            startMeeting(roomId);
        } else {
            alert("Meeting room not found");
        }
    } catch (e) {
        console.error("Error joining meeting:", e);
    }
};

// --- Meeting Logic (WebRTC & SL) ---

async function startMeeting(roomId) {
    currentRoomId = roomId;
    displayRoomId.textContent = roomId;
    lobbySection.classList.add('hidden');
    meetingSection.classList.remove('hidden');

    try {
        localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        addVideoStream(localVideo, localStream, 'localVideoContainer', 'You');

        // Initialize Peer
        myPeer = new Peer();

        myPeer.on('open', id => {
            console.log("My Peer ID:", id);
            connectWebSocket(id);
        });

        myPeer.on('call', call => {
            call.answer(localStream);
            const video = document.createElement('video');
            call.on('stream', userVideoStream => {
                // We'll get the name via signaling soon
                addVideoStream(video, userVideoStream, call.peer, 'User');
            });
        });

        startSLIntelligence();

    } catch (e) {
        console.error("Failed to start meeting:", e);
        alert("Camera/Mic access required");
        location.reload();
    }
}

function connectWebSocket(peerId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.port === '3000' ? '127.0.0.1:8000' : window.location.host;
    socket = new WebSocket(`${protocol}//${wsHost}/ws/meeting/${currentRoomId}`);

    socket.onopen = () => {
        socket.send(JSON.stringify({
            type: 'join',
            peerId: peerId,
            username: currentUser
        }));
    };

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'join' && data.peerId !== peerId) {
            // New user joined, call them
            setTimeout(() => connectToNewUser(data.peerId, data.username, localStream), 1000);
        }

        if (data.type === 'sl-update') {
            showSLOverlay(data.peerId, data.username, data.transcription, data.plan);
        }
    };
}

function connectToNewUser(peerId, name, stream) {
    const call = myPeer.call(peerId, stream);
    const video = document.createElement('video');
    call.on('stream', userVideoStream => {
        addVideoStream(video, userVideoStream, peerId, name);
    });
    call.on('close', () => {
        const container = document.getElementById(peerId);
        if (container) container.remove();
    });
    peers[peerId] = call;
}

function addVideoStream(video, stream, containerId, labelText) {
    let container = document.getElementById(containerId);
    if (!container) {
        container = document.createElement('div');
        container.id = containerId;
        container.className = 'video-container';

        const label = document.createElement('div');
        label.className = 'participant-label';
        label.textContent = labelText;

        const slOverlay = document.createElement('div');
        slOverlay.className = 'sl-overlay';
        slOverlay.id = `sl-${containerId}`;
        const slImg = document.createElement('img');
        slOverlay.appendChild(slImg);

        container.appendChild(video);
        container.appendChild(label);
        container.appendChild(slOverlay);
        videoGrid.append(container);
    } else {
        // Update label if it's already there but just "User"
        const label = container.querySelector('.participant-label');
        if (label && label.textContent === 'User' && labelText !== 'User') {
            label.textContent = labelText;
        }
    }

    video.srcObject = stream;
    video.addEventListener('loadedmetadata', () => {
        video.play();
    });
}

// --- SL Intelligence Logic ---

async function startSLIntelligence() {
    console.log("SL Intelligence: Initializing MediaRecorder...");

    // Detect supported mime types
    let mimeType = 'audio/webm';
    if (typeof MediaRecorder.isTypeSupported === 'function') {
        if (!MediaRecorder.isTypeSupported(mimeType)) {
            console.log("audio/webm not supported, trying ogg...");
            mimeType = 'audio/ogg';
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                console.log("audio/ogg not supported, using default...");
                mimeType = '';
            }
        }
    }

    let mediaRecorder;
    try {
        const options = mimeType ? { mimeType } : {};
        mediaRecorder = new MediaRecorder(localStream, options);
    } catch (e) {
        console.error("Failed to create MediaRecorder:", e);
        return;
    }

    let audioChunks = [];

    mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
            audioChunks.push(e.data);
        }
    };

    mediaRecorder.onstop = async () => {
        if (audioChunks.length === 0) {
            console.log("SL Intelligence: No audio data captured in this chunk.");
            restartRecorder();
            return;
        }

        try {
            console.log(`SL Intelligence: Processing ${audioChunks.length} chunks...`);
            const blob = new Blob(audioChunks, { type: mimeType || 'audio/webm' });
            audioChunks = [];

            const base64Audio = await blobToBase64(blob);
            console.log(`SL Intelligence: Sending transcription request to ${TRANSCRIBE_URL}...`);

            const transRes = await fetch(TRANSCRIBE_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ audio_data: base64Audio })
            });

            if (transRes.ok) {
                const transData = await transRes.json();
                const text = transData.transcription;
                console.log("SL Intelligence Transcription result:", text);

                if (text && text.trim().length > 0) {
                    addToHistory(currentUser, text);

                    console.log("SL Intelligence: Translating to SL...");
                    const slRes = await fetch(API_URL, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text })
                    });

                    if (slRes.ok) {
                        const slData = await slRes.json();
                        console.log("SL Intelligence: SL Plan received:", slData.plan);

                        socket.send(JSON.stringify({
                            type: 'sl-update',
                            peerId: myPeer.id,
                            username: currentUser,
                            transcription: text,
                            plan: slData.plan
                        }));
                        showSLOverlay(myPeer.id, currentUser, text, slData.plan);
                    }
                }
            } else {
                const errData = await transRes.json().catch(() => ({}));
                console.error("SL Intelligence Transcription API Error:", errData.detail || "Unknown error");
                interpreterPlaceholder.textContent = `API Error: ${errData.detail || 'Failed to reach backend'}`;
                interpreterPlaceholder.classList.add('text-red-400');
            }
        } catch (err) {
            console.error("SL Intelligence Loop Error:", err);
        } finally {
            restartRecorder();
        }
    };

    function restartRecorder() {
        if (currentRoomId && mediaRecorder.state === 'inactive') {
            try {
                mediaRecorder.start();
                console.log("SL Intelligence: Recorder restarted for next chunk.");
            } catch (e) {
                console.error("Failed to restart MediaRecorder:", e);
            }
        }
    }

    // Start initial recording
    try {
        mediaRecorder.start();
        console.log("SL Intelligence: Loop started successfully.");
    } catch (e) {
        console.error("Failed to start MediaRecorder:", e);
    }

    transcriptionInterval = setInterval(() => {
        if (mediaRecorder.state === 'recording') {
            console.log("SL Intelligence Heartbeat: Stopping recorder to process chunk...");
            mediaRecorder.stop();
        } else if (mediaRecorder.state === 'inactive') {
            restartRecorder();
        }
    }, 2000);
}

function showSLOverlay(peerId, name, text, plan) {
    speakerName.textContent = `${name}: `;
    transcriptionText.textContent = text;

    // Update main interpreter panel
    interpreterPlaceholder.style.display = 'none';
    playSLSequence(interpreterGif, plan);

    const containerId = peerId === myPeer.id ? 'localVideoContainer' : peerId;
    const overlay = document.querySelector(`#sl-${containerId}`);
    if (!overlay) return;

    const img = overlay.querySelector('img');
    overlay.style.display = 'block';
    playSLSequence(img, plan, overlay);
}

async function playSLSequence(imgElem, plan, overlayElem = null) {
    for (const item of plan) {
        if (item.type === 'sign' && item.assets.gif) {
            const assetBase = window.location.port === '3000' ? 'http://127.0.0.1:8000' : window.location.origin;
            imgElem.src = assetBase + item.assets.gif;
            await new Promise(r => setTimeout(r, 2000));
        }
    }
    setTimeout(() => {
        if (overlayElem) overlayElem.style.display = 'none';
    }, 1000);
}

function addToHistory(name, text) {
    const item = document.createElement('div');
    item.className = 'bg-slate-900/80 p-3 rounded-xl border border-slate-800 animate-in fade-in slide-in-from-right-4 duration-300';
    item.innerHTML = `<span class="text-teal-400 font-bold text-xs uppercase block mb-1">${name}</span><p class="text-sm">${text}</p>`;
    transcriptionHistory.prepend(item);
}

function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result.split(',')[1]);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

// --- Controls ---

document.getElementById('toggleMic').onclick = () => {
    const audioTrack = localStream.getAudioTracks()[0];
    audioTrack.enabled = !audioTrack.enabled;
    document.getElementById('toggleMic').classList.toggle('off', !audioTrack.enabled);
};

document.getElementById('toggleCam').onclick = () => {
    const videoTrack = localStream.getVideoTracks()[0];
    videoTrack.enabled = !videoTrack.enabled;
    document.getElementById('toggleCam').classList.toggle('off', !videoTrack.enabled);
};

document.getElementById('leaveBtn').onclick = () => {
    location.reload();
};

// Initial Load
if (currentUser) {
    showLobby();
} else {
    portalSection.classList.remove('hidden');
}
