const API_URL = "/api/translate";
const TRANSCRIBE_URL = "/api/transcribe";

// DOM Elements
const authSection = document.getElementById('authSection');
const lobbySection = document.getElementById('lobbySection');
const meetingSection = document.getElementById('meetingSection');

const loginForm = document.getElementById('loginForm');
const signupForm = document.getElementById('signupForm');
const showSignup = document.getElementById('showSignup');
const showLogin = document.getElementById('showLogin');

const userGreeting = document.getElementById('userGreeting');
const displayRoomId = document.getElementById('displayRoomId');
const videoGrid = document.getElementById('videoGrid');
const localVideo = document.getElementById('localVideo');
const transcriptionText = document.getElementById('transcriptionText');
const speakerName = document.getElementById('speakerName');

// Meeting State
let currentUser = null;
let currentToken = localStorage.getItem('token');
let currentRoomId = null;
let localStream = null;
let myPeer = null;
let peers = {};
let socket = null;
let transcriptionInterval = null;

// --- Authentication UI Logic ---

showSignup.onclick = () => { loginForm.classList.add('hidden'); signupForm.classList.remove('hidden'); };
showLogin.onclick = () => { signupForm.classList.add('hidden'); loginForm.classList.remove('hidden'); };

document.getElementById('signupBtn').onclick = async () => {
    const username = document.getElementById('signupUsername').value;
    const password = document.getElementById('signupPassword').value;
    const res = await fetch('/api/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
    });
    if (res.ok) {
        alert("Signup successful! Please login.");
        showLogin.onclick();
    } else {
        const data = await res.json();
        alert(data.detail || "Signup failed");
    }
};

document.getElementById('loginBtn').onclick = async () => {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    const res = await fetch('/api/token', {
        method: 'POST',
        body: formData
    });

    if (res.ok) {
        const data = await res.json();
        currentToken = data.access_token;
        localStorage.setItem('token', currentToken);
        initApp();
    } else {
        alert("Login failed");
    }
};

document.getElementById('logoutBtn').onclick = () => {
    localStorage.removeItem('token');
    location.reload();
};

// --- App Initialization ---

async function initApp() {
    if (!currentToken) {
        authSection.classList.remove('hidden');
        lobbySection.classList.add('hidden');
        return;
    }

    try {
        const res = await fetch('/api/rooms/join/test', { // Just checking auth
            headers: { 'Authorization': `Bearer ${currentToken}` }
        });

        if (res.status === 401) throw new Error("Unauthorized");

        // Success
        const data = await res.json();
        currentUser = data.username;
        userGreeting.textContent = currentUser;

        authSection.classList.add('hidden');
        lobbySection.classList.remove('hidden');

    } catch (e) {
        localStorage.removeItem('token');
        authSection.classList.remove('hidden');
    }
}

// --- Lobby Logic ---

document.getElementById('createMeetingBtn').onclick = async () => {
    const res = await fetch('/api/rooms/create', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${currentToken}` }
    });
    if (res.ok) {
        const data = await res.json();
        startMeeting(data.room_id);
    }
};

document.getElementById('joinBtn').onclick = () => {
    const roomId = document.getElementById('joinRoomId').value.trim();
    if (roomId) startMeeting(roomId);
};

// --- Meeting Logic (WebRTC & SL) ---

async function startMeeting(roomId) {
    currentRoomId = roomId;
    displayRoomId.textContent = roomId;
    lobbySection.classList.add('hidden');
    meetingSection.classList.remove('hidden');

    try {
        localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        addVideoStream(localVideo, localStream, 'localVideoContainer');

        // Initialize Peer
        myPeer = new Peer(undefined, {
            host: '/',
            port: '8000', // Uvicorn port
            path: '/peerjs' // We'll need to support this or use default PeerJS cloud
        });

        // Actually PeerJS server is not built-in, so for local dev without a dedicated PeerServer,
        // we'll use the default PeerJS cloud servers (remove parameters).
        myPeer = new Peer();

        myPeer.on('open', id => {
            console.log("My Peer ID:", id);
            connectWebSocket(id);
        });

        myPeer.on('call', call => {
            call.answer(localStream);
            const video = document.createElement('video');
            call.on('stream', userVideoStream => {
                addVideoStream(video, userVideoStream, call.peer);
            });
        });

        // Start SL Intelligence Loop
        startSLIntelligence();

    } catch (e) {
        console.error("Failed to start meeting:", e);
        alert("Camera/Mic access required");
    }
}

function connectWebSocket(peerId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(`${protocol}//${window.location.host}/ws/meeting/${currentRoomId}`);

    socket.onopen = () => {
        // Announce myself to the room
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
            setTimeout(() => connectToNewUser(data.peerId, localStream), 1000);
        }

        if (data.type === 'sl-update') {
            // Show SL overlay for a specific user
            showSLOverlay(data.peerId, data.username, data.transcription, data.plan);
        }
    };
}

function connectToNewUser(peerId, stream) {
    const call = myPeer.call(peerId, stream);
    const video = document.createElement('video');
    call.on('stream', userVideoStream => {
        addVideoStream(video, userVideoStream, peerId);
    });
    call.on('close', () => {
        const container = document.getElementById(peerId);
        if (container) container.remove();
    });
    peers[peerId] = call;
}

function addVideoStream(video, stream, containerId) {
    let container = document.getElementById(containerId);
    if (!container) {
        container = document.createElement('div');
        container.id = containerId;
        container.className = 'video-container';

        const label = document.createElement('div');
        label.className = 'participant-label';
        label.textContent = containerId === 'localVideoContainer' ? 'You' : 'User';

        const slOverlay = document.createElement('div');
        slOverlay.className = 'sl-overlay';
        slOverlay.id = `sl-${containerId}`;
        const slImg = document.createElement('img');
        slOverlay.appendChild(slImg);

        container.appendChild(video);
        container.appendChild(label);
        container.appendChild(slOverlay);
        videoGrid.append(container);
    }

    video.srcObject = stream;
    video.addEventListener('loadedmetadata', () => {
        video.play();
    });
}

// --- SL Intelligence Logic ---

async function startSLIntelligence() {
    const mediaRecorder = new MediaRecorder(localStream, { mimeType: 'audio/webm' });
    let audioChunks = [];

    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);

    mediaRecorder.onstop = async () => {
        const blob = new Blob(audioChunks, { type: 'audio/webm' });
        audioChunks = [];

        // 1. Transcribe
        const base64Audio = await blobToBase64(blob);
        const transRes = await fetch(TRANSCRIBE_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ audio_data: base64Audio })
        });

        if (transRes.ok) {
            const transData = await transRes.json();
            const text = transData.transcription;

            if (text && text.trim().length > 2) {
                // 2. Translate
                const slRes = await fetch(API_URL, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });

                if (slRes.ok) {
                    const slData = await slRes.json();

                    // 3. Broadcast to Room
                    socket.send(JSON.stringify({
                        type: 'sl-update',
                        peerId: myPeer.id,
                        username: currentUser,
                        transcription: text,
                        plan: slData.plan
                    }));

                    // Also show locally for feedback
                    showSLOverlay(myPeer.id, currentUser, text, slData.plan);
                }
            }
        }

        // Continue loop if still in meeting
        if (currentRoomId) mediaRecorder.start();
    };

    // Trigger every 4 seconds
    transcriptionInterval = setInterval(() => {
        if (mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
        } else {
            mediaRecorder.start();
        }
    }, 4000);
}

function showSLOverlay(peerId, name, text, plan) {
    // 1. Update Transcription Ticker
    speakerName.textContent = `${name}: `;
    transcriptionText.textContent = text;

    // 2. Play SL Sequence on the Overlay
    const containerId = peerId === myPeer.id ? 'localVideoContainer' : peerId;
    const overlay = document.querySelector(`#sl-${containerId}`);
    if (!overlay) return;

    const img = overlay.querySelector('img');
    overlay.style.display = 'block';

    // Play sequence
    playSLSequence(img, plan, overlay);
}

async function playSLSequence(imgElem, plan, overlayElem) {
    for (const item of plan) {
        if (item.type === 'sign' && item.assets.gif) {
            imgElem.src = window.location.origin + item.assets.gif;
            await new Promise(r => setTimeout(r, 2000));
        }
    }
    // Hide overlay after sequence ends (with small delay)
    setTimeout(() => {
        overlayElem.style.display = 'none';
    }, 1000);
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
initApp();
