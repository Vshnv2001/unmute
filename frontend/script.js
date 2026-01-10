const API_URL = "http://127.0.0.1:8000/api/translate";
const POSE_API_URL = "http://127.0.0.1:8000/api/pose";

const inputText = document.getElementById('inputText');
const translateBtn = document.getElementById('translateBtn');
const outputSection = document.getElementById('outputSection');
const glossDisplay = document.getElementById('glossDisplay');
const signPlayer = document.getElementById('signPlayer');
const placeholder = document.getElementById('placeholder');
const statusNotes = document.getElementById('statusNotes');
const playerLabel = document.getElementById('playerLabel');
const replayBtn = document.getElementById('replayBtn');
const poseCanvas = document.getElementById('poseCanvas');
const poseLabel = document.getElementById('poseLabel');

let currentPlan = [];
let isPlaying = false;
let poseVisualizer = null;

// PoseVisualizer class for rendering 3D hand landmarks
class PoseVisualizer {
    constructor(canvasElement) {
        this.canvas = canvasElement;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.handObjects = { left: null, right: null };
        this.currentFrames = null;
        this.animationFrameId = null;
        this.init();
    }

    init() {
        if (!this.canvas || typeof THREE === 'undefined') {
            console.error('Canvas element or Three.js not available');
            return;
        }

        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);

        // Camera setup
        const aspect = this.canvas.clientWidth / this.canvas.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        this.camera.position.set(0, 0, 2);
        this.camera.lookAt(0, 0, 0);

        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
        this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        this.scene.add(directionalLight);

        // Handle resize
        window.addEventListener('resize', () => this.handleResize());
    }

    handleResize() {
        if (!this.camera || !this.renderer || !this.canvas) return;
        const aspect = this.canvas.clientWidth / this.canvas.clientHeight;
        this.camera.aspect = aspect;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
    }

    // MediaPipe hand landmark connections
    getHandConnections() {
        return [
            // Wrist to finger bases
            [0, 1], [0, 5], [0, 9], [0, 13], [0, 17],
            // Thumb
            [1, 2], [2, 3], [3, 4],
            // Index
            [5, 6], [6, 7], [7, 8],
            // Middle
            [9, 10], [10, 11], [11, 12],
            // Ring
            [13, 14], [14, 15], [15, 16],
            // Pinky
            [17, 18], [18, 19], [19, 20]
        ];
    }

    createHandGroup(landmarks, color, side) {
        const group = new THREE.Group();
        const connections = this.getHandConnections();

        // Check if hand is present (not all zeros)
        const isPresent = landmarks.some(lm => 
            Math.abs(lm[0]) > 0.001 || Math.abs(lm[1]) > 0.001 || Math.abs(lm[2]) > 0.001
        );

        if (!isPresent) {
            return group; // Return empty group
        }

        // Create spheres for landmarks
        const sphereGeometry = new THREE.SphereGeometry(0.02, 8, 8);
        const sphereMaterial = new THREE.MeshPhongMaterial({ color });
        const spheres = [];

        landmarks.forEach((lm, idx) => {
            const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial.clone());
            sphere.position.set(lm[0], -lm[1], lm[2]); // Flip Y for screen coordinates
            sphere.userData.landmarkIndex = idx;
            group.add(sphere);
            spheres.push(sphere);
        });

        // Create lines for connections
        const lineMaterial = new THREE.LineBasicMaterial({ color, linewidth: 2 });
        connections.forEach(([start, end]) => {
            if (spheres[start] && spheres[end]) {
                const geometry = new THREE.BufferGeometry().setFromPoints([
                    spheres[start].position,
                    spheres[end].position
                ]);
                const line = new THREE.Line(geometry, lineMaterial);
                group.add(line);
            }
        });

        return group;
    }

    renderFrame(frameIndex) {
        if (!this.currentFrames || frameIndex < 0 || frameIndex >= this.currentFrames.length) {
            return;
        }

        // Clear existing hands
        if (this.handObjects.left) {
            this.scene.remove(this.handObjects.left);
        }
        if (this.handObjects.right) {
            this.scene.remove(this.handObjects.right);
        }

        const frame = this.currentFrames[frameIndex];
        if (!frame) return;

        // Create hand groups
        const leftColor = 0x00ff00; // Green for left
        const rightColor = 0xff0000; // Red for right
        
        this.handObjects.left = this.createHandGroup(frame.left, leftColor, 'left');
        this.handObjects.right = this.createHandGroup(frame.right, rightColor, 'right');

        // Position hands side by side (offset right hand to the right)
        if (this.handObjects.right) {
            this.handObjects.right.position.x += 0.5;
        }
        if (this.handObjects.left) {
            this.handObjects.left.position.x -= 0.5;
        }

        this.scene.add(this.handObjects.left);
        this.scene.add(this.handObjects.right);

        // Render
        this.renderer.render(this.scene, this.camera);
    }

    async loadPoseData(signName) {
        try {
            const response = await fetch(`${POSE_API_URL}/${signName}`);
            if (!response.ok) {
                throw new Error(`Failed to load pose data: ${response.status}`);
            }
            const data = await response.json();
            this.currentFrames = data.frames || [];
            return data;
        } catch (error) {
            console.error('Error loading pose data:', error);
            this.currentFrames = null;
            return null;
        }
    }

    async animateSequence(frames, duration) {
        if (!frames || frames.length === 0) return;

        const frameCount = frames.length;
        const frameDuration = duration / frameCount;
        let currentFrame = 0;
        const startTime = Date.now();

        const animate = () => {
            const elapsed = Date.now() - startTime;
            const targetFrame = Math.floor((elapsed / duration) * frameCount);
            
            if (targetFrame < frameCount) {
                if (targetFrame !== currentFrame) {
                    currentFrame = targetFrame;
                    this.renderFrame(currentFrame);
                }
                this.animationFrameId = requestAnimationFrame(animate);
            } else {
                // Animation complete, render last frame
                this.renderFrame(frameCount - 1);
            }
        };

        // Start animation
        this.animationFrameId = requestAnimationFrame(animate);
    }

    stopAnimation() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    clear() {
        this.stopAnimation();
        if (this.handObjects.left) {
            this.scene.remove(this.handObjects.left);
            this.handObjects.left = null;
        }
        if (this.handObjects.right) {
            this.scene.remove(this.handObjects.right);
            this.handObjects.right = null;
        }
        this.currentFrames = null;
        this.renderer.render(this.scene, this.camera);
    }
}

// Initialize pose visualizer when DOM and Three.js are ready
function initPoseVisualizer() {
    if (poseCanvas && typeof THREE !== 'undefined') {
        poseVisualizer = new PoseVisualizer(poseCanvas);
    } else if (poseCanvas && typeof THREE === 'undefined') {
        // Wait for Three.js to load
        setTimeout(initPoseVisualizer, 100);
    }
}

// Initialize after DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPoseVisualizer);
} else {
    initPoseVisualizer();
}

translateBtn.addEventListener('click', async () => {
    const text = inputText.value.trim();
    if (!text) return;

    setLoading(true);
    try {
        const res = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        
        if (!res.ok) throw new Error("API Error");
        
        const data = await res.json();
        renderResult(data);
    } catch (e) {
        alert("Translation Failed: " + e.message);
    } finally {
        setLoading(false);
    }
});

replayBtn.addEventListener('click', () => {
    if (currentPlan.length > 0 && !isPlaying) {
        playSequence(currentPlan);
    }
});

function setLoading(loading) {
    translateBtn.disabled = loading;
    translateBtn.textContent = loading ? "Translating..." : "Translate";
    translateBtn.classList.toggle('opacity-50', loading);
}

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
    
    // Show pose visualization canvas
    if (poseCanvas && poseLabel) {
        poseLabel.classList.remove('hidden');
    }

    for (const item of plan) {
        if (item.type === 'sign' && item.sign_name) {
            // Update UI
            playerLabel.textContent = item.token;
            
            // Show new GIF
            if (item.assets.gif) {
                const baseUrl = "http://127.0.0.1:8000" + item.assets.gif; 
                signPlayer.src = baseUrl;
            }
            
            // Load and animate pose data if available
            if (poseVisualizer && item.sign_name) {
                poseVisualizer.stopAnimation();
                const poseData = await poseVisualizer.loadPoseData(item.sign_name);
                if (poseData && poseData.frames && poseData.frames.length > 0) {
                    // Start pose animation (2 seconds duration per sign)
                    poseVisualizer.animateSequence(poseData.frames, 2000);
                } else {
                    poseVisualizer.clear();
                }
            }
            
            // Wait duration (2 seconds per sign)
            await new Promise(r => setTimeout(r, 2000));
        } else {
            // Text only fallback
            playerLabel.textContent = item.token + " (No Asset)";
            signPlayer.src = ""; // Clear or placeholder
            if (poseVisualizer) {
                poseVisualizer.stopAnimation();
                poseVisualizer.clear();
            }
            await new Promise(r => setTimeout(r, 1500));
        }
    }
    
    // Reset
    isPlaying = false;
    playerLabel.textContent = "DONE";
    if (poseVisualizer) {
        poseVisualizer.stopAnimation();
        // Keep last frame visible for a moment
        setTimeout(() => {
            if (poseVisualizer) {
                poseVisualizer.clear();
            }
            if (poseLabel) {
                poseLabel.classList.add('hidden');
            }
        }, 1000);
    } else {
        setTimeout(() => {
            playerLabel.classList.add('hidden');
            if (poseLabel) {
                poseLabel.classList.add('hidden');
            }
        }, 1000);
    }
}

// Sign to Text Logic
const signTranslateBtn = document.getElementById('signTranslateBtn');
const videoInput = document.getElementById('videoInput');
const signOutputSection = document.getElementById('signOutputSection');
const signResultsList = document.getElementById('signResultsList');
const SIGN_API_URL = "http://127.0.0.1:8000/api/sign2text";

// Webcam Elements
const webcamPreview = document.getElementById('webcamPreview');
const startRecordBtn = document.getElementById('startRecordBtn');
const stopRecordBtn = document.getElementById('stopRecordBtn');
const recordingIndicator = document.getElementById('recordingIndicator');

let mediaRecorder;
let recordedChunks = [];
let stream;

// Initialize Webcam on load with explicit 30 FPS and resolution
async function initWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: {
                frameRate: { ideal: 30, max: 30 },
                width: { ideal: 640 },
                height: { ideal: 480 }
            }, 
            audio: false 
        });
        webcamPreview.srcObject = stream;
        console.log("Webcam initialized at 30 FPS, 640x480");
    } catch (err) {
        console.error("Error accessing webcam:", err);
        alert("Could not access webcam. Please allow camera permissions.");
    }
}

// Start Recording
startRecordBtn.addEventListener('click', () => {
    if (!stream) {
        initWebcam();
        return;
    }
    
    recordedChunks = [];
    // Try to use a mimeType that is widely supported
    const mimeType = MediaRecorder.isTypeSupported("video/webm; codecs=vp9") 
        ? "video/webm; codecs=vp9" 
        : "video/webm";
        
    mediaRecorder = new MediaRecorder(stream, { mimeType });
    
    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };
    
    mediaRecorder.onstop = uploadRecording;
    
    mediaRecorder.start();
    
    // UI Updates
    startRecordBtn.disabled = true;
    startRecordBtn.classList.add('opacity-50', 'cursor-not-allowed');
    stopRecordBtn.disabled = false;
    stopRecordBtn.classList.remove('bg-slate-700', 'text-slate-400', 'cursor-not-allowed');
    stopRecordBtn.classList.add('bg-red-600', 'hover:bg-red-500', 'text-white');
    recordingIndicator.classList.remove('hidden');
});

// Stop Recording
stopRecordBtn.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        
        // UI Updates
        startRecordBtn.disabled = false;
        startRecordBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        stopRecordBtn.disabled = true;
        stopRecordBtn.classList.add('bg-slate-700', 'text-slate-400', 'cursor-not-allowed');
        stopRecordBtn.classList.remove('bg-red-600', 'hover:bg-red-500', 'text-white');
        recordingIndicator.classList.add('hidden');
    }
});

async function uploadRecording() {
    const blob = new Blob(recordedChunks, { type: "video/webm" });
    const file = new File([blob], "webcam_recording.webm", { type: "video/webm" });
    
    setSignLoading(true, "Processing Recording...");
    try {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch(SIGN_API_URL, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) throw new Error("API Error");
        
        const data = await res.json();
        renderSignResults(data.results);
    } catch (e) {
        alert("Sign Translation Failed: " + e.message);
    } finally {
        setSignLoading(false);
    }
}

// Initialize webcam immediately? Or wait for interactions?
// Let's init immediately for a "Live" feel
initWebcam();


signTranslateBtn.addEventListener('click', async () => {
    const file = videoInput.files[0];
    if (!file) {
        alert("Please select a video file first.");
        return;
    }

    setSignLoading(true);
    try {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch(SIGN_API_URL, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) throw new Error("API Error");
        
        const data = await res.json();
        renderSignResults(data.results);
    } catch (e) {
        alert("Sign Translation Failed: " + e.message);
    } finally {
        setSignLoading(false);
    }
});

function setSignLoading(loading, text) {
    // Handle both buttons loading state slightly differently if needed, 
    // but for now main concern is feedback.
    
    // If text provided, use it, else default
    const originalText = "Translate Upload";
    const label = text || "Analyzing...";
    
    signTranslateBtn.disabled = loading;
    signTranslateBtn.textContent = loading ? label : originalText;
    signTranslateBtn.classList.toggle('opacity-50', loading);
    
    if (loading) {
        startRecordBtn.disabled = true;
        startRecordBtn.classList.add('opacity-50');
    } else {
        startRecordBtn.disabled = false;
        startRecordBtn.classList.remove('opacity-50');
    }
}

function renderSignResults(results) {
    signOutputSection.classList.remove('hidden');
    signResultsList.innerHTML = "";
    
    if (!results || results.length === 0) {
        signResultsList.innerHTML = "<li class='text-slate-400'>No confident matches found. Try signing more clearly.</li>";
        return;
    }

    results.forEach(item => {
        const li = document.createElement('li');
        li.className = "flex justify-between items-center bg-slate-800 p-3 rounded hover:bg-slate-700 transition-colors";
        
        const label = document.createElement('span');
        label.className = "text-lg font-mono text-purple-300";
        label.textContent = item.label;
        
        const score = document.createElement('span');
        score.className = "text-xs text-emerald-400 font-medium";
        score.textContent = item.confidence || `${(item.similarity * 100).toFixed(1)}%`;
        
        li.appendChild(label);
        li.appendChild(score);
        signResultsList.appendChild(li);
    });
}
