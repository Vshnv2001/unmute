const API_URL = "http://127.0.0.1:8000/api/translate";
const TRANSCRIBE_API_URL = "http://127.0.0.1:8000/api/transcribe";

// DOM Elements
const inputText = document.getElementById('inputText');
const translateBtn = document.getElementById('translateBtn');
const outputSection = document.getElementById('outputSection');
const glossDisplay = document.getElementById('glossDisplay');
const signPlayer = document.getElementById('signPlayer');
const placeholder = document.getElementById('placeholder');
const statusNotes = document.getElementById('statusNotes');
const playerLabel = document.getElementById('playerLabel');
const replayBtn = document.getElementById('replayBtn');

// Voice mode elements
const textModeTab = document.getElementById('textModeTab');
const voiceModeTab = document.getElementById('voiceModeTab');
const textInputSection = document.getElementById('textInputSection');
const voiceInputSection = document.getElementById('voiceInputSection');
const micBtn = document.getElementById('micBtn');
const micIcon = document.getElementById('micIcon');
const stopIcon = document.getElementById('stopIcon');
const micStatus = document.getElementById('micStatus');
const audioWave = document.getElementById('audioWave');

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

// Mode switching
textModeTab.addEventListener('click', () => {
    textModeTab.classList.add('active');
    textModeTab.classList.remove('text-slate-400');
    voiceModeTab.classList.remove('active');
    voiceModeTab.classList.add('text-slate-400');
    textInputSection.classList.remove('hidden');
    voiceInputSection.classList.add('hidden');
});

voiceModeTab.addEventListener('click', () => {
    voiceModeTab.classList.add('active');
    voiceModeTab.classList.remove('text-slate-400');
    textModeTab.classList.remove('active');
    textModeTab.classList.add('text-slate-400');
    voiceInputSection.classList.remove('hidden');
    textInputSection.classList.add('hidden');
});

// Text translation
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
            renderResult(data);
        } else if (data.transcription && data.transcription.trim()) {
            // Transcription only - show review step (fallback case)
            // This happens if auto-translate failed but transcription succeeded
            transcriptionInput.value = data.transcription;
            transcriptionReviewSection.classList.remove('hidden');
            outputSection.classList.add('hidden');
        } else {
            // No transcription received or empty transcription
            alert('No speech detected. Please try again.');
            transcriptionReviewSection.classList.add('hidden');
            outputSection.classList.add('hidden');
        }

    } catch (error) {
        console.error('Error transcribing audio:', error);
        alert('Error transcribing audio: ' + error.message);
        micStatus.textContent = 'Click to start recording';
        transcriptionReviewSection.classList.add('hidden');
        outputSection.classList.add('hidden');
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
            // Remove the data URL prefix (e.g., "data:audio/webm;base64,")
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
    translateBtn.textContent = loading ? "Translating..." : "Translate";
    translateBtn.classList.toggle('opacity-50', loading);
}

function renderResult(data) {
    outputSection.classList.remove('hidden');
    
    // Safely extract data with defaults
    currentPlan = data.plan || [];
    const gloss = data.gloss || [];
    const unmatched = data.unmatched || [];
    const notes = data.notes || "";

    // Render Gloss
    glossDisplay.innerHTML = "";
    if (gloss.length > 0) {
        gloss.forEach(token => {
            const badge = document.createElement('span');
            badge.className = "bg-teal-500/20 text-teal-300 px-2 py-1 rounded text-sm font-bold border border-teal-500/30";
            badge.textContent = token;
            glossDisplay.appendChild(badge);
        });
    } else {
        glossDisplay.innerHTML = '<span class="text-slate-500 italic">No gloss tokens found</span>';
    }

    // Notes
    let notesText = "";
    if (unmatched.length > 0) notesText += `Unmatched: ${unmatched.join(", ")}. `;
    if (notes) notesText += notes;
    statusNotes.textContent = notesText || "";

    // Start Playback if plan exists
    if (currentPlan.length > 0) {
        playSequence(currentPlan);
    } else {
        statusNotes.textContent = (statusNotes.textContent || "") + " No signs to display.";
    }
}

async function playSequence(plan) {
    if (isPlaying) return;
    isPlaying = true;
    placeholder.classList.add('hidden');
    signPlayer.classList.remove('hidden');
    playerLabel.classList.remove('hidden');

    for (const item of plan) {
        if (item.type === 'sign' && item.assets.gif) {
            // Update UI
            playerLabel.textContent = item.token;

            // Show new GIF
            // Hack to force reload gif if same src
            const baseUrl = "http://127.0.0.1:8000" + item.assets.gif;
            signPlayer.src = baseUrl;

            // Wait duration (2 seconds per sign)
            await new Promise(r => setTimeout(r, 2000));
        } else {
            // Text only fallback
            playerLabel.textContent = item.token + " (No Asset)";
            signPlayer.src = ""; // Clear or placeholder
            await new Promise(r => setTimeout(r, 1500));
        }
    }

    // Reset
    isPlaying = false;
    playerLabel.textContent = "DONE";
    setTimeout(() => {
        playerLabel.classList.add('hidden');
    }, 1000);
}
