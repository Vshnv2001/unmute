const API_URL = "http://127.0.0.1:8000/api/translate";

const inputText = document.getElementById('inputText');
const translateBtn = document.getElementById('translateBtn');
const outputSection = document.getElementById('outputSection');
const glossDisplay = document.getElementById('glossDisplay');
const signPlayer = document.getElementById('signPlayer');
const placeholder = document.getElementById('placeholder');
const statusNotes = document.getElementById('statusNotes');
const playerLabel = document.getElementById('playerLabel');
const replayBtn = document.getElementById('replayBtn');

let currentPlan = [];
let isPlaying = false;

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
