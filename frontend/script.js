import { AvatarController } from './avatar.js';

const API_URL = "http://127.0.0.1:8000/api/translate";
const LANDMARKS_URL = "http://127.0.0.1:8000/api/sign";

const inputText = document.getElementById('inputText');
const translateBtn = document.getElementById('translateBtn');
const outputSection = document.getElementById('outputSection');
const glossDisplay = document.getElementById('glossDisplay');
const statusNotes = document.getElementById('statusNotes');
const replayBtn = document.getElementById('replayBtn');
const signPlayer = document.getElementById('signPlayer');
const placeholder = document.getElementById('placeholder');
const playerLabel = document.getElementById('playerLabel');

// Initialize 3D Avatar
const avatar = new AvatarController('avatarContainer');

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

    // Resize avatar canvas now that container is visible
    avatar.resize();

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

    // Start Playback with 3D Avatar
    playSequence(currentPlan);
}

async function playSequence(plan) {
    if (isPlaying) return;
    isPlaying = true;

    // Debug: log the plan
    console.log('Plan received:', plan.map(p => p.sign_name));

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

    console.log('Unique plan:', uniquePlan.map(p => p.sign_name));

    for (let i = 0; i < uniquePlan.length; i++) {
        const item = uniquePlan[i];

        if (item.type === 'sign' && item.sign_name) {
            console.log(`Starting sign ${i + 1}/${uniquePlan.length}: ${item.sign_name}`);

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
                    console.log(`Playing skeleton: ${item.token} (${data.frames.length} frames)`);

                    // Start skeleton animation
                    const skeletonPromise = avatar.playSequence(data.frames, 30);

                    // Hide GIF after ~3 seconds (typical GIF duration) to prevent visual looping
                    const gifHidePromise = new Promise(resolve => {
                        setTimeout(() => {
                            signPlayer.classList.add('hidden');
                            signPlayer.src = '';
                            resolve();
                        }, 3000);
                    });

                    // Wait for skeleton to finish (GIF will hide after 3s)
                    await skeletonPromise;
                    console.log(`Finished skeleton: ${item.token}`);
                } else {
                    console.warn(`No 3D data for ${item.sign_name}`);
                    await new Promise(r => setTimeout(r, 2000));
                    signPlayer.classList.add('hidden');
                    signPlayer.src = '';
                }
            } catch (e) {
                console.error("Fetch error", e);
            }

            // Brief pause between words
            if (i < uniquePlan.length - 1) {
                await new Promise(r => setTimeout(r, 300));
            }

        } else {
            console.log(`Skipping non-sign: ${item.token}`);
            await new Promise(r => setTimeout(r, 500));
        }
    }

    // Final cleanup
    placeholder.classList.remove('hidden');
    playerLabel.classList.add('hidden');
    isPlaying = false;
    console.log('Sequence complete');
}

