# Codex Task Breakdown: Text → SGSL (Gloss) → Sign Output (Video Playlist + 2D Avatar)

This task list covers **text-to-sign language** as a standalone workflow. It assumes the preprocessing pipeline exists (GIF→MediaPipe→PKL) and `sgsl_processed/vocab.json` is available.

Primary output modes:
1) **2D Avatar playback** using preprocessed landmark sequences (preferred for MVP polish)
2) **GIF/Video playlist** fallback (fastest and always works)

---

## 0) Inputs / Outputs

### Inputs
- User typed English text (or STT output)
- Dataset:
  - `sgsl_dataset/{sign_name}/*.gif`
  - `sgsl_dataset/{sign_name}/*.json`
- Preprocessed:
  - `sgsl_processed/landmarks_pkl/{sign_name}.pkl`
  - `sgsl_processed/vocab.json`

### Outputs
- A constrained **gloss token sequence** (only tokens present in vocab)
- A sign rendering plan:
  - `[ { token, sign_name, asset_url, seq_url } ... ]`
- Playback:
  - either 2D avatar animation (landmarks)
  - or GIF playlist (fallback)

---

## 1) Vocabulary & Mapping Layer (Core prerequisite)
### 1.1 `backend/vocab.py`
- [ ] Load `sgsl_processed/vocab.json`
- [ ] Implement canonicalization `canon(str)->TOKEN`
- [ ] Implement:
  - [ ] `token_to_sign(token)->sign_name | None`
  - [ ] `sign_to_token(sign_name)->TOKEN`
- [ ] Implement alias support (optional MVP):
  - [ ] `aliases.json` file for manual mapping: “thanks”→“THANK_YOU”, “pls”→“PLEASE”, etc.
  - [ ] `apply_aliases(tokens)->tokens`

### 1.2 Retrieval of allowed tokens for prompting
- [ ] Implement `get_allowed_tokens(text, max_tokens=200)`
  - [ ] keyword match against vocab tokens
  - [ ] always include “core words” list for general sentences
  - [ ] if match count too low, widen by including top-N most common tokens (or include all)

**Acceptance**
- [ ] For input text, function returns a reasonably sized allowed list (50–200) containing obvious words if present in vocab.

---

## 2) Gemini: English → Constrained Gloss (Main logic)
### 2.1 `backend/gemini_client.py`
- [ ] Implement `text_to_gloss(text, allowed_tokens)->{gloss, unmatched, notes}`
- [ ] Prompt constraints:
  - output strict JSON only
  - gloss tokens must be subset of allowed tokens
  - encourage SGSL-friendly simplification:
    - drop articles (“the”, “a”)
    - simplify tense
    - allow reordering to a “topic-comment” style if possible using available tokens

### 2.2 Server-side validation
- [ ] Implement `validate_gloss(gloss, allowed_tokens, vocab)`
  - [ ] remove out-of-vocab tokens; append to `unmatched`
  - [ ] map synonyms via alias table if possible
  - [ ] if empty gloss: fallback to keyword-extracted tokens

### 2.3 Optional: Generate multiple gloss candidates
- [ ] Implement `text_to_gloss_nbest(text, allowed_tokens, n=3)`
  - [ ] ask Gemini for 2–3 variants
  - [ ] return ranked list

**Acceptance**
- [ ] Given typical input (“I need help”), returns a non-empty gloss using only known tokens.

---

## 3) Plan Construction: Gloss → Render Plan
### 3.1 `backend/planner.py`
- [ ] Implement `build_render_plan(gloss_tokens)->plan`
  - For each token:
    - [ ] resolve `sign_name = token_to_sign(token)`
    - [ ] build URLs:
      - [ ] `seq_url = /sign-seq/{sign_name}`
      - [ ] `gif_url = /sign-gif/{sign_na
