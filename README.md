# Unmute — Singapore Sign Language Translator

A web application that translates text and voice input into Singapore Sign Language (SgSL) with real-time 3D avatar animations. Built with FastAPI, Three.js, and Google Gemini AI.

## Overview

Unmute is an interactive sign language translator that bridges communication gaps by converting spoken/written language into animated Singapore Sign Language signs. The application supports multiple input methods (text and voice) and multiple languages including English, Chinese, Malay, and Tamil.

### Key Features

- **Text-to-Sign Translation**: Input text in multiple languages and get SgSL gloss tokens
- **Voice-to-Sign Translation**: Record audio and automatically transcribe and translate to sign language
- **Avatar Animation with MediaPipe**: Visualize signs using hand and body landmark animations
- **Multi-language Support**: Supports English, Chinese (Simplified/Traditional), Malay, Tamil, and other languages
- **Real-time Processing**: Fast translation and rendering pipeline using Google Gemini AI
- **Interactive UI**: Modern, warm beige-themed interface with smooth animations

## Dataset

This project uses sign language data from the **Singapore Sign Language Sign Bank** maintained by Nanyang Technological University (NTU):

**Source**: [https://blogs.ntu.edu.sg/sgslsignbank/signs/](https://blogs.ntu.edu.sg/sgslsignbank/signs/)

The dataset contains over 1,100 words and 1,307 signs from the Singapore Sign Language corpus, providing a comprehensive vocabulary for translation.

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Google Gemini AI** - Text-to-gloss translation and audio transcription
- **MediaPipe** - Hand and pose landmark extraction

### Frontend
- **Three.js** - 3D avatar rendering
- **Tailwind CSS** - Styling
- **Web Audio API** - Voice recording

### Processing
- **OpenCV** - Video/image processing
- **Pydub** - Audio processing

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- A Google Gemini API key 

### Installation

1. **Clone the repository**
   ```bash
   git clone github.com/Vshnv2001/unmute
   cd singapore-sign-language
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install backend dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   cd ..
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the `backend/` directory:
   ```bash
   cd backend
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   cd ..
   ```
   
   Replace `your_api_key_here` with your actual Google Gemini API key.

5. **Prepare the dataset**
   
   The application expects processed sign language data in the following structure:
   ```
   sgsl_dataset/
     └── {sign_name}/
         └── {sign_name}.gif
   
   sgsl_processed/
     ├── vocab.json
     └── landmarks_pkl/
         └── {sign_name}.pkl
   ```
   
   Use the preprocessing scripts in the `scripts/` directory to process raw sign data:
   ```bash
   python scripts/preprocess_gifs_to_pkl.py
   python scripts/build_vocab_from_json.py
   ```

### Running the Application

1. **Start the backend server**
   ```bash
   # From the project root
   cd backend
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Access the application**
   
   Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

   The FastAPI server serves both the API endpoints and the frontend static files.

## Usage

### Text Translation Mode

1. Click on the "Text" tab in the interface
2. Type or paste your text in the input field
3. Click "Translate" or press Enter
4. View the translated gloss tokens and 3D sign animations

### Voice Translation Mode

1. Click on the "Voice" tab
2. Click the microphone button to start recording
3. Speak your message
4. Click stop when finished
5. The audio will be automatically transcribed and translated to sign language

### API Endpoints

The backend exposes the following REST API endpoints:

- `GET /health` - Health check and vocabulary size
- `POST /api/translate` - Translate text to SgSL gloss tokens
  ```json
  {
    "text": "hello world",
    "language": "en"  // optional, auto-detects if omitted
  }
  ```
- `POST /api/transcribe` - Transcribe audio and optionally translate
  ```json
  {
    "audio_data": "base64_encoded_audio",
    "mime_type": "audio/webm",
    "language": "en",  // optional
    "auto_translate": true  // optional
  }
  ```
- `GET /api/sign/{sign_name}/landmarks` - Get 3D landmark data for a sign

## Project Structure

```
singapore-sign-language/
├── backend/
│   ├── app.py                 # FastAPI application and routes
│   ├── gemini_client.py       # Google Gemini API client
│   ├── vocab.py               # Vocabulary management
│   ├── planner.py             # Render plan builder
│   ├── sign_seq.py            # Sign sequence manager
│   ├── hand_embedder.py       # Hand landmark extraction
│   ├── aliases.json           # Sign name aliases
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── index.html             # Main HTML interface
│   ├── script.js              # Frontend JavaScript logic
│   └── avatar.js              # 3D avatar rendering
├── scripts/
│   ├── preprocess_gifs_to_pkl.py    # Dataset preprocessing
│   ├── build_vocab_from_json.py     # Vocabulary builder
│   ├── test_gemini.py               # Gemini API testing
│   └── ...                          # Other utility scripts
├── utils/
│   ├── generate_pose_data.py        # Pose data generation
│   ├── softdtw_nn_pipeline.py       # Soft-DTW neural network
│   └── webcam_demo.py               # Webcam demo utility
├── sgsl_dataset/              # Raw sign GIFs (not in repo)
├── sgsl_processed/            # Processed landmarks and vocab (not in repo)
└── README.md                  # This file
```

## Development

### Testing

Run individual test scripts:
```bash
python scripts/test_gemini.py
python scripts/verify_vocab.py
python scripts/test_planner.py
```

### Preprocessing Pipeline

To process new sign language data:

1. **Extract landmarks from GIFs**
   ```bash
   python scripts/preprocess_gifs_to_pkl.py
   ```

2. **Build vocabulary**
   ```bash
   python scripts/build_vocab_from_json.py
   ```

3. **Visualize and verify**
   ```bash
   python scripts/test_preprocess_visualize.py
   ```
