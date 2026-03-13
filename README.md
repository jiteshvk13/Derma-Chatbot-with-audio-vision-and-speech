# 🩺 DermAI — AI Dermatologist

An AI-powered skin condition analyzer that lets you speak your symptoms, 
upload a skin image, and get a diagnosis read back to you.

## Features
- 🎤 Speak your symptoms — Whisper transcribes your voice
- 👁️ Upload a skin image — Gemini 2.5 Flash analyzes it  
- 🔊 Get diagnosis read aloud — ElevenLabs speaks the response
- 📄 Download diagnosis report as a text file

## Tech Stack
- **Gemini 2.5 Flash** — vision + language model
- **OpenAI Whisper** — speech to text
- **ElevenLabs** — text to speech
- **Streamlit** — web UI
- **Python** — everything glued together

## Setup

### 1. Clone the repo
git clone https://github.com/jiteshvk13/Derma-Chatbot-with-audio-vision-and-speech.git
cd Derma-Chatbot-with-audio-vision-and-speech

### 2. Install dependencies
pip install -r requirements.txt

### 3. Add your API keys
cp .env.example .env
# Edit .env and add your keys

### 4. Run the app
streamlit run dermaai.py

## Environment Variables
GOOGLE_API_KEY=your_google_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

## Project Structure
DermAI/
├── dermaai.py        ← Streamlit app
├── .env.example      ← API keys template
├── requirements.txt  ← dependencies
└── README.md

## Disclaimer
⚠️ This tool is for informational purposes only and does not 
constitute medical advice. Always consult a qualified dermatologist.

## Author
Jitesh — [LinkedIn](https://www.linkedin.com/in/jitesh-p-b45351153/)



