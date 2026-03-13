from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import os
import io
import whisper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="DermAI API")

# Allow Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once at startup ───────────────────────────────────────────────
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "ok", "message": "DermAI API is running"}


# ── Endpoint 1: Transcribe audio → text ──────────────────────────────────────
@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Accepts a WAV/MP3 file, returns transcribed text."""
    audio_bytes = await audio.read()
    tmp_path = "/tmp/input_audio.wav"
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)
    result = whisper_model.transcribe(tmp_path)
    return {"text": result["text"]}


# ── Endpoint 2: Analyze skin condition ───────────────────────────────────────
@app.post("/diagnose")
async def diagnose(
    question: str = Form(...),
    image: UploadFile = File(None),  # image is optional
):
    """Accepts a question + optional image, returns AI diagnosis."""

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    prompt = f"""You are a professional dermatologist.

Analyze the skin condition if an image is provided.

Provide a clear, structured response with:
1. Possible skin condition
2. Possible causes
3. Basic treatment recommendations
4. When to see a doctor

User question: {question}"""

    content = [{"type": "text", "text": prompt}]

    # Add image if provided
    if image is not None:
        image_bytes = await image.read()
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
        )

    response = llm.invoke([HumanMessage(content=content)])
    return {"diagnosis": response.content}


# ── Endpoint 3: Text to speech ────────────────────────────────────────────────
@app.post("/speak")
async def speak(text: str = Form(...)):
    """Accepts text, returns MP3 audio stream via ElevenLabs."""
    audio_generator = elevenlabs_client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",  # George — free plan
        model_id="eleven_multilingual_v2",
    )
    audio_bytes = b"".join(chunk for chunk in audio_generator)
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")
