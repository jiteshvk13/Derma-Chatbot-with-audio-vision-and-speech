import streamlit as st
import base64
import os
import whisper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DermAI", page_icon="🩺")
st.title("🩺 DermAI — AI Dermatologist")


# ── Load Whisper once ──────────────────────────────────────────────────────────
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")


whisper_model = load_whisper()


# ── Load ElevenLabs client once ────────────────────────────────────────────────
@st.cache_resource
def load_elevenlabs():
    return ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))


elevenlabs_client = load_elevenlabs()


# ── Helper: encode image to base64 ────────────────────────────────────────────
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── Helper: transcribe audio using Whisper ────────────────────────────────────
def transcribe_audio(audio_bytes: bytes) -> str:
    tmp_path = os.path.join(os.environ.get("TEMP", "."), "voice_input.wav")
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)
    result = whisper_model.transcribe(tmp_path)
    return result["text"]


# ── Helper: ask Gemini ────────────────────────────────────────────────────────
def ask_gemini(question: str, image_path: str = None) -> str:
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

    if image_path:
        encoded = encode_image(image_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
        )

    response = llm.invoke([HumanMessage(content=content)])
    return response.content


# ── Helper: text to speech via ElevenLabs ────────────────────────────────────
def text_to_speech(text: str) -> bytes:
    audio_generator = elevenlabs_client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",  # George — free plan
        model_id="eleven_multilingual_v2",
    )
    return b"".join(chunk for chunk in audio_generator)


# ── UI: Step 1 — Upload image ─────────────────────────────────────────────────
st.subheader("Step 1: Upload a Skin Image (optional)")
uploaded_image = st.file_uploader(
    "Upload a photo of the affected area", type=["jpg", "jpeg", "png"]
)

image_path = None
if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", width=300)
    image_path = os.path.join(os.environ.get("TEMP", "."), "skin_image.jpg")
    with open(image_path, "wb") as f:
        f.write(uploaded_image.read())


# ── UI: Step 2 — Type OR speak your question ──────────────────────────────────
st.subheader("Step 2: Describe Your Concern")
text_question = st.text_area(
    "Type your question here",
    placeholder="e.g. I have a red, itchy rash on my forearm…",
)

st.write("**Or speak directly into your microphone:**")

# st.audio_input records directly in the browser — no file upload needed
audio_recording = st.audio_input("Click the mic button to record")

transcribed_text = ""
if audio_recording is not None:
    with st.spinner("Transcribing your voice..."):
        transcribed_text = transcribe_audio(audio_recording.read())
    st.success(f"📝 Transcribed: {transcribed_text}")


# ── UI: Step 3 — Options & Analyze ───────────────────────────────────────────
st.subheader("Step 3: Get AI Diagnosis")
voice_output = st.toggle("🔊 Read response aloud (ElevenLabs)", value=False)

if st.button("🔍 Analyze", use_container_width=True):
    # Typed question takes priority, voice transcription is fallback
    question = text_question.strip() or transcribed_text.strip()

    if not question:
        st.warning("Please type a question or record your voice.")
    else:
        with st.spinner("Analyzing with Gemini..."):
            result = ask_gemini(question, image_path)

        st.subheader("🧑‍⚕️ AI Diagnosis")
        st.markdown(result)

        if voice_output:
            with st.spinner("Generating voice response..."):
                audio_bytes = text_to_speech(result)
            st.audio(audio_bytes, format="audio/mp3")

        st.download_button(
            label="📄 Download Report",
            data=result,
            file_name="dermai_report.txt",
            mime="text/plain",
        )


# ── Disclaimer ────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ Medical Disclaimer: This tool is for informational purposes only and does not constitute medical advice. Always consult a qualified dermatologist or healthcare professional."
)
