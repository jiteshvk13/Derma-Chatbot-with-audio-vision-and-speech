import gradio as gr
import base64
import os
import numpy as np
from scipy.io.wavfile import write
import whisper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ── Load Whisper once ──────────────────────────────────────────────────────────
whisper_model = whisper.load_model("base")

# ── ElevenLabs (optional) ──────────────────────────────────────────────────────
try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import play

    elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    ELEVENLABS_AVAILABLE = True
except Exception:
    ELEVENLABS_AVAILABLE = False


# ── Core helpers ───────────────────────────────────────────────────────────────


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def transcribe_audio(audio_tuple) -> str:
    """Accepts a (sample_rate, numpy_array) tuple from Gradio's mic component."""
    if audio_tuple is None:
        return ""
    sample_rate, audio_data = audio_tuple
    audio_data = audio_data.astype(np.float32)
    if audio_data.max() > 1.0:
        audio_data /= 32768.0
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    tmp_path = os.path.join(os.environ.get("TEMP", "."), "gradio_voice.wav")
    write(tmp_path, sample_rate, audio_data)
    result = whisper_model.transcribe(tmp_path)
    return result["text"]


def ask_gemini(question: str, image_path: str | None) -> str:
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


def tts_response(text: str) -> str | None:
    """Convert text to speech via ElevenLabs; return tmp audio path or None."""
    if not ELEVENLABS_AVAILABLE or not os.getenv("ELEVENLABS_API_KEY"):
        return None
    audio_generator = elevenlabs_client.text_to_speech.convert(
        text=text,
        voice_id="21m00Tcm4TlvDq8ikWAM",
        model_id="eleven_multilingual_v2",
    )
    tmp_path = os.path.join(os.environ.get("TEMP", "."), "gradio_response.mp3")
    with open(tmp_path, "wb") as f:
        for chunk in audio_generator:
            f.write(chunk)
    return tmp_path


# ── Main pipeline ──────────────────────────────────────────────────────────────


def run_diagnosis(text_question, audio_input, skin_image, voice_output_enabled):
    question = text_question.strip() if text_question else ""
    if not question and audio_input is not None:
        question = transcribe_audio(audio_input)
    if not question:
        return "⚠️ Please type a question or record your voice.", None, ""

    image_path = skin_image
    diagnosis = ask_gemini(question, image_path)

    audio_path = None
    if voice_output_enabled:
        audio_path = tts_response(diagnosis)

    return diagnosis, audio_path, question


# ── CSS theme ─────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --cream: #f5f0e8;
    --warm-white: #faf8f4;
    --sage: #7a9e7e;
    --sage-dark: #527a56;
    --terracotta: #c4694f;
    --charcoal: #2c2c2c;
    --mid-grey: #6b6b6b;
    --light-grey: #e8e3da;
}

body, .gradio-container {
    background: var(--cream) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--charcoal) !important;
}

#header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--light-grey);
    margin-bottom: 2rem;
}

#header h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: clamp(2rem, 5vw, 3.2rem) !important;
    color: var(--charcoal) !important;
    margin: 0 0 0.4rem !important;
}

#header p {
    color: var(--mid-grey) !important;
    font-size: 1rem;
    font-weight: 300;
    margin: 0;
}

.section-label {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--sage-dark) !important;
    margin-bottom: 0.5rem !important;
}

.disclaimer {
    background: #fdf3ee;
    border-left: 3px solid var(--terracotta);
    border-radius: 0 6px 6px 0;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: var(--mid-grey);
    margin-top: 1rem;
}

footer { display: none !important; }
"""

# ── UI layout ─────────────────────────────────────────────────────────────────

with gr.Blocks(css=CSS, title="DermAI — AI Dermatologist") as demo:

    gr.HTML(
        """
    <div id="header">
        <h1>🩺 DermAI</h1>
        <p>AI-powered dermatology assistant &nbsp;·&nbsp;</p>
    </div>
    """
    )

    with gr.Row(equal_height=False):

        with gr.Column(scale=1):
            gr.HTML('<p class="section-label">Skin Image (optional)</p>')
            skin_image = gr.Image(
                type="filepath",
                label="Upload a photo of the affected area",
                show_label=False,
                height=220,
            )

            gr.HTML(
                '<p class="section-label" style="margin-top:1.25rem">Your Question</p>'
            )
            text_question = gr.Textbox(
                placeholder="e.g. I have a red, itchy rash on my forearm…",
                label="",
                lines=3,
            )

            gr.HTML(
                '<p class="section-label" style="margin-top:1rem">Or Record Your Voice</p>'
            )
            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="",
                show_label=False,
            )

            voice_output_enabled = gr.Checkbox(
                label="🔊 Read response aloud (ElevenLabs)",
                value=False,
            )

            with gr.Row():
                submit_btn = gr.Button("Analyze", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")

        with gr.Column(scale=1):
            gr.HTML('<p class="section-label">Transcribed Question</p>')
            transcribed = gr.Textbox(
                label="",
                interactive=False,
                placeholder="Your question will appear here after transcription…",
                lines=2,
            )

            gr.HTML(
                '<p class="section-label" style="margin-top:1.25rem">AI Diagnosis</p>'
            )
            diagnosis_output = gr.Textbox(
                label="",
                interactive=False,
                lines=14,
                placeholder="The AI dermatologist's analysis will appear here…",
            )

            audio_output = gr.Audio(
                label="Voice Response",
                visible=True,
                interactive=False,
            )

            gr.HTML(
                """
            <div class="disclaimer">
                ⚠️ <strong>Medical Disclaimer:</strong> This tool is for informational purposes only
                and does not constitute medical advice. Always consult a qualified dermatologist
                or healthcare professional for diagnosis and treatment.
            </div>
            """
            )

    submit_btn.click(
        fn=run_diagnosis,
        inputs=[text_question, audio_input, skin_image, voice_output_enabled],
        outputs=[diagnosis_output, audio_output, transcribed],
    )

    clear_btn.click(
        fn=lambda: (None, None, "", False, "", None, ""),
        inputs=[],
        outputs=[
            skin_image,
            audio_input,
            text_question,
            voice_output_enabled,
            diagnosis_output,
            audio_output,
            transcribed,
        ],
    )


if __name__ == "__main__":
    demo.launch(share=False)
