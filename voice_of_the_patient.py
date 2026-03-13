import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import numpy as np
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

sample_rate = 16000
duration = 8

# load whisper once
whisper_model = whisper.load_model("base")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def record_voice():

    print("Speak now...")

    audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1)
    sd.wait()

    audio = np.squeeze(audio)

    write("voice.wav", sample_rate, audio)

    result = whisper_model.transcribe("voice.wav")

    return result["text"]


def ask_gemini(question, image_path=None):

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    prompt = f"""
            You are a professional dermatologist.

            Analyze the skin condition if an image is provided.

            Provide:
            1. Possible skin condition
            2. Possible causes
            3. Basic treatment
            4. When to see a doctor

            User question: {question}
            """

    content = [{"type": "text", "text": prompt}]

    if image_path is not None:

        encoded_image = encode_image(image_path)

        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
            }
        )

    message = HumanMessage(content=content)

    response = llm.invoke([message])

    return response.content
