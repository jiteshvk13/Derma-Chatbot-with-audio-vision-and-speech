import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from voice_of_the_patient import record_voice

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))


def run_ai_doctor(
    question=None, image_path=None, input_mode="text", output_mode="text"
):

    # Step 1: Get question
    if input_mode == "audio":
        question = record_voice()

    if question is None:
        raise ValueError("Question must be provided if input_mode='text'")

    # Step 2: Build prompt
    user_prompt = f"""
You are a professional dermatologist.

Analyze the provided skin image carefully and answer the user's question.

Provide:
1. Possible skin condition
2. Possible causes
3. Basic treatment
4. When to see a doctor

User question: {question}
"""

    content = [{"type": "text", "text": user_prompt}]

    # Step 3: Add image if provided
    if image_path is not None:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
            }
        )

    message = HumanMessage(content=content)

    # Step 4: Ask Gemini
    response = llm.invoke([message])
    ai_answer = response.content

    print(f"\nAI Dermatologist Response:\n{ai_answer}")

    # Step 5: Convert to speech if needed
    if output_mode == "audio":
        audio = elevenlabs.text_to_speech.convert(
            text=ai_answer,
            voice_id="21m00Tcm4TlvDq8ikWAM",
            model_id="eleven_multilingual_v2",
        )

        play.play_audio(audio)

    return ai_answer
