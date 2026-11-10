import base64
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent

# Load .env from the project folder, whatever directory the server is started from
load_dotenv(BASE_DIR / ".env")

# Models can be overridden in .env. The defaults are the ones this project's key can use.
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5-nano")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")

# Audio containers the transcription API accepts. The browser records webm (Chrome/Firefox)
# or mp4 (Safari); the file extension must match the real format or the API rejects it.
ALLOWED_AUDIO_SUFFIXES = {".webm", ".ogg", ".mp4", ".m4a", ".mp3", ".mpeg", ".mpga", ".wav", ".flac"}
DEFAULT_AUDIO_SUFFIX = ".webm"
MAX_HISTORY_MESSAGES = 20


def load_api_key() -> str:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise SystemExit(f"OPENAI_API_KEY is missing. Put it in {BASE_DIR / '.env'}")
    if not key.startswith("sk-"):
        raise SystemExit(
            "OPENAI_API_KEY does not start with 'sk-' (check for a capital S or stray characters in .env)"
        )
    return key


app = FastAPI()

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=load_api_key())

# Simple in-memory chat history
chat_history: List[Dict[str, str]] = []


def audio_suffix(filename: Optional[str]) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in ALLOWED_AUDIO_SUFFIXES else DEFAULT_AUDIO_SUFFIX


def transcribe_audio(audio_path: str) -> str:
    """Convert speech to text."""
    with open(audio_path, "rb") as audio:
        transcript = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=audio,
            response_format="text",
        )
    return transcript.strip()


def process_with_llm(text: str) -> str:
    """Answer the user, using the conversation history."""
    global chat_history
    messages = [
        {"role": "system", "content": "You are a helpful voice assistant. Keep responses concise and natural."}
    ] + chat_history + [{"role": "user", "content": text}]

    # gpt-5 models reason before answering; keep that minimal so the voice reply is quick
    extra_body = {"reasoning_effort": "minimal"} if CHAT_MODEL.startswith("gpt-5") else None
    chat_completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        extra_body=extra_body,
    )
    response = (chat_completion.choices[0].message.content or "").strip()

    chat_history = (chat_history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": response},
    ])[-MAX_HISTORY_MESSAGES:]
    return response


def convert_to_speech(text: str) -> bytes:
    """Convert text to speech (mp3)."""
    speech_response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
    )
    return speech_response.content


def error_response(step: str, error: Exception) -> JSONResponse:
    print(f"{step} error: {error}")
    return JSONResponse(content={"error": f"{step} failed: {error}"}, status_code=500)


@app.post("/process-voice")
async def process_voice(audio_file: UploadFile = File(...)):
    content = await audio_file.read()
    if not content:
        return JSONResponse(content={"error": "No audio received"}, status_code=400)

    with tempfile.NamedTemporaryFile(delete=False, suffix=audio_suffix(audio_file.filename)) as temp_audio:
        temp_audio.write(content)
        temp_audio_path = temp_audio.name

    try:
        try:
            transcript = transcribe_audio(temp_audio_path)
        except Exception as e:
            return error_response("Transcription", e)
        if not transcript:
            return JSONResponse(content={"error": "Didn't catch any speech, try again"}, status_code=422)

        try:
            response_text = process_with_llm(transcript)
        except Exception as e:
            return error_response("LLM processing", e)

        try:
            audio_content = convert_to_speech(response_text)
        except Exception as e:
            return error_response("Text-to-speech", e)
    finally:
        Path(temp_audio_path).unlink(missing_ok=True)

    return JSONResponse(content={
        "transcript": transcript,
        "response_text": response_text,
        "audio": base64.b64encode(audio_content).decode("utf-8"),
        "history": chat_history,
    })


# Serve the page from the same server, so the mic works at http://localhost:8000
@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/frames.json")
async def frames():
    return FileResponse(BASE_DIR / "frames.json")


if __name__ == "__main__":
    print("Open http://localhost:8000 in Chrome, Firefox or Safari")
    uvicorn.run(app, host="127.0.0.1", port=8000)
