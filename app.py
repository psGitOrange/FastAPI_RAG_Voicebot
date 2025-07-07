import os
import asyncio
import tempfile
from typing import Literal, Optional

import edge_tts
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag.indexing import create_chroma_index
from rag.querying import create_query_engine
from rag.chat_engine import create_chat_engine

app = FastAPI()

@app.get("/")
async def root():
    """
    API root endpoint with basic information
    """
    return {
        "message": "Welcome to the RAG Query API!, Edge TTS API is running",
        "available_voices": {
            "en": "English (US) - Female (Ava)",
            "hi": "Hindi - Female (Swara)"
        },
        "endpoints": {
            "/query": "POST - Retrieve query response", 
            "/tts": "POST - Generate speech from text"
        }
    }

index = create_chroma_index()
query_engine = create_query_engine(index)
chat_engine = create_chat_engine(index)

response = chat_engine.chat("Hello Meera! Are you online?")
print(f"✅ Chat engine test response: {response.response}")

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def ask_question(req: QueryRequest):
    # response = query_engine.query(req.query)
    response = chat_engine.chat(req.query)
    return {"response": response.response}


# Hard-coded voice options
VOICES = {
    "en": "en-US-AvaNeural",  # English (US) - Female
    "hi": "hi-IN-SwaraNeural"  # Hindi - Female
}

# Hard-coded rate and pitch settings for natural sound
DEFAULT_RATE = "+0%"  # Neutral rate
DEFAULT_PITCH = "+0Hz"  # Neutral pitch


class TTSRequest(BaseModel):
    text: str
    voice: Literal["en", "hi"] = "en"


async def text_to_speech(text: str, voice: str) -> str:
    """
    Convert text to speech using Edge TTS

    Args:
        text: The text to convert to speech
        voice: The voice identifier to use

    Returns:
        Path to the generated audio file
    """
    if not text.strip():
        raise ValueError("Text cannot be empty")

    voice_name = VOICES.get(voice)
    if not voice_name:
        raise ValueError(f"Invalid voice selection: {voice}")

    communicate = edge_tts.Communicate(text, voice_name, rate=DEFAULT_RATE, pitch=DEFAULT_PITCH)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_path = tmp_file.name
        await communicate.save(tmp_path)

    return tmp_path

@app.post("/tts")
async def generate_speech(request: TTSRequest):
    """
    Generate speech from text and return the audio file
    """
    try:
        audio_path = await text_to_speech(request.text, request.voice)
        return FileResponse(
            audio_path,
            media_type="audio/mpeg",
            filename="speech.mp3",
            background=None
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app)
