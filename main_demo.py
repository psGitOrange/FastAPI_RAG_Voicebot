import os
import asyncio
import tempfile

import edge_tts

from rag.indexing import create_chroma_index
# from rag.querying import create_query_engine
from rag.chat_engine import create_chat_engine

import gradio as gr
import speech_recognition as sr
import io
import numpy as np
from scipy.io import wavfile


def transcribe_audio(audio):
    if audio is None:
        return "No audio received"
    
    # Gradio returns audio as (sample_rate, audio_data)
    sample_rate, audio_data = audio
    
    # Convert to wav format in memory
    audio_bytes = io.BytesIO()
    wavfile.write(audio_bytes, sample_rate, audio_data.astype(np.int16))
    audio_bytes.seek(0)
    
    # Use speech recognition
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_bytes) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Error: {e}"
    

# title="Edge TTS API",
# description="A simple API for text-to-speech conversion using Microsoft Edge TTS",

# Available voices
VOICE_OPTIONS = ["en-US-AvaNeural", "hi-IN-SwaraNeural", "en-US-AriaNeural", "en-IN-NeerjaNeural", "en-AU-NatashaNeural"]

# Hard-coded rate and pitch settings for natural sound
DEFAULT_RATE = "+0%"  # Neutral rate
DEFAULT_PITCH = "+0Hz"  # Neutral pitch


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

    voice_name = voice  # VOICES.get(voice)
    # if not voice_name:
    #     raise ValueError(f"Invalid voice selection: {voice}")

    communicate = edge_tts.Communicate(text, voice_name, rate=DEFAULT_RATE, pitch=DEFAULT_PITCH)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_path = tmp_file.name
        await communicate.save(tmp_path)

    return tmp_path

index = create_chroma_index()
# query_engine = create_query_engine(index)
chat_engine = create_chat_engine(index)


# def voice_assistant(audio, voice_select):
#     sr_text = transcribe_audio(audio)
#     print(sr_text)
#     response = chat_engine.chat(sr_text)
#     print(response)
#     audio_file_path = asyncio.run(text_to_speech(response.response, voice_select))
#     print(audio_file_path)
#     return audio_file_path

# # Create Gradio interface
# demo = gr.Interface(
#     fn=voice_assistant,
#     inputs=[gr.Audio(sources="microphone", type="numpy"),
#             gr.Dropdown(choices=VOICE_OPTIONS, label="Select Voice")],
#     outputs=[gr.Audio(type="filepath", label="Synthesized Audio"),],
#     title="Speech to Speech"
# )

# demo.launch()

def voice_assistant(audio, voice_select):
    sr_text = transcribe_audio(audio)
    yield sr_text, "Generating response...", None

    response = chat_engine.chat(sr_text).response
    yield sr_text, response, None

    audio_file_path = asyncio.run(text_to_speech(response, voice_select))
    yield sr_text, response, audio_file_path


# Create Gradio interface with streaming updates
with gr.Blocks(title="Real-time Speech to Speech Assistant") as demo:
    
    with gr.Row():
        with gr.Column(scale=2):
            audio_input = gr.Audio(sources="microphone", type="numpy", label="Voice Input")
            voice_select = gr.Dropdown(choices=VOICE_OPTIONS, label="Select Voice", value=VOICE_OPTIONS[0] if VOICE_OPTIONS else None)
            submit_btn = gr.Button("Query!", variant="primary")
        
        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("Step 1: Speech Recognition")
                transcribed_text = gr.Textbox(label="Transcribed Text", lines=2, placeholder="Your speech will appear here...")
            
            with gr.Group():
                gr.Markdown("Step 2: RAG Response")
                rag_response = gr.Textbox(label="AI Response", lines=4, placeholder="AI response will appear here...")
            
            with gr.Group():
                gr.Markdown("Step 3: Text-to-Speech")
                audio_output = gr.Audio(label="Synthesized Audio", type="filepath")
    
    # Connect the streaming function
    submit_btn.click(            
        fn=voice_assistant,
        inputs=[audio_input, voice_select],
        outputs=[transcribed_text, rag_response, audio_output]
    )

demo.launch()