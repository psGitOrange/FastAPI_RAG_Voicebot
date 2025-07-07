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

# Create Gradio interface
iface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(sources="microphone", type="numpy"),
    outputs=gr.Textbox(label="Transcribed Text"),
    title="Speech to Text"
)

iface.launch()

# works fine