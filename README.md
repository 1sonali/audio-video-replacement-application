!pip install gradio
!pip install moviepy
!pip install azure-cognitiveservices-speech
!pip install openai
!pip install pydub
import gradio as gr
import os
import tempfile
from moviepy.editor import VideoFileClip, AudioFileClip
import requests
import json
from openai import AzureOpenAI
import numpy as np
from pydub import AudioSegment
import io
import time

# Azure OpenAI Service credentials
AZURE_ENDPOINT = "https://curio-m22u9hu0-swedencentral.openai.azure.com"
AZURE_KEY = "3b7e7e5627b2414a8755a7a42b7f5922"

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_KEY,
    api_version="2024-06-01",
    azure_endpoint=AZURE_ENDPOINT
)

def transcribe_audio(audio_path):
    """Transcribe audio using Azure OpenAI's Whisper model"""
    try:
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper",
                file=audio_file,
                response_format="text"
            )
            # Verify we got a response
            if not response:
                raise Exception("No transcription received from the API")
            return str(response).strip()
    except Exception as e:
        raise Exception(f"Failed to transcribe audio: {str(e)}")

def improve_text_locally(text):
    """Improve the transcribed text using Azure OpenAI"""
    try:
        if not text or not text.strip():
            raise Exception("Input text is empty")

        system_prompt = """You are an expert at improving spoken text.
        Your task is to improve the given transcription while maintaining its original meaning.
        Fix any grammar issues, remove filler words, and make the text more coherent and professional.
        Keep the same tone and key information."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please improve this transcription: {text}"}
        ]

        response = client.chat.completions.create(
            model="gpt-4",  # or whatever model you have deployed
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )

        improved_text = response.choices[0].message.content.strip()

        if not improved_text:
            raise Exception("No improved text generated")

        return improved_text

    except Exception as e:
        print(f"Error in improve_text_locally: {str(e)}")
        # Return original text if improvement fails
        return text if text else "Error improving text"

def generate_speech_with_rest_api(text):
    """Generate speech using Azure TTS REST API with improved error handling"""
    try:
        url = f"{AZURE_ENDPOINT}/openai/deployments/tts/audio/speech?api-version=2024-05-01-preview"
        headers = {
            'api-key': AZURE_KEY,
            'Content-Type': 'application/json'
        }

        body = {
            "model": "tts-1",
            "input": text,
            "voice": "alloy"
        }

        response = requests.post(url, headers=headers, json=body, timeout=30)

        if response.status_code == 200:
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_audio.write(response.content)
            temp_audio.close()
            return temp_audio.name
        else:
            error_msg = f"Speech synthesis failed with status code: {response.status_code}"
            if response.content:
                try:
                    error_details = response.json()
                    error_msg += f"\nDetails: {json.dumps(error_details)}"
                except:
                    error_msg += f"\nResponse: {response.content.decode('utf-8', errors='ignore')}"
            raise Exception(error_msg)

    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error during speech synthesis: {str(e)}")
    except Exception as e:
        raise Exception(f"Error in speech synthesis: {str(e)}")

def extract_audio(video_path):
    """Extract audio from video file"""
    try:
        video = VideoFileClip(video_path)
        if not video.audio:
            raise Exception("No audio track found in video")

        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        video.audio.write_audiofile(temp_audio.name, codec='pcm_s16le')
        video.close()
        return temp_audio.name
    except Exception as e:
        raise Exception(f"Failed to extract audio: {str(e)}")

def replace_audio(video_path, new_audio_path):
    """Replace audio in video with new audio"""
    try:
        video = VideoFileClip(video_path)
        audio = AudioFileClip(new_audio_path)

        # If audio is longer than video, trim it
        if audio.duration > video.duration:
            audio = audio.subclip(0, video.duration)

        # If video is longer than audio, loop the audio
        elif video.duration > audio.duration:
            n_loops = int(np.ceil(video.duration / audio.duration))
            audio = AudioFileClip(new_audio_path)
            concatenated_audio = audio
            for _ in range(n_loops - 1):
                concatenated_audio = concatenated_audio.concatenate_audioclips([audio])
            audio = concatenated_audio.subclip(0, video.duration)

        video = video.set_audio(audio)
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        video.write_videofile(output_path, codec='libx264', audio_codec='aac')

        video.close()
        audio.close()

        return output_path
    except Exception as e:
        raise Exception(f"Failed to replace audio: {str(e)}")

def process_video(video_path, progress=gr.Progress()):
    """Main processing function with improved error handling"""
    temp_files = []

    try:
        # Step 1: Extract audio
        progress(0, desc="Extracting audio...")
        audio_path = extract_audio(video_path)
        temp_files.append(audio_path)

        # Step 2: Transcribe audio
        progress(0.25, desc="Transcribing audio...")
        transcription = transcribe_audio(audio_path)
        if not transcription:
            raise Exception("Failed to get transcription")

        # Step 3: Improve text
        progress(0.5, desc="Improving text...")
        improved_text = improve_text_locally(transcription)
        if not improved_text:
            raise Exception("Failed to improve text")

        # Step 4: Generate new speech
        progress(0.75, desc="Generating new speech...")
        new_audio_path = generate_speech_with_rest_api(improved_text)
        temp_files.append(new_audio_path)

        # Step 5: Replace audio
        progress(0.9, desc="Replacing audio...")
        output_path = replace_audio(video_path, new_audio_path)

        return output_path, transcription, improved_text

    except Exception as e:
        error_msg = f"Error in video processing: {str(e)}"
        print(error_msg)
        return None, error_msg, error_msg

    finally:
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Error cleaning up temporary file {temp_file}: {str(e)}")

# Create Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Video Audio Replacement App")

    with gr.Row():
        input_video = gr.Video(label="Upload Video")
        output_video = gr.Video(label="Processed Video")

    with gr.Row():
        original_text = gr.Textbox(label="Original Transcription", lines=4)
        improved_text = gr.Textbox(label="Improved Text", lines=4)

    process_btn = gr.Button("Process Video")
    process_btn.click(
        fn=process_video,
        inputs=[input_video],
        outputs=[output_video, original_text, improved_text]
    )

# Launch the app
app.launch(share=True)
