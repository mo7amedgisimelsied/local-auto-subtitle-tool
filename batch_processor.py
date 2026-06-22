import os
import subprocess
import requests
from faster_whisper import WhisperModel

def format_timestamp(seconds):
    """Converts seconds into the SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def translate_to_arabic(text):
    """Sends English text to the local Ollama TranslateGemma model."""
    # TranslateGemma relies heavily on a highly specific prompt structure to avoid rambling.
    prompt = (
        "You are a professional English (en) to Arabic (ar) translator. Your goal is to accurately convey "
        "the meaning and nuances of the original English text while adhering to Arabic grammar, vocabulary, "
        "and cultural sensitivities. Produce only the Arabic translation, without any additional explanations "
        "or commentary. Please translate the following English text into Arabic: " + text
    )
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "translategemma",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        # Extract and strip the translation to ensure no rogue whitespaces
        return response.json().get('response', '').strip()
    except Exception as e:
        print(f"Error translating text '{text}': {e}")
        return ""

def process_video(input_video, output_video):
    # 1. Initialize Faster-Whisper
    print("Loading faster-whisper model...")
    # 'base' model is fast, but you can change to 'small' or 'medium' for better accuracy
    model = WhisperModel("base", device="cpu", compute_type="int8") 
    
    # 2. Extract and transcribe audio
    print(f"Transcribing {input_video}...")
    segments, _ = model.transcribe(input_video, beam_size=5, language="en")
    
    # 3. Translate and build the .srt file
    srt_filename = "translated_subtitles.srt"
    print("Translating segments to Arabic via Ollama...")
    
    with open(srt_filename, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(segments, start=1):
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            text = segment.text.strip()
            
            print(f"Original: {text}")
            arabic_text = translate_to_arabic(text)
            print(f"Arabic:   {arabic_text}\n")
            
            # Write SRT format blocks
            srt_file.write(f"{i}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{arabic_text}\n\n")

    # 4. Burn subtitles using FFmpeg
    print("Burning Arabic subtitles into the video...")
    
    # FFmpeg style breakdown:
    # BorderStyle=4: Creates an opaque background box behind the text.
    # PrimaryColour=&H00FFFFFF: Standard white text.
    # BackColour=&H80000000: Black background box with 50% transparency (for aesthetics). Change to &H00000000 for solid black.
    # FontName=Arial: Ensures rendering engine recognizes Arabic script.
    style = "FontName=Arial,FontSize=10,PrimaryColour=&H00FFFFFF,BackColour=&H00000000,BorderStyle=4,Outline=0,Shadow=0"
    
    # Use relative paths for subtitles to avoid absolute path escaping issues in FFmpeg
    ffmpeg_command = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", f"subtitles={srt_filename}:force_style='{style}'",
        "-c:a", "copy",  # Pass original audio through without quality loss
        output_video
    ]
    
    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"\nSuccess! Output saved to: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"\nFFmpeg encountered an error: {e}")

if __name__ == "__main__":
    # Set your filenames here
    INPUT_VIDEO_PATH = "test2.mp4" 
    OUTPUT_VIDEO_PATH = "myVideo.mp4"
    
    if os.path.exists(INPUT_VIDEO_PATH):
        process_video(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)
    else:
        print(f"Input video '{INPUT_VIDEO_PATH}' not found in the current directory.")