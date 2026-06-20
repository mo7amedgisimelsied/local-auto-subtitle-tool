import os
import subprocess
import requests
import pandas as pd
import gradio as gr
from faster_whisper import WhisperModel

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def format_timestamp(seconds):
    """Converts seconds into the SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def translate_to_arabic(text):
    """Sends a single English sentence to the local Ollama TranslateGemma model."""
    prompt = (
        "Translate the following English text into professional Arabic. "
        "Provide ONLY the direct translation. Do not include any notes, explanations, "
        "or introductory text.\n"
        f"English: {text}\n"
        "Arabic:"
    )
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "translategemma",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1}
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get('response', '').strip()
    except Exception as e:
        print(f"Error translating text '{text}': {e}")
        return ""

# ---------------------------------------------------------
# Pipeline Functions
# ---------------------------------------------------------
def transcribe_and_translate(video_path):
    """Phase 1: Generates the transcription and translation for editing."""
    if not video_path:
        return pd.DataFrame(), "Please upload a video first."

    print("Loading faster-whisper model...")
    model = WhisperModel("base", device="cpu", compute_type="int8") 
    
    print(f"Transcribing {video_path}...")
    segments, _ = model.transcribe(video_path, beam_size=5, language="en")
    
    data = []
    print("Translating segments...")
    for i, segment in enumerate(segments, start=1):
        start_time = format_timestamp(segment.start)
        end_time = format_timestamp(segment.end)
        text = segment.text.strip()
        arabic_text = translate_to_arabic(text)
        
        # We store this in a dictionary to easily convert to a Pandas DataFrame
        data.append({
            "ID": i,
            "Start": start_time,
            "End": end_time,
            "English (Original)": text,
            "Arabic (Editable)": arabic_text
        })
        print(f"Processed line {i}")
        
    df = pd.DataFrame(data)
    return df, "Transcription and Translation complete. You can now edit the Arabic text below."

def burn_subtitles(video_path, edited_df):
    """Phase 2: Takes the edited table, makes an SRT, and burns it to the video."""
    if not video_path or edited_df.empty:
        return None, "Missing video or subtitle data."

    # 1. Generate the SRT file using a safe relative path
    srt_filename = "final_subtitles.srt"
    with open(srt_filename, "w", encoding="utf-8") as srt_file:
        for index, row in edited_df.iterrows():
            srt_file.write(f"{row['ID']}\n")
            srt_file.write(f"{row['Start']} --> {row['End']}\n")
            srt_file.write(f"{row['Arabic (Editable)']}\n\n")

    # 2. Burn subtitles using FFmpeg
    output_video = "output_arabic_subs_web.mp4"
    style = "FontName=Arial,FontSize=10,PrimaryColour=&H00FFFFFF,BackColour=&000000000,BorderStyle=4,Outline=0,Shadow=0"
    
    # Using the relative filename directly prevents the Windows drive letter colon (:) clash
    ffmpeg_command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"subtitles={srt_filename}:force_style='{style}'",
        "-c:a", "copy",
        output_video
    ]
    
    try:
        print("Burning subtitles...")
        subprocess.run(ffmpeg_command, check=True)
        return output_video, "Success! Subtitles burned."
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        return None, f"FFmpeg error occurred. Check terminal logs."

# ---------------------------------------------------------
# Gradio Web Interface Layout
# ---------------------------------------------------------
with gr.Blocks(title="Local AI Video Translator", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎬 Local AI Video Translator")
    gr.Markdown("Upload a video, generate Arabic translations via Ollama, edit them, and burn them into a new file.")
    
    with gr.Row():
        # Left Column: Inputs & Status
        with gr.Column(scale=1):
            input_video = gr.Video(label="1. Upload Source Video")
            btn_process = gr.Button("2. Transcribe & Translate", variant="primary")
            status_text = gr.Textbox(label="Status Logging", interactive=False)
            
        # Right Column: Data Editor & Final Output
        with gr.Column(scale=2):
            # This is the interactive table where you can click and edit the Arabic text
            subtitle_editor = gr.Dataframe(
                label="3. Review & Edit Subtitles", 
                interactive=True,
                wrap=True
            )
            btn_burn = gr.Button("4. Burn Subtitles to Video", variant="primary")
            output_video = gr.Video(label="5. Final Output Video")

    # Wire up the buttons to the Python functions
    btn_process.click(
        fn=transcribe_and_translate,
        inputs=[input_video],
        outputs=[subtitle_editor, status_text]
    )
    
    btn_burn.click(
        fn=burn_subtitles,
        inputs=[input_video, subtitle_editor],
        outputs=[output_video, status_text]
    )

if __name__ == "__main__":
    print("Starting local web server...")
    app.launch(inbrowser=True)