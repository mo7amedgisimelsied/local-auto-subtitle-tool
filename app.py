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
    """Phase 1: Generates the translation for editing."""
    if not video_path:
        return pd.DataFrame(), "Please choose a video file first."

    print("Loading text extraction model...")
    model = WhisperModel("base", device="cpu", compute_type="int8") 
    
    print(f"Reading audio from {video_path}...")
    segments, _ = model.transcribe(video_path, beam_size=5, language="en")
    
    data = []
    print("Translating lines...")
    for i, segment in enumerate(segments, start=1):
        start_time = format_timestamp(segment.start)
        end_time = format_timestamp(segment.end)
        text = segment.text.strip()
        arabic_text = translate_to_arabic(text)
        
        data.append({
            "Line": i,
            "Start Time": start_time,
            "End Time": end_time,
            "Original English": text,
            "Arabic Translation (Click to edit)": arabic_text
        })
        
    df = pd.DataFrame(data)
    return df, "Done! Review the translation below, make any changes you like, then click step 4."

def burn_subtitles(video_path, edited_df):
    """Phase 2: Takes the edited table, makes an SRT, and adds it to the video."""
    if not video_path or edited_df.empty:
        return None, "Please complete step 2 before creating the final video."

    srt_filename = "final_subtitles.srt"
    with open(srt_filename, "w", encoding="utf-8") as srt_file:
        for index, row in edited_df.iterrows():
            srt_file.write(f"{row['Line']}\n")
            srt_file.write(f"{row['Start Time']} --> {row['End Time']}\n")
            srt_file.write(f"{row['Arabic Translation (Click to edit)']}\n\n")

    output_video = "output_arabic_subs_web.mp4"
    style = "FontName=Arial,FontSize=20,PrimaryColour=&H00FFFFFF,BackColour=&H80000000,BorderStyle=4,Outline=0,Shadow=0"
    
    ffmpeg_command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"subtitles={srt_filename}:force_style='{style}'",
        "-c:a", "copy",
        output_video
    ]
    
    try:
        print("Creating final video...")
        subprocess.run(ffmpeg_command, check=True)
        # We return the actual file path. Gradio will package it into a downloadable card.
        return output_video, f"Success! The finished video has been created at: {os.path.abspath(output_video)}"
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None, "Something went wrong while making the video. Please try again."

# ---------------------------------------------------------
# Simplified Web Interface Layout (With Download Card)
# ---------------------------------------------------------
with gr.Blocks(title="Easy Video Subtitler", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>🎬 Automatic Video Translator</h1>
            <p>Upload an English video, get an automatic Arabic translation, make quick fixes, and save your new video with subtitles.</p>
        </div>
        """
    )
    
    with gr.Row():
        # Left Column: Inputs & Status
        with gr.Column(scale=1):
            input_video = gr.Video(label="1. Choose your video")
            btn_process = gr.Button("2. Start Translation", variant="primary")
            status_text = gr.Textbox(label="Current Status", interactive=False)
            
        # Right Column: Data Editor
        with gr.Column(scale=2):
            subtitle_editor = gr.Dataframe(
                label="3. Review and Edit Text", 
                interactive=True,
                wrap=True
            )
            btn_burn = gr.Button("4. Add Subtitles to Video", variant="primary")

            # The compact File component replacing the bulky video player
            output_file = gr.File(label="5. Download Finished Video")

    # Wire up buttons
    btn_process.click(
        fn=transcribe_and_translate,
        inputs=[input_video],
        outputs=[subtitle_editor, status_text]
    )
    
    btn_burn.click(
        fn=burn_subtitles,
        inputs=[input_video, subtitle_editor],
        outputs=[output_file, status_text]
    )

if __name__ == "__main__":
    print("Starting local web server...")
    app.launch(inbrowser=True)