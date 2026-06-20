from faster_whisper import WhisperModel

# Use "cuda" if you configured Step 3; otherwise use "cpu"
device = "cpu" 
compute_type = "int8" # Use "int8" if running on CPU

print("Loading model...")
model = WhisperModel("base", device=device, compute_type=compute_type)

# Replace with the path to your local audio file
audio_path = "output.mp3"

print("Transcribing...")
segments, info = model.transcribe(audio_path, beam_size=5)

print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
