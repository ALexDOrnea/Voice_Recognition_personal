import whisper
import sounddevice as sd
import numpy as np
import queue
import threading
import torch

# ================================
# CONFIGURATION
# ================================
MODEL_NAME = "tiny.en"   # small, base, etc.
SAMPLE_RATE = 16000      # 16 kHz
BLOCK_DURATION = 5       # seconds per chunk (larger chunks = faster throughput)
DEVICE_INDEX = 8     # set to your input device index (None = default)

# ================================
# INITIALIZE MODEL
# ================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Whisper model '{MODEL_NAME}' on {device}...")
model = whisper.load_model(MODEL_NAME, device=device)

# ================================
# AUDIO QUEUE
# ================================
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """Callback that stores audio chunks from the microphone."""
    if status:
        print("Audio status:", status)
    # Convert float64 → float32 for Whisper
    audio_queue.put(indata.copy().astype(np.float32).flatten())

# ================================
# TRANSCRIPTION THREAD
# ================================
def transcribe_stream():
    print("🔊 Transcribing audio stream...\n")
    while True:
        audio_chunk = audio_queue.get()
        if audio_chunk is None:
            break  # stop signal

        # Transcribe directly from NumPy array
        try:
            result = model.transcribe(
                audio_chunk,
                fp16=(device == "cuda"),
                language="en"
            )
            text = result["text"].strip()
            if text:
                print(f"🗣️  {text}")
        except Exception as e:
            print("Transcription error:", e)

# ================================
# MAIN PROGRAM
# ================================
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    callback=audio_callback,
    blocksize=int(BLOCK_DURATION * SAMPLE_RATE),
    device=DEVICE_INDEX
)

with stream:
    transcribe_thread = threading.Thread(target=transcribe_stream, daemon=True)
    transcribe_thread.start()

    print("🎙️ Listening... Press Ctrl+C to stop.\n")

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\n🛑 Exiting...")
        audio_queue.put(None)
