from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import queue
import threading
import torch
import time

print(torch.cuda.is_available())
print(torch.version.cuda)
# ================================
# CONFIGURATION
# ================================
MODEL_NAME = "tiny.en"
SAMPLE_RATE = 16000
CHUNK_DURATION = 1       # seconds of audio collected per callback
BUFFER_DURATION = 5      # seconds of rolling buffer for context
DEVICE_INDEX = None      # use default input

# ================================
# MODEL INITIALIZATION
# ================================
device = "cpu" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"Loading Faster-Whisper model '{MODEL_NAME}' on {device} ({compute_type})...")
model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)

# ================================
# GLOBALS
# ================================
audio_queue = queue.Queue()
rolling_buffer = np.zeros(int(BUFFER_DURATION * SAMPLE_RATE), dtype=np.float32)
buffer_lock = threading.Lock()
last_text = ""

# ================================
# AUDIO CALLBACK
# ================================
def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    audio_queue.put(indata.copy().astype(np.float32).flatten())

# ================================
# TRANSCRIPTION THREAD
# ================================
def transcribe_stream():
    global rolling_buffer, last_text
    print("🔊 Real-time transcription started...\n")

    while True:
        audio_chunk = audio_queue.get()
        if audio_chunk is None:
            break

        with buffer_lock:
            # Roll the buffer and append new chunk
            rolling_buffer = np.roll(rolling_buffer, -len(audio_chunk))
            rolling_buffer[-len(audio_chunk):] = audio_chunk

        # Transcribe the current rolling buffer
        try:
            segments, _ = model.transcribe(
                rolling_buffer,
                beam_size=1,
                language="en",
                vad_filter=False
            )

            # Combine text
            text = "".join([seg.text for seg in segments]).strip()

            # Print only new additions
            if text and text != last_text:
                new_text = text[len(last_text):].strip()
                if new_text:
                    print(f"{new_text}", flush=True)
                last_text = text

        except Exception as e:
            print("Transcription error:", e)

# ================================
# MAIN
# ================================
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    callback=audio_callback,
    blocksize=int(CHUNK_DURATION * SAMPLE_RATE),
    device=DEVICE_INDEX
)

with stream:
    transcribe_thread = threading.Thread(target=transcribe_stream, daemon=True)
    transcribe_thread.start()

    print(" Listening in real time... Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting...")
        audio_queue.put(None)
