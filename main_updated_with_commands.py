import warnings
warnings.filterwarnings("ignore")

import sounddevice as sd
import numpy as np
import queue
import threading
import time
from faster_whisper import WhisperModel
from collections import deque

# ================================
# CONFIG
# ================================
WAKE_WORD = "garmin"
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5
PAUSE_THRESHOLD = 1.0
MIN_COMMAND_DURATION = 1.0
DEVICE_INDEX = None
WAKE_WORD_DELAY = 1.5

# ================================
# BEEP
# ================================
def play_beep():
    duration = 0.2
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    tone = 0.3 * np.sin(2 * np.pi * 1000 * t)
    sd.play(tone, samplerate=SAMPLE_RATE)
    sd.wait()

# ================================
# MODEL
# ================================
print("Loading Faster-Whisper (tiny.en)...")
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

# ================================
# GLOBALS
# ================================
audio_queue = queue.Queue()
rolling_buffer = deque(maxlen=int(3 * SAMPLE_RATE))
command_buffer = []
recording = False
wake_detected = False
last_speech_time = time.time()
command_start_delay = 0.0  # <--- NOU
buffer_lock = threading.Lock()

# ================================
# VAD
# ================================
def is_speech(chunk, threshold=0.01):
    return np.mean(np.abs(chunk)) > threshold

# ================================
# WAKE WORD
# ================================
def detect_wake_word():
    global wake_detected, recording, command_buffer, command_start_delay
    if wake_detected or len(rolling_buffer) < SAMPLE_RATE:
        return

    audio = np.array(rolling_buffer, dtype=np.float32)
    try:
        segments, _ = model.transcribe(
            audio, language="en", beam_size=1, word_timestamps=True, temperature=0.0
        )
        text = " ".join(s.text for s in segments).lower()
        if WAKE_WORD in text:
            wake_detected = True
            recording = True
            command_buffer = list(audio[-int(SAMPLE_RATE * 1):])
            command_start_delay = time.time() + WAKE_WORD_DELAY  # <--- 1.5s pauză
            print(f"\n[WAKE WORD DETECTED: {WAKE_WORD.upper()}]")
            threading.Thread(target=play_beep, daemon=True).start()
            with buffer_lock:
                rolling_buffer.clear()
    except Exception as e:
        print("Wake error:", e)

# ================================
# AUDIO CALLBACK
# ================================
def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    chunk = indata.copy().astype(np.float32).flatten()
    audio_queue.put(chunk)

# ================================
# WORKER
# ================================
def worker():
    global recording, last_speech_time, command_buffer, wake_detected, command_start_delay

    print("Say 'GARMIN' to activate... (you have 1.5s after beep)\n")

    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break

        with buffer_lock:
            rolling_buffer.extend(chunk)

        if not wake_detected and len(rolling_buffer) % int(SAMPLE_RATE * 0.5) < len(chunk):
            threading.Thread(target=detect_wake_word, daemon=True).start()

        if wake_detected:
            current_time = time.time()

            # === PAUZĂ DE 1.5s DUPĂ BEEP ===
            if current_time < command_start_delay:
                last_speech_time = current_time  # ține VAD-ul activ
                command_buffer.extend(chunk)
                continue  # nu verifica pauza încă

            # === DUPĂ 1.5s: începe VAD-ul normal ===
            command_buffer.extend(chunk)
            if is_speech(chunk):
                last_speech_time = current_time
            else:
                if current_time - last_speech_time > PAUSE_THRESHOLD:
                    if len(command_buffer) > SAMPLE_RATE * MIN_COMMAND_DURATION:
                        transcribe_command()
                    command_buffer.clear()
                    wake_detected = False
                    recording = False

            if len(command_buffer) > SAMPLE_RATE * 15:
                transcribe_command()
                command_buffer.clear()
                wake_detected = False
                recording = False

# ================================
# TRANSCRIBE
# ================================
def transcribe_command():
    if not command_buffer:
        return
    audio = np.array(command_buffer, dtype=np.float32)
    print("Transcribing command...")
    try:
        segments, _ = model.transcribe(audio, language="en", beam_size=5, temperature=0.0)
        text = " ".join(s.text for s in segments).strip()
        if text:
            print(f"Command: {text}")
    except Exception as e:
        print("Error:", e)

# ================================
# MAIN
# ================================
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='float32',
    callback=audio_callback,
    blocksize=int(CHUNK_DURATION * SAMPLE_RATE),
    device=DEVICE_INDEX
)

print("Voice Assistant STARTED.\n")

with stream:
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        audio_queue.put(None)