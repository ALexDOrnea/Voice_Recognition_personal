# Voice Recognition Personal Assistant — Technical Documentation

This project implements a **fully local, wake‑word‑activated voice assistant** built in Python. It uses **Faster Whisper** for real‑time speech recognition and **RapidFuzz** for high‑accuracy fuzzy matching between transcribed commands and a list of command variants defined in a CSV file. The transcribed command is sent to a **Raspberry Pi 5** connected to the same network as the laptop, and it executes various actions by controlling the GPIO pins.

The system is optimized for low latency, continuous audio streaming, robust wake‑word detection, and modular command handling. 

---

# System Architecture

```
 ┌──────────────────────────────────────────────────────────┐
 │                         MAIN LOOP                        │
 │  - Audio streaming from microphone                       │
 │  - Wake-word detection ("garmin")                        │
 │  - Command segmentation & buffering                      │
 └───────────────┬──────────────────────────────────────────┘
                 │
                 ▼
       ┌────────────────────┐
       │  Faster Whisper    │
       │  Speech-to-Text    │
       └────────────────────┘
                 │ text
                 ▼
       ┌────────────────────┐
       │  RapidFuzz Matcher │
       │  Command Mapping   │
       └────────────────────┘
                 │ action
                 ▼
       ┌──────────────────────────────────────────┐
       │  Raspberry Pi 5 — MQTT Action Executor   │
       │  - Subscribes to command topics          │
       │  - Parses received actions               │
       │  - Executes GPIO commands                │
       └──────────────────────────────────────────┘

```

---

# Audio Processing Pipeline

The audio system is built around:

* `sounddevice.InputStream()` for real-time audio capture
* A **worker thread** that processes audio chunks
* A **rolling buffer** used for wake‑word scanning
* A dedicated **command buffer** used to accumulate speech after activation

### Key Parameters

| Variable                    | Meaning                                     |
| --------------------------- | ------------------------------------------- |
| `SAMPLE_RATE=16000`         | Model‑friendly sample rate                  |
| `CHUNK_DURATION=0.5s`       | Audio chunk size                            |
| `PAUSE_THRESHOLD=1.0s`      | Silence that ends command                   |
| `MIN_COMMAND_DURATION=1.0s` | Minimum speech duration required            |
| `WAKE_WORD="garmin"`        | Activation phrase                           |
| `WAKE_WORD_DELAY=1.2s`      | Delay after wake word until command capture |

---

# 💤 Wake-Word Detection

Wake‑word detection is performed by running Faster Whisper on a rolling 1.5–3 second buffer.

### Steps:

1. Rolling audio buffer updated by `audio_callback`
2. Worker thread calls `detect_wake_word()` asynchronously
3. Segments are transcribed with VAD enabled
4. If wake word is found:

   * Assistant enters **recording mode**
   * Beep sound played
   * Command buffer initialized
   * Debounce applied to avoid double triggers

### Snippet

```python
def detect_wake_word(audio):
    segments, _ = model.transcribe(audio, vad_filter=True)
    text = " ".join(seg.text for seg in segments).lower()
    if WAKE_WORD in text:
        wake_detected = True
```

---

# Command Recognition

Once recording is active:

* All speech is accumulated in `command_buffer`
* Silence is measured using `is_speech()`
* When silence exceeds `PAUSE_THRESHOLD`, the assistant transcribes the full buffer

### Transcription

```python
segments, _ = model.transcribe(audio, beam_size=3, vad_filter=True)
text = " ".join(seg.text for seg in segments).lower()
```

Wake‑word is removed from the command if present:

```python
text = text.replace(WAKE_WORD, "").strip()
```

---

# Command Matching (RapidFuzz)

Commands are defined in `commands.csv` with the following schema:

```
command_key,variants,action
```

Example:

```
turn_on_led,turn on led|activate led|...,LED ON
```

### Loading Commands

```python
var2act, variants = load_commands("commands.csv")
```

* `var2act` maps each phrase variant → action
* `variants` is a flat list used by RapidFuzz

### Matching

```python
result = process.extractOne(text, variants, score_cutoff=cutoff)
```

* The best matching variant above `cutoff` (default 70%) is selected

---

# Running the Project

Install dependencies:

```bash
pip install sounddevice numpy faster-whisper rapidfuzz torch paho-mqtt
```

Run assistant:

```bash
python main.py
```
On the Raspberry Pi, the Mosquitto broker needs to be activated:

```
sudo systemctl enable mosquitto
sudo systemctl start mosquitto
```
All the code that should be on the Raspberry for the project to work is in the **RPi_code** directory.
Run subscriber on the Raspberry:

```
python mqtt_subscriber.py
```
Now start sending commands to the vocal assistant!

---


# Security

This example uses an unauthenticated broker (local network). For production, secure Mosquitto with TLS and authentication.


# License

This documentation describes a technical research-oriented project intended for local experimentation and personal automation.
