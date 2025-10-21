import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample

duration = 3
sample_rate = 16000  # native for ALC287
device_index = 5     # your analog mic

print("🎙️ Using device:", sd.query_devices(device_index)['name'])
print("🎙️ Speak now...")

audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
               channels=1, dtype='float32', device=device_index)
sd.wait()

# 🔁 Resample to 16kHz for Whisper
target_rate = 16000
resampled = resample(audio, int(len(audio) * target_rate / sample_rate))

wav.write("test_mic.wav", target_rate, (resampled * 32767).astype(np.int16))
print("✅ Saved as test_mic.wav (resampled to 16 kHz)")
