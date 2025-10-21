import sounddevice as sd

print(sd.query_devices(5))       # or the index of your mic
print("\nSupported samplerates:")
for rate in [8000,16000,22050,32000,44100,48000]:
    try:
        sd.check_input_settings(device=5, samplerate=rate)
        print(f"✅ {rate} Hz works")
    except Exception as e:
        print(f"❌ {rate} Hz not supported:", e)
