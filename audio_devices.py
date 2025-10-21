import sounddevice as sd

print("Available devices:")
print(sd.query_devices())
print("Default device",sd.default.device)