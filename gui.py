# gui.py
import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import threading

import main_1_0 as engine   # 🔴 redenumește fișierul: main_1_0.py (fără punct!)

class VoiceAssistantGUI:
    def __init__(self, root):
        self.root = root
        root.title("Voice Assistant")

        self.device_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="Stopped")

        self.running = False

        self.build_ui()

    def build_ui(self):
        frame = tk.Frame(self.root, padx=10, pady=10)
        frame.pack()

        tk.Label(frame, text="Microphone device index:").grid(row=0, column=0, sticky="w")
        tk.Entry(frame, textvariable=self.device_var, width=5).grid(row=0, column=1)

        tk.Button(frame, text="List devices", command=self.list_devices).grid(row=0, column=2, padx=5)

        tk.Button(frame, text="Start", bg="green", fg="white", command=self.start).grid(row=1, column=0, pady=10)
        tk.Button(frame, text="Stop", bg="red", fg="white", command=self.stop).grid(row=1, column=1)

        tk.Label(frame, textvariable=self.status_var, fg="blue").grid(row=2, column=0, columnspan=3)

    def list_devices(self):
        devices = sd.query_devices()
        msg = ""
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                msg += f"{i}: {d['name']}\n"
        messagebox.showinfo("Input devices", msg)

    def start(self):
        if self.running:
            return

        self.running = True
        self.status_var.set("Running...")

        threading.Thread(
            target=engine.start_assistant,
            args=(self.device_var.get(),),
            daemon=True
        ).start()

    def stop(self):
        if not self.running:
            return

        engine.stop_assistant()
        self.running = False
        self.status_var.set("Stopped")


if __name__ == "__main__":
    root = tk.Tk()
    VoiceAssistantGUI(root)
    root.mainloop()
