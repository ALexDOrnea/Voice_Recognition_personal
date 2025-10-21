import whisper #interpretor voce
import sounddevice as sd #utilizare microfon
import numpy as np #matrici si vectori
import queue #pentru a stoca safe chunckuri de fisiere audio
import threading #pentru a crea mai multe threaduri (transformarea in text se face in fundal, pe alt thread)
import tempfile #pentru a crea fisiere temporare audio
import os #epentru handling fisiere audio
import scipy.io.wavfile as wav #pentru a salva audiouri in format wav

#model whisper tiny base small medium large
model=whisper.load_model("base")

#Parametrii audio
sample_rate=16000 #Khz -bitrate
block_duration=2 #s -duratia fiecarui chunk audio

#creaza un queue care ajuta la stocarea fisierelor wav de la microfon
audio_queue=queue.Queue()

#functie callback care se activeaza de fiecare data cand microfonul detecteaza audio
#indata- array cu numpy ce contine recordingul, frames-numarul frameurilor din bloc, time-informatii despre timp status-erori
def audio_callback(in_data, frame_count, time_info, status):
    if status:
        print("status")
    audio_queue.put(in_data.copy())

#functie pentru transcribe -ruleaza constant in fundal
def transcribe_stream():

    print("transcribing audio")

    while True:
        #obtine un nou chunk audio
        audio_chunk=audio_queue.get()

        #il salveaza intr-un fisier temporar WAV
        with tempfile.NamedTemporaryFile(delete=False,suffix=".wav") as tmpfile:
            wav.write(tmpfile.name, sample_rate, (audio_chunk*32767).astype(np.int16))

            #trimite la whisper
            result=model.transcribe(tmpfile.name,fp16=False,language="en")

            #extract si clean up al textului
            text=result["text"].strip()

            #printare text
            if text:
                print(text)

            #stergere fisiere
            os.remove(tmpfile.name)

#capturare audio microfon
stream = sd.InputStream(
    samplerate=sample_rate, #bitrate de 16khz
    channels=1, #input mono
    callback=audio_callback, #functia folosita pentru fiecare chunck audio
    blocksize=int(block_duration*sample_rate), #numarul de samples per chunk
    device=8
)

#program principal
with stream:
    #incepe transcriptia in alt thread
    transcribe_thread=threading.Thread(target=transcribe_stream, daemon=True)
    transcribe_thread.start()

    try:
        #main thread functioneaza while true
        while True:
            pass
    except KeyboardInterrupt:
        print("exiting")