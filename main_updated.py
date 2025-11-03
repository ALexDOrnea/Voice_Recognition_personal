import warnings  #avertismente
#warnings.filterwarnings("ignore")  # dezactiveaza avertistemte

import sounddevice as sd  #utilizare microfon
import numpy as np         #array-uri audio
import queue               #coada de seqmente audio
import threading           #creaza threaduri intre inregistrare si transcribe
import time                #pentru delay
from faster_whisper import WhisperModel  #whisper mai rapid
from collections import deque            #coada double-end

# SETTINGS
WAKE_WORD="garmin"
SAMPLE_RATE=16000
CHUNK_DURATION=0.5
PAUSE_THRESHOLD=1.0         #limita de timp fara voce pentru sfarsit comanda
MIN_COMMAND_DURATION=1.0
DEVICE_INDEX=None
WAKE_WORD_DELAY=1.5         # delay de la wake-word pana la comanda


#BEEP
def play_beep():
    duration=0.2 #beep duration
    t=np.linspace(0,duration,int(SAMPLE_RATE*duration),False) #vector de timp t cu puncte de esantionare
    tone=0.3*np.sin(2*np.pi*1000*t)  # ton sinusoidal 1000 Hz
    sd.play(tone,samplerate=SAMPLE_RATE)
    sd.wait() #asteapta sa se termine audio-ul de redat


#MODEL
print("Loading whisper...")
model=WhisperModel("tiny.en",device="cpu",compute_type="int8") ## device="cuda",compute_type="float16" -pt gpu

#VARIABILE GLOBALE
audio_queue=queue.Queue()  #Coada arrayuri
rolling_buffer=deque(maxlen=int(3*SAMPLE_RATE))  #buffer audio - ultimele 3 secunde din audio
command_buffer=[] #audio stocat pentru comanda vocala
recording=False
wake_detected=False #detectare cuvant activare
last_speech_time=time.time() #ultima activitate
command_start_delay=0.0   # timp dupa beep
buffer_lock=threading.Lock() #lock pentru acces sincronizat la buffer


#Voice Activity Detection
def is_speech(chunk,threshold=0.01):
    #daca un segment de voce contine audio -bazat pe amplitudine medie
    return np.mean(np.abs(chunk))>threshold


# WAKE WORD DETECTION
def detect_wake_word():
    #ruleaza pe ultimele 3 secunde din buffer(rolling buffer)
    global wake_detected,recording,command_buffer,command_start_delay
    if wake_detected or len(rolling_buffer)<SAMPLE_RATE:
        return  #daca deja e activat sau bufferul e prea mic, return

    audio=np.array(rolling_buffer,dtype=np.float32)
    try:
        #transcrie ultimele secunde din buffer
        segments,_=model.transcribe(
            audio,language="en",beam_size=1,word_timestamps=True,temperature=0.0
        )
        text = " ".join(s.text for s in segments).lower()
        #Cauta cuvântul de activare
        if WAKE_WORD in text:
            wake_detected=True
            recording=True
            #Salveaza ultima secunda de audio(ca să nu pierdem inceputul)
            command_buffer=list(audio[-int(SAMPLE_RATE*1):])
            command_start_delay=time.time()+WAKE_WORD_DELAY # pauza de 1.5s
            print(f"[WAKE WORD DETECTED: {WAKE_WORD.upper()}]")
            threading.Thread(target=play_beep,daemon=True).start()
            with buffer_lock:
                rolling_buffer.clear()  #reseteaza bufferul
    except Exception as e:
        print("Wake error:", e)


# AUDIO CALLBACK
def audio_callback(indata,frames,time_info,status):
    #callback apelat de souddevice care este executata dupa ce este inregistrat un chunk
    if status:
        print("Audio status:",status) #erori
    chunk=indata.copy().astype(np.float32).flatten()
    audio_queue.put(chunk)

# WORKER THREAD
def worker():
    #Thread principal care detecteaza wake wrod si apoi il inregistreaza si da transcribe
    global recording,last_speech_time,command_buffer,wake_detected,command_start_delay
    print(f"Say {WAKE_WORD} to activate...\n")

    while True:
        chunk=audio_queue.get()
        if chunk is None:
            break
        #adauga bucata curenta în bufferul circular
        with buffer_lock:
            rolling_buffer.extend(chunk)
        #detecteaza cuvantul WAKE_WORD la fiecare 0.5 secunde
        if not wake_detected and len(rolling_buffer)%int(SAMPLE_RATE * 0.5) < len(chunk):
            threading.Thread(target=detect_wake_word,daemon=True).start() #daca nu a fost detectat wake word si avem chunckuri intregi este cautat wake work

        #daca s a activat modul de ascultare pentru comanda
        if wake_detected:
            current_time=time.time()

            #pauza de 1.5 secunde
            if current_time<command_start_delay:
                last_speech_time=current_time
                command_buffer.extend(chunk)
                continue

            #procesare comanda
            command_buffer.extend(chunk)
            if is_speech(chunk):
                last_speech_time=current_time
            else:
                #finalizare comanda la pauza mai mare de 1s
                if current_time-last_speech_time>PAUSE_THRESHOLD:
                    if len(command_buffer)>SAMPLE_RATE*MIN_COMMAND_DURATION:
                        transcribe_command()
                    command_buffer.clear()
                    wake_detected = False
                    recording = False

            #dacă comanda dureaza prea mult (>15 sec) o finalizează automat
            if len(command_buffer)>SAMPLE_RATE*15:
                transcribe_command()
                command_buffer.clear()
                wake_detected=False
                recording=False

# TRANSCRIBE COMMAND
def transcribe_command():
    #da transcribe la bufferul de comanda si afiseaza
    if not command_buffer:
        return
    audio=np.array(command_buffer,dtype=np.float32)
    print("Transcribing command...")
    try:
        segments,_=model.transcribe(audio,language="en",beam_size=5,temperature=0.0)
        text = " ".join(s.text for s in segments).strip()
        if text:
            print(f"Command: {text}")
    except Exception as e:
        print("Error:", e)


# MAIN
#creaza stream audio
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='float32',
    callback=audio_callback,
    blocksize=int(CHUNK_DURATION*SAMPLE_RATE),
    device=DEVICE_INDEX
)
print("Program started\n")

#porneste thread procesare audio
with stream:
    t=threading.Thread(target=worker,daemon=True)
    t.start()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting..")
        audio_queue.put(None)
