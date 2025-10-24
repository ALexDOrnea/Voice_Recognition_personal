# Voice_Recognition_personal
utilizare:

pip install openai-whisper sounddevice numpy

ruleaza audio_devices.py

incearca sa inregistrezi un audio demo cu microphone_test.py , 
selectand deviceul twu si sample rate 16000. daca nu merge incearca 
sa rulezi mic_specs.py cu microfonul dorit pentru a vedea birateurile acceptate, 
si le poti modifica in microphone test (sample rate). 
main.py este programul care functioneaza doar pe 16khz din pacate, selectezi la linia 
59 dispozitivul si speri sa mearga

fisiere::
--main.py -script origial extrem de ineficient
--main_faster.py -script optimizat, merge mult mai rapid
--fastest_whisper.py -foloseste faster_whisper -si mai rapid
--audio.devices.py -detecteaza dispozitive I/O audio
--microphone_test.py -test microfon 
--mic_specs --afiseaza birateurile suportate de microfon

momentan totul e facut de procesor,ceea ce face procesul mult mai lent.
e nevoie de placa video nvidia cu drivere si cuda. eu am o problema la cuda
si nu pot implementa rularea scriptului pe placa video, dar poate fi de 3-5
ori mai rapid.