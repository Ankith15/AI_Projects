import speech_recognition
# import pyttsx3

def get_audio():
    sr = speech_recognition.Recognizer()
    with speech_recognition.Microphone() as source2:
        print("Please Silence")

        sr.adjust_for_ambient_noise(source2,duration=2)

        print("speak for some time")
        audio2 = sr.listen(source2)

        

get_audio()




import streamlit as st
import sounddevice as sd
import numpy as np
import wavio
import speech_recognition as sr

# Function to record audio using sounddevice
def record_audio(duration=5, fs=44100):
    st.write("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    st.write("Recording finished")
    text = sr.recognize_google(recording)
    text = text.lower()
    print(text)
    return text, fs