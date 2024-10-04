import speech_recognition 
import streamlit as st

def get_audio():
    recognizer = speech_recognition.Recognizer()

    st.write("Please remain silent, calibrating microphone for ambient noise...")
    
    with speech_recognition.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        st.write("You can now speak...")
        audio = recognizer.listen(source)
        st.write("Recording finished, processing...")

    try:
        text = recognizer.recognize_google(audio)
        return text
    except speech_recognition.UnknownValueError:
        return "Google Speech Recognition could not understand the audio"
    except speech_recognition.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"
