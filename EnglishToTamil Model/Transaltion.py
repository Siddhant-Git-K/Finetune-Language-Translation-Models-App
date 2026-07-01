import speech_recognition as sr
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
from playsound import playsound
import os
import torch

# =========================
# LOAD MODEL (ONLY ONE)
# =========================

en_ta_path = r"C:\Langauge Translator Project\en_tm langauge translator\en_to_ta"

en_ta_tokenizer = MarianTokenizer.from_pretrained(en_ta_path)
en_ta_model = MarianMTModel.from_pretrained(en_ta_path)

# =========================
# SPEECH INPUT
# =========================

recognizer = sr.Recognizer()

def listen():
    with sr.Microphone() as source:
        print("🎤 Speak in English...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language="en-IN")
        print("You said:", text)
        return text
    except:
        print("❌ Could not understand")
        return ""

# =========================
# TRANSLATION (IMPORTANT FIX)
# =========================

def en_to_ta(text):
    with torch.no_grad():
        text = ">>ta<< " + text   # 👈 CRITICAL
        inputs = en_ta_tokenizer(text, return_tensors="pt", padding=True)
        output = en_ta_model.generate(**inputs, max_length=60)

    return en_ta_tokenizer.decode(output[0], skip_special_tokens=True)

# =========================
# TEXT TO SPEECH
# =========================

def speak(text):
    tts = gTTS(text=text, lang="ta")
    filename = "output.mp3"
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

# =========================
# MAIN LOOP
# =========================

while True:
    cmd = input("\nPress ENTER to speak (q to quit): ")

    if cmd.lower() == "q":
        break

    text = listen()
    if text:
        translated = en_to_ta(text)
        print("Tamil:", translated)
        speak(translated)

print("✅ Program ended")