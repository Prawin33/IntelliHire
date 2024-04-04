# for data
import os
import time
import datetime
import numpy as np

# for speech-to-text
import speech_recognition as sr

# for text-to-speech
from gtts import gTTS

# for language model
import transformers

import pygame


# Build the AI
class ChatBot():
    def __init__(self, name):
        print("--- starting up", name, "---")
        self.name = name
        pygame.mixer.init()

    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            recognizer.adjust_for_ambient_noise(mic, duration=1)
            print("listening...")
            audio = recognizer.listen(mic, timeout=10)
        try:
            self.text = recognizer.recognize_google(audio)
            print("me --> ", self.text)
        except:
            self.text = "ERROR"
            print("me -->  ", self.text)

        return self.text

    def wake_up(self, text):
        return True if self.name in text.lower() else False

    @staticmethod
    def text_to_speech(audio_file, text):
    # def text_to_speech(text):
        print("ai --> ", text)
        speaker = gTTS(text=text, lang="en", slow=False)
        # speaker.save("res.mp3")
        speaker.save(audio_file)

        # time.sleep(15)
        print('')
        # os.system("start res.mp3")  # macbook->afplay | windows->start
        # os.system("start " + '\\audio_file')  # macbook->afplay | windows->start
        # os.system(f'start "" "{audio_file}"')
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pass

        pygame.mixer.music.stop()
        pygame.mixer.music.unload()

        # os.remove("res.mp3")
        os.remove(audio_file)

    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')

    @staticmethod
    def polite_response():
        return np.random.choice(["you're welcome!", "anytime!", "no problem!", "cool!", "I'm here if you need me!", "peace out!"])

    @staticmethod
    def conversation(text, nlp):
        chat = nlp(transformers.Conversation(text), pad_token_id=50256)
        res = str(chat)
        res = res[res.find("bot >> ") + 6:].strip()
        return res