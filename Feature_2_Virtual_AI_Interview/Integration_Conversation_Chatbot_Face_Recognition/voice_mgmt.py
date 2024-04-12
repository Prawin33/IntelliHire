import os

# for speech-to-text
import speech_recognition as sr

# for text-to-speech
from gtts import gTTS
import pygame

# for language model
import transformers

# # for posting messages to the webapp
# import requests

# Build the AI
class ChatBot():
    def __init__(self, name):
        print("--- starting up", name, "---")
        self.name = name # Name of the chatbot
        pygame.mixer.init() # Initialization of the mixer module of pygame, used for playing back the audio file

    def speech_to_text(self):
        '''
        Function to convert the speech captured from the microphone to text.
        The speech converted to text is printed out on the console.
        Speech Recognition library has been exploited in this function.

        Returns:
        The text from the speech is returned.
        '''

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

        # # Update interviewee window with response text
        # requests.post("http://127.0.0.1:5000", data={"text": self.text})

        return self.text

    # def wake_up(self, text):
    #     return True if self.name in text.lower() else False

    @staticmethod
    def text_to_speech(audio_file, text):
        '''
        Function used to convert text to speech.
        Google's text-to-speech library has been utilized in this function

        Args:
            audio_file: empty audio file with its complete path
            text: the text (interview questions & answers as well as intro message from the chatbot)

        Returns:

        '''
        print("ai --> ", text)
        speaker = gTTS(text=text, lang="en", slow=False)
        # speaker.save("res.mp3")
        speaker.save(audio_file)

        # # Update interviewer window with response text
        # requests.post("http://127.0.0.1:5000", data={"text": text})

        print('')
        # # Attempt to use OS to replay the saved audio file
        # os.system("start res.mp3")  # macbook->afplay | windows->start
        # os.system("start " + '\\audio_file')  # macbook->afplay | windows->start
        # os.system(f'start "" "{audio_file}"')

        # Replay the audio file by first loading the file
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        # Wait till the audio file is being played
        while pygame.mixer.music.get_busy():
            pass

        # Stop the pygame mixer and unload the audio file
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()

        # Remove the saved audio file
        # os.remove("res.mp3")
        os.remove(audio_file)

    # # Functions implemented for the initial basic chatbot. They are not used in IntelliHire SW.
    # @staticmethod
    # def action_time():
    #     return datetime.datetime.now().time().strftime('%H:%M')
    #
    # @staticmethod
    # def polite_response():
    #     return np.random.choice(["you're welcome!", "anytime!", "no problem!", "cool!", "I'm here if you need me!", "peace out!"])
    #
    # @staticmethod
    # def conversation(text, nlp):
    #     chat = nlp(transformers.Conversation(text), pad_token_id=50256)
    #     res = str(chat)
    #     res = res[res.find("bot >> ") + 6:].strip()
    #     return res