import os
import sys

import random
import voice_mgmt
import numpy as np

## for language model
import transformers




def main_func():
    path_audios_folder = r"C:\Batool_Data (2024)\Privat\BT_Related to Durham College\Semester #1\AIDI_1003_Capstone_Term_Project\IntelliHire_ConversationAIForVirtualInterview\src\audios"
    file_name = "res.mp3"
    path_candidate_audio_file = path_audios_folder + "\\" + file_name

    ai = voice_mgmt.ChatBot(name="maya")
    nlp = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    while True:
        ai.speech_to_text()

        ## wake up
        if ai.wake_up(ai.text) is True:
            res = "Hello I am Maya the AI, what can I do for you?"

        ## action time
        elif "time" in ai.text:
            res = ai.action_time()

        ## respond politely
        elif any(i in ai.text for i in ["thank", "thanks"]):
            res = ai.polite_response()

        ## conversation
        else:
            res = ai.conversation(ai.text, nlp)

        ai.text_to_speech(path_candidate_audio_file, res)
        # ai.text_to_speech(res)

# Run the AI
if __name__ == "__main__":
    main_func()