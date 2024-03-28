import os
import sys

import random
import voice_mgmt
import nlp_mgmt
import numpy as np
import resources.dataset as ds

## for language model
import transformers




def main_func():
    path_audios_folder = r"C:\Batool_Data (2024)\Privat\BT_Related to Durham College\Semester #1\AIDI_1003_Capstone_Term_Project\IntelliHire_ConversationAIForVirtualInterview\src\audios"
    file_name = "res.mp3"
    path_candidate_audio_file = path_audios_folder + "\\" + file_name

    ai = voice_mgmt.ChatBot(name="maya")
    nlp = nlp_mgmt.NLP_Block()
    # nlp = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium")
    # os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Start the interview
    res = "Hello! I'm InterviewBot. Let's start the interview."
    ai.text_to_speech(audio_file=path_candidate_audio_file, text=res)
    for question in ds.ml_engineer_questions:
        ai.text_to_speech(audio_file=path_candidate_audio_file, text=question)

        user_input = ai.speech_to_text()

        # Summarize user's response
        summarized_response = nlp.summarize_text(text=user_input)
        print(f"Summary: {summarized_response}")

        nlp.compare_candidates_answers_with_fixed_answers(summarized_response=summarized_response, fixed_answers=ds.fixed_answers)
        print('')
        pass

    # while True:
    #     ai.speech_to_text()
    #
    #     ## wake up
    #     if ai.wake_up(ai.text) is True:
    #         res = "Hello I am Maya the AI, what can I do for you?"
    #
    #     ## action time
    #     elif "time" in ai.text:
    #         res = ai.action_time()
    #
    #     ## respond politely
    #     elif any(i in ai.text for i in ["thank", "thanks"]):
    #         res = ai.polite_response()
    #
    #     ## conversation
    #     else:
    #         res = ai.conversation(ai.text, nlp)
    #
    #     ai.text_to_speech(path_candidate_audio_file, res)
    #     # ai.text_to_speech(res)

# Run the AI
if __name__ == "__main__":
    main_func()