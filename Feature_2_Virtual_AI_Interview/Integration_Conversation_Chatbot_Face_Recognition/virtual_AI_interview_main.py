# virtual_AI_interview_main.py

import os
import cv2
import face_mgmt
import voice_mgmt
import nlp_mgmt
import resources.dataset as ds

# Initialize the ChatBot and NLP_Block instances
ai = voice_mgmt.ChatBot(name="IntelliHire")
nlp = nlp_mgmt.NLP_Block()
ip = face_mgmt.ImageProcessing_Block()

# Callback function for starting the verification process
def start_verification():
    # global verification_completed
    path_images_folder = r"C:\Batool_Data (2024)\Privat\BT_Related to Durham College\Semester #1\AIDI_1003_Capstone_Term_Project\IntelliHire_ConversationAIForVirtualInterview\src\images"
    file_name = "BT_sample_image1.jpg"
    path_candidate_image_file = os.path.join(path_images_folder, file_name)

    # Load the image & generate the face encodings from the image
    req_encoding_from_candidate_image = ip.load_image_candidate_face(given_image=path_candidate_image_file)

    # Start the camera & video capture
    cap = cv2.VideoCapture(0)

    try:
        while True:  # for interviewee

            while True:  # for camera
                # Read the captured video
                ret, frame = cap.read()

                if not ret:
                    print("Error: Unable to capture video.")
                    break

                # Extract face from the captured video frame
                live_candidate_face_locations, live_candidate_face_encodings = ip.capture_candidate_face_from_video(frame=frame)

                # Match faces, i.e., the face from the static image & the one extracted from the captured video frame
                face_match_status = ip.match_face(frame=frame,
                                                         live_face_locations=live_candidate_face_locations,
                                                         live_face_encodings=live_candidate_face_encodings,
                                                         static_face_encoding=req_encoding_from_candidate_image)

                # Display the captured video frame
                cv2.imshow('Verification Camera', frame)

                # If face matched, set verification_completed flag to True
                if face_match_status == "Verified Candidate":
                    # verification_completed = True
                    print("Candidate Verified Successfully")
                    with open("verification_status.txt", "w") as f:
                        f.write("True")
                    break

                # Break the loop when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        # Release the video capture object when done
        cap.release()
        cv2.destroyAllWindows()

# Callback function for starting the interview
def start_interview():
    path_audios_folder = r"C:\Batool_Data (2024)\Privat\BT_Related to Durham College\Semester #1\AIDI_1003_Capstone_Term_Project\IntelliHire_ConversationAIForVirtualInterview\src\audios"
    file_name = "res.mp3"
    path_candidate_audio_file = os.path.join(path_audios_folder, file_name)

    # Start the interview
    res = "Hello! I'm IntelliHireBot. Let's start the interview."
    ai.text_to_speech(audio_file=path_candidate_audio_file, text=res)

    # Loop through the interview questions
    for question in ds.ml_engineer_questions:
        # Chatbot asking the interview question
        ai.text_to_speech(audio_file=path_candidate_audio_file, text=question)

        # Recording inteviewee's response by converting the speech to text
        interviewee_input = ai.speech_to_text()

        # Summarize interviewee's response
        summarized_response = nlp.summarize_text(text=interviewee_input)
        print(f"Summary: {summarized_response}")

        # Compare the summarized response from the interviewee and the benckmark answers
        nlp.compare_candidates_answers_with_fixed_answers(summarized_response=summarized_response, benchmark_answers=ds.fixed_answers)
        print('')

# Entry point of the script
if __name__ == "__main__":
    # Run the verification process
    start_verification()
