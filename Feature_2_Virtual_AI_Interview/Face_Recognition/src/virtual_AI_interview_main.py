import os
import sys

import random
import cv2
import face_mgmt


def main_func():
    path_images_folder = r"C:\Batool_Data (2024)\Privat\BT_Related to Durham College\Semester #1\AIDI_1003_Capstone_Term_Project\IntelliHire_ConversationAIForVirtualInterview\src\images"
    file_name = "BT_sample_image1.jpg"
    # file_name = "LordDurham_sample_image1.jpg"
    path_candidate_image_file = path_images_folder + "\\" + file_name

    req_encoding_from_candidate_image = face_mgmt.load_image_candidate_face(given_image=path_candidate_image_file)

    cap = cv2.VideoCapture(0)

    try:
        while True:  # for customers
            new_face = True
            messages = []
            # face_encoding, face_id, person_name = None, None, None

            while True:  # for camera
                ret, frame = cap.read()

                if not ret:
                    print("Error: Unable to capture video.")
                    break

                live_candidate_face_locations, live_candidate_face_encodings = face_mgmt.capture_candidate_face_from_video(frame=frame)

                # cv2.imshow('Video', frame)
                # # To load and hold the image
                # cv2.waitKey(0)

                face_mgmt.match_face(frame=frame,
                                     live_face_locations=live_candidate_face_locations,
                                     live_face_encodings=live_candidate_face_encodings,
                                     static_face_encoding=req_encoding_from_candidate_image)

                # Break the loop when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        # Release the video capture object when done
        cap.release()
        cv2.destroyAllWindows()
