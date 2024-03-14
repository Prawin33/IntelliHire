import cv2
import face_recognition

def load_image_candidate_face(given_image):
    # Load the known image (the static picture)
    # known_image = face_recognition.load_image_file("known_person.jpg")
    candidate_image = face_recognition.load_image_file(given_image)
    candidate_encoding = face_recognition.face_encodings(candidate_image)[0]
    return candidate_encoding


def capture_candidate_face_from_video(frame):
    # Initialize some variables
    candidate_face_locations = []
    candidate_face_encodings = []
    # face_names = []

    # Find all the faces and face encodings in the current frame
    candidate_face_locations = face_recognition.face_locations(frame)
    candidate_face_encodings = face_recognition.face_encodings(frame, candidate_face_locations)
    return candidate_face_locations, candidate_face_encodings


def match_face(frame, live_face_locations, live_face_encodings, static_face_encoding):
    face_names = []

    for face_encoding in live_face_encodings:
        # Check if the face matches the known face
        matches = face_recognition.compare_faces([static_face_encoding], face_encoding)
        name = "Unverified Candidate"

        if matches[0]:
            name = "Verified Candidate"
            print("Known Person")

        face_names.append(name)
        print(f"faces_name {face_names}")

    # Display the results
    print("Display the results")
    for (top, right, bottom, left), name in zip(live_face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    print("Display the resulting frame")
    cv2.imshow('Video', frame)

    # # Break the loop when 'q' is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    #
    # # Close windows
    # # cap.release()
    # cv2.destroyAllWindows()
