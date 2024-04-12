import cv2
import face_recognition

class ImageProcessing_Block():
    def __init__(self):
        print('Initializing the face recognition/ image processing block')

    def load_image_candidate_face(self, given_image):
        '''
        Function to load an image from the specified location to the face_recognition module
        for generating image encodings. These encodings are used to match the static image
        to the face captured from the camera.

        Args:
            given_image: static image to be used for facial matching

        Returns:
            candidate_encoding: generated image encodings

        '''
        # Load the known image (the static picture)
        candidate_image = face_recognition.load_image_file(given_image)

        # Generate face encodings for the image
        candidate_encoding = face_recognition.face_encodings(candidate_image)[0]
        return candidate_encoding


    def capture_candidate_face_from_video(self, frame):
        '''
        Function to extract a face from the video captured from the camera

        Args:
            frame: captured video frame

        Returns:
            Face locations and encodings

        '''

        # Initialize some variables
        candidate_face_locations = []
        candidate_face_encodings = []

        # Find all the faces and face encodings in the current frame
        candidate_face_locations = face_recognition.face_locations(frame)
        candidate_face_encodings = face_recognition.face_encodings(frame, candidate_face_locations)
        return candidate_face_locations, candidate_face_encodings


    def match_face(self, frame, live_face_locations, live_face_encodings, static_face_encoding):
        '''
        Function to match faces in the video frame and static images
        Args:
            frame: captured video frame
            live_face_locations: face locations in the captured video frames
            live_face_encodings: face encoding in the captured video frames
            static_face_encoding: face locations in the static images

        Returns:
            The status if there is match between the frame face and the static image face

        '''
        face_names = []
        name = "init face match process"

        for face_encoding in live_face_encodings:
            # Check if the face matches the known face
            matches = face_recognition.compare_faces([static_face_encoding], face_encoding)
            name = "Unverified Candidate"

            if matches[0]:
                name = "Verified Candidate"
                # print("Known Person")

            face_names.append(name)
            print(f"faces_name {face_names}")

        # Display the results
        # print("Display the results")
        for (top, right, bottom, left), name in zip(live_face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        return name