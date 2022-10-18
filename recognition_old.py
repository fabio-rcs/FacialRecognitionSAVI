#!/usr/bin/env python3
import face_recognition
import cv2
import numpy as np

class Recognition:
    def __init__(self):
        pass
    def process_frame(self,frame,known_faces):
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # BGR to RGB
        rgb_small_frame = small_frame[:, :, ::-1]
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        #face_locations_temp = []
        #for (top, right, bottom, left) in face_locations:
        #    top *= 4
        #    right *= 4
        #    bottom *= 4
        #    left *= 4
        #    face_locations_temp.append((top, right, bottom, left))
        # Variable separation
        known_face_encodings, known_face_names = known_faces
        # Start new variable to store the names in the frame
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
#        face_locations = face_locations_temp
        return zip(face_locations, face_names)

    def draw_rectangles(self, frame, face_inframe, color):
        (top, right, bottom, left), name = face_inframe
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        font = cv2.FONT_HERSHEY_DUPLEX
        textsize = cv2.getTextSize(name, font, 1.0, 1)
        cv2.putText(frame, name, (int((left + right)/2) - int(textsize[0][0] / 2) , bottom + int(textsize[0][1] / 2) + 4 + textsize[1]), font, 1.0, (0, 0, 255), 1)

    def encode_frame(self, frame, face_location):
        image_encoded = face_recognition.face_encodings(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))[0]
        return image_encoded
        