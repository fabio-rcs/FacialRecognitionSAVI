#!/usr/bin/env python3
import face_recognition
import cv2
import numpy as np

frame_reduction = 4

class Recognition:
    def __init__(self,frame,known_face_encodings, known_face_names):
        self.frame = frame
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names

    def process_frame(self):
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(self.frame, (0, 0), fx=1/frame_reduction, fy=1/frame_reduction)
        # BGR to RGB
        rgb_small_frame = small_frame[:, :, ::-1]
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        # Start new variable to store the names in the frame
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)
        return face_locations, face_names, face_encodings

    def draw_rectangles(self, face_inframe, color):
        (top, right, bottom, left), name = face_inframe
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= frame_reduction
        right *= frame_reduction
        bottom *= frame_reduction
        left *= frame_reduction

        # Draw a box around the face
        cv2.rectangle(self.frame, (left, top), (right, bottom), color, 2)

        # Draw the name below the face
        font = cv2.FONT_HERSHEY_DUPLEX
        textsize = cv2.getTextSize(name, font, 1.0, 1)
        cv2.putText(self.frame, name, (int((left + right)/2) - int(textsize[0][0] / 2) , bottom + int(textsize[0][1] / 2) + textsize[0][1]), font, 1.0, color, 1)

    def remove_name(self,original_frame, final_frame, face_location, name):
        (top, right, bottom, left) = face_location
        top *= frame_reduction
        right *= frame_reduction
        bottom *= frame_reduction
        left *= frame_reduction
        font = cv2.FONT_HERSHEY_DUPLEX
        textsize = cv2.getTextSize(name, font, 1.0, 1)
        mask = original_frame[(bottom - int(textsize[0][1] / 2) + textsize[0][1]) : (bottom + int(textsize[0][1] / 2) + textsize[0][1]),(int((left + right)/2) - int(textsize[0][0] / 2)):(int((left + right)/2) + int(textsize[0][0] / 2))]
        final_frame[(bottom - int(textsize[0][1] / 2) + textsize[0][1]) : (bottom + int(textsize[0][1] / 2) + textsize[0][1]),(int((left + right)/2) - int(textsize[0][0] / 2)):(int((left + right)/2) + int(textsize[0][0] / 2))] = mask
        #cv2.imshow('unknown', mask)


        # cv2.putText(self.frame, name, (int((left + right)/2) - int(textsize[0][0] / 2) , bottom + int(textsize[0][1] / 2) + 4 + textsize[1]), font, 1.0, color, 1)
    # def save_face(self, face_names, face_encodings):
    #     image_encoded = face_recognition.face_encodings(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))[0]
    #     return image_encoded
        