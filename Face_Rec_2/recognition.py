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
            # Inicialize name variable
            name = "Unknown"
            # Get face "Distances"
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            # Get the minimal distance ID
            best_match_index = np.argmin(face_distances)
            # Get the best match with minimal distance
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            # Add name to the list of names in frame
            face_names.append(name)
        # Return some useful variables
        return face_locations, face_names, face_encodings

    def draw_rectangles(self, face_inframe, color):
        (top, right, bottom, left), name = face_inframe
        # Scale back up face locations since the frame we detected in was scaled down
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
        # Scale back up face locations since the frame we detected in was scaled down
        top *= frame_reduction
        right *= frame_reduction
        bottom *= frame_reduction
        left *= frame_reduction
        font = cv2.FONT_HERSHEY_DUPLEX
        # Get name size
        textsize = cv2.getTextSize(name, font, 1.0, 1)
        # Get a image from inicial frame with the same size and location of the name
        mask = original_frame[(bottom - int(textsize[0][1] / 2) + textsize[0][1]) : (bottom + int(textsize[0][1] / 2) + textsize[0][1] + 1),(int((left + right)/2) - int(textsize[0][0] / 2)):(int((left + right)/2) + int(textsize[0][0] / 2))]
        # Replace name with the previews image
        final_frame[(bottom - int(textsize[0][1] / 2) + textsize[0][1]) : (bottom + int(textsize[0][1] / 2) + textsize[0][1] + 1),(int((left + right)/2) - int(textsize[0][0] / 2)):(int((left + right)/2) + int(textsize[0][0] / 2))] = mask

    def identify_unknown(self, window_name, original_frame, analyzing_frame,unknown_idx, face_encodings, face_locations, state):
        state = False
        for i in unknown_idx:
            # Verify if is there any repeated face
            if True in face_recognition.compare_faces(self.known_face_encodings, face_encodings[i]):
                pass
            else:
                # Switch from unknown to question in the frame
                self.remove_name(original_frame, analyzing_frame, face_locations[i], 'Unknown')
                self.draw_rectangles((face_locations[i], 'Who are you?'), (255,0,0))

                # Update image
                cv2.imshow(window_name, analyzing_frame)
                cv2.waitKey(1000)

                # Get name
                name = input('Who are you?')

                # Switch from question to red painted name in the frame
                self.remove_name(original_frame, analyzing_frame, face_locations[i], 'Who are you?')
                self.draw_rectangles((face_locations[i], name), (0,0,255))

                # Add info to encodings and names list
                self.known_face_encodings.append(face_encodings[i])
                self.known_face_names.append(name)
                
                # Update image
                cv2.imshow(window_name, analyzing_frame)
        state = True