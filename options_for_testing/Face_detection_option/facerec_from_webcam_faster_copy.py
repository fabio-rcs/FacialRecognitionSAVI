#!/usr/bin/env python3
import face_recognition
import cv2
import numpy as np
from recognition import Recognition


# Get a reference to webcam #0 (the default one)V
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("imagem-teste2.png")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("imagem-teste.png")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Goncalo Ribeiro",
    "Goncalo Ribeiro"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
recognition = Recognition()
cycle_interval = 2
cycle = 0
while True:
    cycle += 1
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time

    if process_this_frame or cycle>=cycle_interval:
        process_this_frame = False
        cycle = 0
        face_locations, face_names, face_encodings = recognition.process_frame(frame,known_face_encodings, known_face_names)
    # Display the results
    for (top, right, bottom, left), name, face_encoding in zip(face_locations,face_names, face_encodings):
        if name == 'Unknown':
            recognition.draw_rectangles(frame, ((top, right, bottom, left), name), (0,0,255))

        else:
            recognition.draw_rectangles(frame, ((top, right, bottom, left), name), (0,255,0))
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
