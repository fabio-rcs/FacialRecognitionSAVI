#!/usr/bin/env python3
import face_recognition
import cv2
import numpy as np
from recognition import Recognition
import pickle

dir_db = './Database/database_group.pickle'
# Get a reference to webcam #0 (the default one)V
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
with open(dir_db, 'rb') as f:
     known_face_names, known_face_encodings = pickle.load(f)
# Create arrays of known face encodings and their names

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
cycle_interval =2 
cycle = 0
while True:
    cycle += 1
    # Grab a single frame of video
    ret, frame = video_capture.read()
    recognition = Recognition(frame, known_face_encodings, known_face_names)
    # Only process every other frame of video to save time

    if process_this_frame or cycle>=cycle_interval:
        process_this_frame = False
        cycle = 0
        try:
            face_locations, face_names, face_encodings = recognition.process_frame()
        except ValueError:
            pass
    # Display the results
    count = 0
    unknown_idx = []
    for (top, right, bottom, left), name in zip(face_locations,face_names):
        if name == 'Unknown':
            recognition.draw_rectangles(((top, right, bottom, left), name), (0,0,255))
            unknown_idx.append(count)
        else:
            recognition.draw_rectangles(((top, right, bottom, left), name), (0,255,0))
        count += 1
    
    # print('Identify people')
    # for (top, right, bottom, left), name in zip(face_locations,face_names):
    #     recognition.draw_rectangles(((top, right, bottom, left), name), (0,0,255))

    # Display the resulting image
    cv2.imshow('Video', recognition.frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
