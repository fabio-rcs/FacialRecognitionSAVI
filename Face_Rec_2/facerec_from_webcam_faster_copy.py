#!/usr/bin/env python3
from concurrent.futures import thread
import cv2
from recognition import Recognition
import pickle
import copy
import threading
import matplotlib.pyplot as plt

def teste(window_name, original_frame, analyzing_frame,unknown_idx, face_encodings, face_locations, state):
    recognition.identify_unknown(window_name,original_frame, recognition.frame, unknown_idx, face_encodings, face_locations, state)
    
# Database directories
    # Binary files
dir_db = './Database/database.pickle'
dir_db_backup = './Database/database_backup.pickle'
    # Image files
dir_image = './Database/images'
dir_image_backup = './Database/images_backup'

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load file with lists of names and faces encodings
with open(dir_db, 'rb') as f:
     known_face_names, known_face_encodings = pickle.load(f)

# Initialize some variables
state = True
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
    
    # Save the frame 
    original_frame = copy.deepcopy(frame)
    recognition = Recognition(frame, known_face_encodings, known_face_names)

    # Only process every other frame of video to save time
    if process_this_frame or cycle>=cycle_interval:
        process_this_frame = False
        cycle = 0

        # If any, detect and recognize faces in the frame
        try:
            face_locations, face_names, face_encodings = recognition.process_frame()
        except ValueError as e:
            print(e)

    # Display the results and get unknown people ids
    count = 0
    unknown_idx = []
    for (top, right, bottom, left), name in zip(face_locations,face_names):
        if name == 'Unknown':
            recognition.draw_rectangles(((top, right, bottom, left), name), (0,0,255))
            unknown_idx.append(count)
        else:
            recognition.draw_rectangles(((top, right, bottom, left), name), (0,255,0))
        count += 1
    cv2.imshow('Video', recognition.frame)
    # plt.subplot(2,1,1), plt.imshow(recognition.frame)
    # plt.show()
    if state:
        th = threading.Thread(target=teste, args=('Video2', original_frame, recognition.frame,unknown_idx, face_encodings, face_locations, state))
        th.start()
        #teste('Video', original_frame, recognition.frame,unknown_idx, face_encodings, face_locations, state)
        #recognition.identify_unknown('Video',original_frame, recognition.frame, unknown_idx, face_encodings, face_locations, state)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
