#!/usr/bin/env python3
import cv2
from recognition import Recognition
import face_recognition
import pickle
import copy

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

    # Identify unknown people
    for i in unknown_idx:
        # Verify if is there any repeated face
        if True in face_recognition.compare_faces(known_face_encodings, face_encodings[i]):
            pass
        else:
            # Switch from unknown to question in the frame
            recognition.remove_name(original_frame, recognition.frame, face_locations[i], 'Unknown')
            recognition.draw_rectangles((face_locations[i], 'Who are you?'), (255,0,0))

            # Update image
            cv2.imshow('Video', recognition.frame)
            cv2.waitKey(1)

            # Get name
            name = input('Who are you?')

            # Switch from question to red painted name in the frame
            recognition.remove_name(original_frame, recognition.frame, face_locations[i], 'Who are you?')
            recognition.draw_rectangles((face_locations[i], name), (0,0,255))

            # Add info to encodings and names list
            recognition.known_face_encodings.append(face_encodings[i])
            recognition.known_face_names.append(name)
            
            # Update image
            cv2.imshow('Video', recognition.frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
