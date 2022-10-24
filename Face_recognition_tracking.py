#!/usr/bin/env python3

import cv2
from recognition import Recognition
import pickle
import copy
from initialization import Initialization

# -----------------------------------------
# Parameters
# -----------------------------------------

# Database directories
    # Binary files
dir_db = './Database/database.pickle'
dir_db_backup = './Database/database_backup.pickle'
    # Image files
dir_image = './Database/images'
dir_image_backup = './Database/images_backup'

def main():
    # -----------------------------------------
    # Initialization
    # -----------------------------------------

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    # Start Initialization Class
    init=Initialization()
    # Open the app
    DB_Orig, DB_RealT, DB_Reset = init.app()
    # Database view
    init.view_database(dir_image, dir_image_backup)


    # Load file with lists of names and faces encodings
    # with open(dir_db, 'rb') as f:
    #     known_face_names, known_face_encodings = pickle.load(f)

    # Initialize some variables for face recognition
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    cycle_interval =2 
    cycle = 0

    # -----------------------------------------
    # Execution
    # -----------------------------------------

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
        recognition.identify_unknown('Video',original_frame, recognition.frame, unknown_idx, face_encodings, face_locations)
        
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # -----------------------------------------
    # Finalization
    # -----------------------------------------

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  main()