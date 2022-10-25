#!/usr/bin/env python3

import cv2
from recognition import Recognition
from copy import deepcopy
from initialization import Initialization
from detector import Detector
from functions2 import Detection, Tracker, BoundingBox
from collections import defaultdict
from multiprocessing import Process
import pyttsx3
    
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


# Size of the window
# cap_width = 1260
cap_width = 700
cap_height = 700


# Variables for the body detector
model_path = 'model/model.tflite'
input_shape=(192, 192)
score_th = 0.4
nms_th = 0.5
num_threads = None

num_atual_img = 0

def main():
    # -----------------------------------------
    # Initialization
    # -----------------------------------------

    # Get a reference to webcam #0 (the default one)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height) 

    # # Start Initialization Class
    init=Initialization(dir_db,dir_db_backup,dir_image,dir_image_backup)
    # # Open the app
    DB_Orig, DB_RealT, DB_Reset = init.app()
    init.select_diretory()
    known_face_names, known_face_encodings = init.open_database()
    def task():
        # view database
        init.view_database(num_atual_img, known_face_names)
    
    if __name__ == '__main__':
        # create a process
        process = Process(target=task)
        process.start()

    # Initialize some variables for face recognition
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    cycle_interval = 2 
    cycle = 0

    # Initialize some variables for body detection
    detection_counter = 0
    tracker_counter = 0
    trackers = []
    iou_threshold = 0.7 # Threshold for the overlap of bboxes
    frame_counter = 0
    names = [] 
    tracker_name = None

    # Definition of body detector
    detector = Detector(model_path=model_path, input_shape=input_shape, score_th=score_th,
        nms_th=nms_th,providers=['CPUExecutionProvider'], num_threads=num_threads)

    # Variables for follow-up of person
    points_dict = defaultdict (list) # We use the default model for simplification
    points_list = [] # Point storage list

    engine = pyttsx3.init() # object creation
    # -----------------------------------------
    # Execution
    # -----------------------------------------

    while True:
        cycle += 1

        # Grab a single frame of video
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame. Exiting ...")
            break
        
        # Save the frame 
        original_frame = deepcopy(frame)
        
        
        image_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # convert video to grayscale
        image_gui = deepcopy(frame) # Copy to work with
        stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000 # Frame time in seconds
        
        # ------------------------------------------
        #? Detection of persons 
        # ------------------------------------------
        bboxes, _, _ = detector.inference(frame)
        
        # ------------------------------------------
        #? Create Detections per detection bbox
        # ------------------------------------------
        detections = []
        
        # Creation, drawing and register of detection
        for bbox in bboxes: 
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            detection = Detection(x1, y1, x2, y2, image_gray, id=detection_counter, stamp=stamp)
            detection_counter += 1
            detection.draw(image_gui)
            detections.append(detection)
           
        # ------------------------------------------
        #? For each detection, see if there is a tracker to which it should be associated
        # ------------------------------------------
        for detection in detections: # Cycle all detections
            for tracker in trackers: # Cycle all trackers
                tracker_bbox = tracker.detections[-1] # Gets last detection
                iou = detection.computeIOU(tracker_bbox) # Intersection over union
                if iou > iou_threshold: # Associate detection with tracker 
                    tracker.addDetection(detection, image_gray) 
        
        # ------------------------------------------
        #? Track using template matching
        # ------------------------------------------
        for tracker in trackers: # cycle all trackers
            last_detection_id = tracker.detections[-1].id # Gets tracker id
            detection_ids = [d.id for d in detections] 
            if not last_detection_id in detection_ids:
                tracker.track(image_gray)
        
        # ------------------------------------------
        #? Deactivate Tracker if no detection for more than a certain amount of time
        # ------------------------------------------
        for tracker in trackers: # cycle all trackers
            tracker.updateTime(stamp) 

        # ------------------------------------------
        #? Create Tracker for each detection
        # ------------------------------------------
        for detection in detections:
            if not detection.assigned_to_tracker:
                tracker = Tracker(detection, id=tracker_counter, image=image_gray)
                tracker_counter += 1
                trackers.append(tracker)
                
        # ------------------------------------------
        #? Draw stuff
        # ------------------------------------------

        # Draw a rectangle around the upper bodies
        for tracker in trackers:
            tracker.draw(image_gui,tracker_name) # Draws tracker bbox
            id = tracker.id 
            follow_box = tracker.follow() # Follow up

            cv2.circle(image_gui, follow_box, 4, (255,0,255), -1) # Draws the tracking point. Thickness -1 so that the circle is filled
            points_dict[id].append(follow_box) # Append the coordinates associated to a person

            if id not in points_list:# Checks if the point is already in list, or else the drawing of line will give error
                points_list.append(id) 
                continue

            else: #Draws a line between two points for each person
                length = len(points_dict[id]) # We iterate for each person 
                
                for pt in range(length): # Iterate for every point
                        
                    if not pt + 1 == length: # If not the last point draws line
                        start_point = (points_dict[id][pt][0], points_dict[id][pt][1]) 
                        end_point = (points_dict[id][pt+1][0], points_dict[id][pt+1][1])
                        # Pass if distance between points is too big
                        if abs(int(start_point[1])-int(end_point [1])) > 70:
                            pass
                        elif abs(int(start_point[0])-int(end_point [0])) > 70:
                            pass
                        else: # Draws line 
                            cv2.line(image_gui, start_point, end_point, (255,0,255), 2)
        
        frame_counter += 1
        
        #FACE RECOGNITION
        recognition = Recognition(image_gui, known_face_encodings, known_face_names)
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
            bbox = BoundingBox(left*4,top*4,right*4,bottom*4)
            for tracker in trackers:
                tracker_bbox = tracker.bboxes[-1]
                iou = bbox.computeIOU(tracker_bbox)
                #print('IOU( T' + str(tracker.id) + ' D' + str(name) + ' ) = ' + str(iou))
                if iou > iou_threshold: # Associate detection with tracker 
                    tracker_name = name

            if name == 'Unknown':
                recognition.draw_rectangles(((top, right, bottom, left), name), (0,0,255))
                unknown_idx.append(count)
            
            else:
                if name in names:
                    recognition.draw_rectangles(((top, right, bottom, left), name), (0,255,0))
                    
                else:
                    recognition.draw_rectangles(((top, right, bottom, left), name), (0,255,0))
                    engine.setProperty('rate', 125)     # setting up new voice rate
                    engine.setProperty('volume',2.0)    # setting up volume level  between 0 and 1
                    engine.say("Hello" + name)
                    names.append(name)
                    engine.runAndWait()
                    engine.stop()

            count += 1
        cv2.imshow('Video', recognition.frame)

        # Identify unknown people
        recognition.identify_unknown('Video',original_frame, recognition.frame, unknown_idx, face_encodings, face_locations, dir_image)
        init.save_database(recognition.known_face_names, recognition.known_face_encodings)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # -----------------------------------------
    # Finalization
    # -----------------------------------------

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  main()