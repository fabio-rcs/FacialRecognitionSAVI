#!/usr/bin/env python3

from copy import deepcopy
import cv2
from functions2 import Detection, Tracker
from detector import Detector
from collections import defaultdict

def main():

    # ------------------------------------------
    #! Initialization
    # ------------------------------------------
    # Size of the window
    cap_width = 1260
    cap_height = 900

    # Variables for the body detector
    model_path = 'model/model.tflite'
    input_shape=(192, 192)
    score_th = 0.4
    nms_th = 0.5
    num_threads = None
    
    # Opens computer camera
    cap = cv2.VideoCapture(0)
    # Prints error message if getting fails
    if (cap.isOpened() == False):
        print('Error opening the video') 
    # Sets the size of the window
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height) 
   
    # Initialization of variables
    detection_counter = 0
    tracker_counter = 0
    trackers = []
    iou_threshold = 0.7 # Threshold for the overlap of bboxes
    frame_counter = 0

    # Definition of body detector
    detector = Detector(model_path=model_path, input_shape=input_shape, score_th=score_th,
        nms_th=nms_th,providers=['CPUExecutionProvider'], num_threads=num_threads)

    # Variables for follow-up of person
    points_dict = defaultdict (list) # We use the default model for simplification
    points_list = [] # Point storage list

    # ------------------------------------------
    #! Execution
    # ------------------------------------------
    while (cap.isOpened()):
        
        ret, frame = cap.read() # Gets frame

        # If frame is read correctly ret is True
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

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
            tracker.draw(image_gui) # Draws tracker bbox
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
        
        cv2.imshow('Video', image_gui) # Display video
        
        # stop script when "q" key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_counter += 1
        
    # Release capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  main()