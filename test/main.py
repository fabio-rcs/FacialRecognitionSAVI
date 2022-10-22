#!/usr/bin/env python3

from copy import deepcopy
import cv2
from functions2 import Detection, Tracker
from detector import Detector

def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    cap_device = 0
    cap_width = 1260    #640
    cap_height = 900    #360
    
    model_path = 'model/model.tflite'
    input_shape=(192, 192)
    score_th = 0.4
    nms_th = 0.5
    num_threads = None
        
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height) 
   
    detection_counter = 0
    tracker_counter = 0
    trackers = []
    iou_threshold = 0.7
    frame_counter = 0

    detector = Detector(model_path=model_path, input_shape=input_shape, score_th=score_th,
        nms_th=nms_th,providers=['CPUExecutionProvider'], num_threads=num_threads)

    # ------------------------------------------
    # Execution
    # ------------------------------------------
    while (cap.isOpened()):

        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        image_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # convert video to grayscale
        image_gui = deepcopy(frame)
        stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
        
        # ------------------------------------------
        # Detection of persons 
        # ------------------------------------------
        bboxes, _, _ = detector.inference(frame)
        
        # ------------------------------------------
        # Create Detections per haar cascade bbox
        # ------------------------------------------
        detections = []
        for bbox in bboxes: 
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            detection = Detection(x1, y1, x2, y2, image_gray, id=detection_counter, stamp=stamp)
            detection_counter += 1
            detection.draw(image_gui)
            detections.append(detection)

        # ------------------------------------------
        # For each detection, see if there is a tracker to which it should be associated
        # ------------------------------------------
        for detection in detections: # cycle all detections
            for tracker in trackers: # cycle all trackers
                tracker_bbox = tracker.detections[-1]
                iou = detection.computeIOU(tracker_bbox)
                if iou > iou_threshold: # associate detection with tracker 
                    tracker.addDetection(detection, image_gray)
        
        # ------------------------------------------
        # Track using template matching
        # ------------------------------------------
        for tracker in trackers: # cycle all trackers
            last_detection_id = tracker.detections[-1].id
            detection_ids = [d.id for d in detections]
            if not last_detection_id in detection_ids:
                tracker.track(image_gray)
        
        # ------------------------------------------
        # Deactivate Tracker if no detection for more than T
        # ------------------------------------------
        for tracker in trackers: # cycle all trackers
            tracker.updateTime(stamp)

        # ------------------------------------------
        # Create Tracker for each detection
        # ------------------------------------------
        for detection in detections:
            if not detection.assigned_to_tracker:
                tracker = Tracker(detection, id=tracker_counter, image=image_gray)
                tracker_counter += 1
                trackers.append(tracker)

        # ------------------------------------------
        # Draw stuff
        # ------------------------------------------

        # Draw a rectangle around the upper bodies
        for tracker in trackers:
            tracker.draw(image_gui)
        
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