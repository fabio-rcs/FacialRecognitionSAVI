#!/usr/bin/env python3

from copy import deepcopy
import cv2
import imutils
import time
import numpy as np
from functions import Detection, Tracker
from detector import Detector


def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    cap_device = 0
    cap_width = 640
    cap_height = 360

    model_path = 'model/model.tflite'
    input_shape=(192, 192)
    score_th = 0.4
    nms_th = 0.5
    num_threads = None
        
    cap = cv2.VideoCapture(cap_device)
   
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
        start_time = time.time()
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        #frame = imutils.resize(frame, width=1000) # resize original video for better viewing performance
        image_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # convert video to grayscale
        frame_gui = deepcopy(frame)
        stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
        
        # ------------------------------------------
        # Detection of persons 
        # ------------------------------------------
        bboxes, scores, class_ids = detector.inference(frame)
        
        elapsed_time = time.time() - start_time
        frame_gui = draw_debug(frame_gui, elapsed_time, score_th, bboxes,
                scores, class_ids)
        # ------------------------------------------
        # Create Detections per haar cascade bbox
        # ------------------------------------------
        detections = []
        for bbox in bboxes: 
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            detection = Detection(x1, y1, x2, y2, image_gray, id=detection_counter, stamp=stamp)
            detection_counter += 1
            detection.draw(frame_gui)
            detections.append(detection)

        # ------------------------------------------
        # For each detection, see if there is a tracker to which it should be associated
        # ------------------------------------------
        for detection in detections: # cycle all detections
            for tracker in trackers: # cycle all trackers
                tracker_bbox = tracker.detections[-1]
                iou = detection.computeIOU(tracker_bbox)
                # print('IOU( T' + str(tracker.id) + ' D' + str(detection.id) + ' ) = ' + str(iou))
                if iou > iou_threshold: # associate detection with tracker 
                    tracker.addDetection(detection, image_gray)
        
        # ------------------------------------------
        # Track using template matching
        # ------------------------------------------
        for tracker in trackers: # cycle all trackers
            last_detection_id = tracker.detections[-1].id
            #print(last_detection_id)
            detection_ids = [d.id for d in detections]
            if not last_detection_id in detection_ids:
                #print('Tracker ' + str(tracker.id) + ' Doing some tracking')
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
            tracker.draw(frame_gui)
        
        cv2.imshow('Video', frame_gui) # Display video
        
        # stop script when "q" key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_counter += 1
    # Release capture
    cap.release()
    cv2.destroyAllWindows()

def draw_debug(image, elapsed_time, score_th, bboxes, scores, class_ids):
    frame_gui = deepcopy(image)

    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if score_th > score:
            continue

        #
        frame_gui = cv2.rectangle(frame_gui, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

        # 
        score = '%.2f' % score
        text = '%s:%s' % (str(int(class_id)), score)
        frame_gui = cv2.putText(frame_gui, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), thickness=2)
    return frame_gui

if __name__ == "__main__":
  main()