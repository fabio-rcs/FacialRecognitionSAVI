#!/usr/bin/env python3

from copy import deepcopy
import cv2
import imutils
from functions import Detection, Tracker

def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    haar_upper_body_cascade = cv2.CascadeClassifier("./haarcascade_upperbody.xml")
    cap = cv2.VideoCapture(0)

    detection_counter = 0
    tracker_counter = 0
    trackers = []
    iou_threshold = 0.7
    frame_counter = 0

    # ------------------------------------------
    # Execution
    # ------------------------------------------
    while (cap.isOpened()):
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        frame = imutils.resize(frame, width=1000) # resize original video for better viewing performance
        image_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # convert video to grayscale
        frame_gui = deepcopy(frame)
        stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
        
        # ------------------------------------------
        # Detection of persons 
        # ------------------------------------------
        bboxes = haar_upper_body_cascade.detectMultiScale(
            image_gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (100, 100), # Min size for valid detection, changes according to video size or body size in the video.
            flags = cv2.CASCADE_SCALE_IMAGE)


        # ------------------------------------------
        # Create Detections per haar cascade bbox
        # ------------------------------------------
        detections = []
        for bbox in bboxes: 
            x1, y1, w, h = bbox
            detection = Detection(x1, y1, w, h, image_gray, id=detection_counter, stamp=stamp)
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

if __name__ == "__main__":
  main()