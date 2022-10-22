#!/usr/bin/env python3

import cv2

class BoundingBox:
    
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        
        self.x2 = x2
        self.y2 = y2

        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1
        self.area = self.w * self.h


    def computeIOU(self, bbox2):
    
        x1_intr = min(self.x1, bbox2.x1)             
        y1_intr = min(self.y1, bbox2.y1)             
        x2_intr = max(self.x2, bbox2.x2)
        y2_intr = max(self.y2, bbox2.y2)

        w_intr = x2_intr - x1_intr
        h_intr = y2_intr - y1_intr
        A_intr = w_intr * h_intr

        A_union = self.area + bbox2.area - A_intr
        
        return A_intr / A_union

    def extractSmallImage(self, image_full):
        return image_full[self.y1:self.y2, self.x1:self.x2]


class Detection(BoundingBox):

    def __init__(self, x1, y1, x2, y2, image_full, id, stamp):
        super().__init__(x1,y1,x2,y2) # call the super class constructor        
        self.id = id
        self.stamp = stamp
        self.image =self.extractSmallImage(image_full)
        self.assigned_to_tracker = False

    def draw(self, image_gui, color=(255,0,0)):
        cv2.rectangle(image_gui,(self.x1,self.y1),(self.x2, self.y2),color,3)

        image = cv2.putText(image_gui, 'D' + str(self.id), (self.x1, self.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)

class Tracker():

    def __init__(self, detection, id, image):
        self.id = id
        self.template = None
        self.active = True
        self.bboxes = []
        self.detections = []
        self.tracker = cv2.TrackerMIL_create()
        self.time_since_last_detection = None

        self.addDetection(detection, image)

    def getLastDetectionStamp(self):
        return self.detections[-1].stamp

    def updateTime(self, stamp):
        self.time_since_last_detection = round(stamp-self.getLastDetectionStamp(),1)

        if self.time_since_last_detection > 8: # deactivate tracker        
            self.active = False

    def drawLastDetection(self, image_gui, color=(255,0,255)):
        last_detection = self.detections[-1] # get the last detection

        cv2.rectangle(image_gui,(last_detection.x1,last_detection.y1),
                      (last_detection.x2, last_detection.y2),color,3)

        image = cv2.putText(image_gui, 'T' + str(self.id), 
                            (last_detection.x2-40, last_detection.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)

    def draw(self, frame_gui, color=(255,0,255)):

        if not self.active:
            color = (100,100,100)
    
        bbox = self.bboxes[-1] # get last bbox

        # display the detected boxes in the color picture
        cv2.rectangle(frame_gui,(bbox.x1, bbox.y1), (bbox.x2, bbox.y2),color,3)

        cv2.putText(frame_gui, 'T' + str(self.id), 
                            (bbox.x2-40, bbox.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)

        cv2.putText(frame_gui, str(self.time_since_last_detection) + ' s', 
                            (bbox.x2-40, bbox.y1-25), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)

    def addDetection(self, detection, image):  
        
        r,c = image.shape
        if detection.x1 <= 0:
            detection.x1 = 1
        if detection.y2 >= r:
            detection.y2 = r-1
        if detection.y1 <= 0:
            detection.y1 = 1
        if (detection.x1 + detection.w) >= c:
            detection.w = c - detection.x1 - 5
        if (detection.y1 + detection.h) >= r:
            detection.h = r - detection.y1 - 5

        self.tracker.init(image, (detection.x1, detection.y1, detection.w, detection.h))

        self.detections.append(detection)
        detection.assigned_to_tracker = True
        self.template = detection.image
        bbox = BoundingBox(detection.x1, detection.y1, detection.x2, detection.y2)
        self.bboxes.append(bbox)

    def track(self, image):

        ret, bbox = self.tracker.update(image)
        x1,y1,w1,h1 = bbox

        x2 = x1 + w1
        y2 = y1 + h1
        bbox = BoundingBox(x1, y1, x2, y2)
        self.bboxes.append(bbox)

        # Update template using new bbox coordinates
        self.template = bbox.extractSmallImage(image)
        
    def __str__(self):
        text =  'T' + str(self.id) + ' Detections = ['
        for detection in self.detections:
            text += str(detection.id) + ', '

        return text