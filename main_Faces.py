#!/usr/bin/env python3

from copy import deepcopy
import cv2

def main():

    # ---------------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------------

    # Object Detection in Real-time
    # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    #Capture Video from Camera
    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():                 
        print("Cannot open camera") 
        exit()

    # ---------------------------------------------------------------------------------
    # Execution
    # ---------------------------------------------------------------------------------
    while True:
        # Capture frame-by-frame
        ret, img_RGB = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        img_gui = deepcopy(img_RGB)   # working image in RGB
        img_gray = cv2.cvtColor(img_gui, cv2.COLOR_BGR2GRAY)    #image in GRAY
        
        # Detection of faces
        bboxs = face_cascade.detectMultiScale(img_gray,scaleFactor=1.2, minNeighbors=5,minSize=(20, 20))
        
        # Create Detections per haar cascade bbox
        for bbox in bboxs:
            x, y, w, h = bbox 
            cv2.rectangle(img_gui,(x,y),(x+w,y+h),(255,0,0),2)

        # Display the resulting frame
        cv2.imshow('video', img_gui)

        if cv2.waitKey(1) == ord('q'):
            break
    # ---------------------------------------------------------------------------------
    # Termination
    # ---------------------------------------------------------------------------------
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

#Sees if function was called in the terminal
if __name__ == "__main__":
    main()