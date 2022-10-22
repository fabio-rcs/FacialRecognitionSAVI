#!/usr/bin/env python3

import cv2
from cv2 import imwrite


def main():

    # ---------------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------------

    # Object Detection in Real-time
    # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

    #Capture Video from Camera
    try:
        cap = cv2.VideoCapture(0)  
    except:
        print("Cannot open camera") 
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

        
        # Display the resulting frame
        cv2.imshow('video', img_RGB)

        if cv2.waitKey(1) == ord('q'):
            cv2.imwrite('Tatiana.png', img_RGB)
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