#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import copy
import time

from detector import Detector

#
cap_device = 0
cap_width = 640
cap_height = 360

model_path = 'model/model.tflite'
input_shape=(192, 192)
score_th = 0.4
nms_th = 0.5
num_threads = None

def main():

    # ###############################################################
    cap = cv2.VideoCapture(cap_device)
   
    # #############################################################
    detector = Detector(
        model_path=model_path,
        input_shape=input_shape,
        score_th=score_th,
        nms_th=nms_th,
        providers=['CPUExecutionProvider'],
        num_threads=num_threads,
    )

    while True:
        start_time = time.time()

        # ################################################
        ret, frame = cap.read()
        if not ret:
            break
        img_gui = copy.deepcopy(frame)

        # ########################################################
        bboxes, scores, class_ids = detector.inference(frame)

        elapsed_time = time.time() - start_time

        # 
        img_gui = draw_debug(img_gui, elapsed_time, score_th, bboxes,
                scores, class_ids)

        # ##############################################
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        # #########################################################
        cv2.imshow('Person Detection Demo', img_gui)

    cap.release()
    cv2.destroyAllWindows()


def draw_debug(image, elapsed_time, score_th, bboxes, scores, class_ids):
    img_gui = copy.deepcopy(image)

    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if score_th > score:
            continue

        #
        img_gui = cv2.rectangle(img_gui, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

        # 
        score = '%.2f' % score
        text = '%s:%s' % (str(int(class_id)), score)
        img_gui = cv2.putText(img_gui, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), thickness=2)
    return img_gui

if __name__ == '__main__':
    main()