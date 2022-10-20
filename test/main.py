#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import copy
import time
import argparse

from detector import Detector


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=360)

    parser.add_argument(
        "--model",
        type=str,
        default='model/model.tflite',
    )
    parser.add_argument(
        '--input_shape',
        type=str,
        default="192,192",
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.4,
    )   
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.5,
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=None,
        help='Valid only when using Tensorflow-Lite',
    )

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.movie is not None:
        cap_device = args.movie

    model_path = args.model
    input_shape = tuple(map(int, args.input_shape.split(',')))
    score_th = args.score_th
    nms_th = args.nms_th
    num_threads = args.num_threads

    # ###############################################################
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

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
        debug_image = copy.deepcopy(frame)

        # ########################################################
        bboxes, scores, class_ids = detector.inference(frame)

        elapsed_time = time.time() - start_time

        # デバッグ描画
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            score_th,
            bboxes,
            scores,
            class_ids,
        )

        # ##############################################
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        # #########################################################
        debug_image = cv2.resize(debug_image, (cap_width, cap_height))
        cv2.imshow('Person Detection Demo', debug_image)

    cap.release()
    cv2.destroyAllWindows()


def draw_debug(
    image,
    elapsed_time,
    score_th,
    bboxes,
    scores,
    class_ids,
):
    debug_image = copy.deepcopy(image)

    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if score_th > score:
            continue

        #
        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            thickness=2,
        )

        # 
        score = '%.2f' % score
        text = '%s:%s' % (str(int(class_id)), score)
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            thickness=2,
        )
    return debug_image

if __name__ == '__main__':
    main()