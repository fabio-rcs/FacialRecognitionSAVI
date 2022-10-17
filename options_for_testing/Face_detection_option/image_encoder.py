#!/usr/bin/env python3
import face_recognition
import cv2
import numpy as np
import os
import pickle

# Load images from database
dir_images = './Database/images'
dir_db = './Database/database_group.pickle'
image_names = os.listdir(dir_images)
names = []
encoded_faces = []

for image in image_names:
    loaded_image = face_recognition.load_image_file(dir_images + '/' + image)
    face_encoding = face_recognition.face_encodings(loaded_image)[0]
    name = image.split('.')[0]
    names.append(name)
    encoded_faces.append(face_encoding)
save = (names, encoded_faces)

with open(dir_db, 'wb') as f:
    pickle.dump(save, f)