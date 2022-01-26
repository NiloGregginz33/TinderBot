import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Activation, MaxPool2D, Dropout, BatchNormalization
from numpy import asarray

import dlib
import openface

import sys
# sys.path.insert(0, 'PoseAI/tf_pose_estimation/')
sys.path.append('../')

import random

import pandas as pd
import numpy as np

import shutil

import pathlib
import pyautogui
from pathlib import Path
import PIL
import PIL.Image
import glob

import cv2
import os, os.path, time

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

for i in range(1):
# Captures screenshot
    myScreenshot = pyautogui.screenshot()
    myScreenshot.save(r'content/Screenshots/temp_shot.png')
  
# Opens screenshot in RGB mode 
    im = cv2.imread(r"content/Screenshots/temp_shot.png")

# Resizes screenshot so other faces are not likely detected
    h = 900
    w = 850
    y = 0
    x = 650
    im1 = im[y:y+h, x:x+w]

# loads HOG face detector
#predictor_model = "shape_predictor_68_face_landmarks.dat"
#face_detector = dlib.get_frontal_face_detector()
#face_pose_predictor = dlib.shape_predictor(predictor_model)
#face_aligner = openface.AlignDlib(predictor_model)
#detected_faces = face_detector(im1, 1)
#saves the files again
#for i, face_rect in enumerate(detected_faces):
#   pose_landmarks = face_pose_predictor(im1, face_rect)
#   alignedFace = face_aligner.align(534, im1, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
#   cv2.imwrite("content/Faces/aligned{}.jpg".format(i), alignedFace)

# Loads face_cascade to detect faces
    face_cascade = cv2.CascadeClassifier('facedetection/haarcascade_frontalface_default.xml')

# Convert into grayscale and find faces
    gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces detected and saves the faces as png files
    padding = 10
    ROI_number = 0
    for (x, y, w, h) in faces: 
        h = h+padding
        w = w+padding
        ROI = im1[y:(y+h), x:(x+w)]
        cv2.imwrite('content/Faces/'+ str(ROI_number)+".png", ROI)
        cv2.rectangle(im1, (x-padding, y-padding), (x+w, y+h), (255, 0, 0), 2)
        ROI_number+=1

# Loads a pretrained model
    model=keras.models.load_model('my_model2.h5')

#preprocceses and predicts face files and prints output from pre-trained model
    new_path = Path('content/Faces/')
    files = os.listdir(new_path)
    for i in range(3):
        time.sleep(1.2)
        if len(files) == 0:
            pyautogui.click(1153, 391)
            myScreenshot = pyautogui.screenshot()
            myScreenshot.save(r'content/Screenshots/temp_shot.png')
            gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
  
# Opens screenshot in RGB mode 
            im = cv2.imread(r"content/Screenshots/temp_shot.png")

# Resizes screenshot so other faces are not likely detected
            h = 900
            w = 850
            y = 0
            x = 650
            im1 = im[y:y+h, x:x+w]
            padding = 24
            ROI_number = 0
            for (x, y, w, h) in faces: 
                h = h+padding
                w = w+padding
                ROI = im1[y:(y+h), x:(x+w)]
                cv2.imwrite('content/Faces/'+ str(ROI_number)+".png", ROI)
                cv2.rectangle(im1, (x-padding, y-padding), (x+w, y+h), (255, 0, 0), 2)
                ROI_number+=1
            new_path = Path('content/Faces/')
            files = os.listdir(new_path)
    if len(files)!=0:
        for i in range(len(files)):
            prediction_path = os.path.join(new_path, str(files[i]))
            img = keras.preprocessing.image.load_img(
            prediction_path, target_size=(128, 128)
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
  # Swipes if 'attractive' (np.argmax(score)/50) >= 6.0
            if np.argmax(score)/50 >= 6.5:
                pyautogui.click(1059,713)
                time.sleep(0.5)
 
            else:
                pyautogui.click(911,713)
                time.sleep(2)

            print(
                "This image most likely belongs to {} with a {:.2f} percent confidence"
                .format([np.argmax(score)/50], np.max(score))
              )
            cv2.imshow("face", i)
            cv2.waitKey(1)
    else:
        pyautogui.click(911,713)
# Deletes all recent files
    files = glob.glob('content/Screenshots/*.png')
    for f in files:
        os.remove(f)
    new_files = glob.glob('content/Faces/*.png')
    for f in new_files:
        os.remove(f)
    newer_files = glob.glob('content/Faces/*.jpg')
    for f in newer_files:
        os.remove(f)
    time.sleep(1)
