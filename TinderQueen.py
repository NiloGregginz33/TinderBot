import tensorflow as tf
import numpy as np
import cv2
import glob
import mediapipe as mp
import os
import sys
import PIL
from PIL import Image
import time
from os import walk, path
import csv
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
import PIL
from PIL import Image




def parse_image(filename):
  parts = tf.strings.split(filename, os.sep)
  label = get_label(filename)
  image = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(image, channels = 3)
  image = tf.image.resize(image, [128, 128])
  image = tf.image.convert_image_dtype(image, tf.float32)/255 
  return image, label

def get_label(filename):
    parts = tf.strings.split(filename, os.path.sep)
    label = parts[-2]
    return float(label)

def process_path(file_path):
  image = tf.io.read_file(file_path)
  image = tf.image.decode_jpeg(image, channels = 3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [350, 650])
  return image

def get_rank(data):
    rank = data[0] 
    return rank

def tensor_map(file):
    
    file_path = os.path.abspath(str(file))

    rank = float(get_label(file_path))

    print(file_path)
    
    image = cv2.imread(file_path)

    flag = True

    data = [rank]
    

    with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:

        try:

            result = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            j = 0

            print("We're coool")

            print(len(result.multi_face_landmarks))

            for face_landmarks in result.multi_face_landmarks:

                for landmark in face_landmarks.landmark:
                    
                
                    print('face_landmarks:', landmark.x, landmark.y, landmark.z)

                    data.append(float(landmark.x))
                    data.append(float(landmark.y))
                    data.append(float(landmark.z))

                    j += 1

        except:
            for t in range(1305):
                data.append(0)
                
            print("face_malfunction")
            flag = False

    if flag:
        with open("skeleton.csv", "a+") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data)
                print(data)
                print("hello")
                    



def datasen(dataset):
    rank = dataset[0]
    return rank
    
def workaround(characters):
    ranks = []
    dataset = []
    
    for soldier in characters:
       file_path = soldier
       data = tensor_map(file_path)
       rank = data[0]
       ranks.append(rank)
       dataset.append(data)

       return dataset, ranks


def df_to_dataset(dataframe, shuffle=True, batch_size=16):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Rank')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

x = 0


df_names = ["Rank"]

for v in range(1305):
    string = "face_landmark_"
    string += str(v+1)
    string += "_"
    a=v%3
    array = ["x","y","z"]
    string += str(array[a])
    df_names.append(string)

df_length = len(df_names)

print(df_names)
image_height = 650
image_width = 350

training_flag = True

characters = list(glob.glob("full_images/train/train/*/*.jpg"))

print(df_names)

char = input("Continue Training? (y/n) : ")

if char != "N" and char != "n":
    training_flag = False
        
if os.path.exists("skeleton.csv"):
    print("CSV obtained")
##else:   
##     with open("skeleton.csv", "w+") as csvfile:
##        writer = csv.DictWriter(csvfile, fieldnames = df_names)
##        writer.writeheader()
##        
if training_flag == False:
    for i in range(len(characters)):
        try:
            tensor_map(characters[i])
            print("Success")

        except:
            print("Malfunction")
      
