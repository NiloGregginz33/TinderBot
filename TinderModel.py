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
from sklearn import preprocessing

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop("Rank")
  labe
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())
  
batch_size = 16

dataframe_names = ["Nose x", "Nose y", "Nose z","Left eye inner x","Left eye inner y","Left eye inner z"
  ,"Left eye x","Left eye y","Left eye z","Left eye outer x","Left eye outer y","Left eye outer z","Right eye inner x","Right eye inner y","Right eye inner z",
  "Right eye x","Right eye y","Right eye z","Right eye outer x","Right eye outer y","Right eye outer z","Left ear x","Left ear y","Left ear z"
  ,"Right ear x","Right ear y","Right ear z","Mouth left x","Mouth left y","Mouth left z","Mouth right x","Mouth right y","Mouth right z"
  ,"Left shoulder x","Left shoulder y","Left shoulder z","Right shoulder x","Right shoulder y","Right shoulder z","Left elbow x","Left elbow y","Left elbow z"
  ,"Right elbow x","Right elbow y","Right elbow z","Left wrist x","Left wrist y","Left wrist z","Right wrist x","Right wrist y","Right wrist z"
  ,"Left pinky x","Left pinky y","Left pinky z","Right pinky x","Right pinky y","Right pinky z","Left index x","Left index y","Left index z",
  "Right index x","Right index y","Right index z","Left thumb x","Left thumb y","Left thumb z", "Right thumb x", "Right thumb y", "Right thumb z",
  "Left hip x", "Left hip y", "Left hip z","Right hip x","Right hip y","Right hip z","Left knee x","Left knee y","Left knee z","Right knee x","Right knee y","Right knee z"
  ,"Left ankle x","Left ankle y","Left ankle z","Right ankle x","Right ankle y","Right ankle z","Left heel x","Left heel y","Left heel z"
  ,"Right heel x","Right heel y","Right heel z","Left foot index x","Left foot index y","Left foot index z","Right foot index x","Right foot index y","Right foot index z"]

dataframe = pd.read_csv("skeletons.csv", )

##for i in len(dataframe_names):
##    dataframe[dataframe_names[i]] = pd.to_numeric(database(dataframe[i]))

dataframe = dataframe.apply(pd.to_numeric, errors='coerce')

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

feature_columns = []

for header in ["Rank","Nose x", "Nose y", "Nose z","Left eye inner x","Left eye inner y","Left eye inner z"
  ,"Left eye x","Left eye y","Left eye z","Left eye outer x","Left eye outer y","Left eye outer z","Right eye inner x","Right eye inner y","Right eye inner z",
  "Right eye x","Right eye y","Right eye z","Right eye outer x","Right eye outer y","Right eye outer z","Left ear x","Left ear y","Left ear z"
  ,"Right ear x","Right ear y","Right ear z","Mouth left x","Mouth left y","Mouth left z","Mouth right x","Mouth right y","Mouth right z"
  ,"Left shoulder x","Left shoulder y","Left shoulder z","Right shoulder x","Right shoulder y","Right shoulder z","Left elbow x","Left elbow y","Left elbow z"
  ,"Right elbow x","Right elbow y","Right elbow z","Left wrist x","Left wrist y","Left wrist z","Right wrist x","Right wrist y","Right wrist z"
  ,"Left pinky x","Left pinky y","Left pinky z","Right pinky x","Right pinky y","Right pinky z","Left index x","Left index y","Left index z",
  "Right index x","Right index y","Right index z","Left thumb x","Left thumb y","Left thumb z", "Right thumb x", "Right thumb y", "Right thumb z",
  "Left hip x", "Left hip y", "Left hip z","Right hip x","Right hip y","Right hip z","Left knee x","Left knee y","Left knee z","Right knee x","Right knee y","Right knee z"
  ,"Left ankle x","Left ankle y","Left ankle z","Right ankle x","Right ankle y","Right ankle z","Left heel x","Left heel y","Left heel z"
  ,"Right heel x","Right heel y","Right heel z","Left foot index x","Left foot index y","Left foot index z","Right foot index x","Right foot index y","Right foot index z"]:
    
    
    feature_columns.append(feature_column.numeric_column(header))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of Nose points:', feature_batch['Nose x'], feature_batch['Nose y'])
  print('A batch of targets:', label_batch )

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  layers.Dense(64, activation = 'relu'),
  layers.Dense(128, activation = 'relu'),
  layers.Dropout(.1),
  layers.Dense(128),
  layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

checkpoint_path = "training_1/cp.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


model.fit(train_ds, validation_data = val_ds, epochs=15, callbacks=[cp_callback])


predictions = model.predict(test_ds.take(1))

model.save("mp_kp_tinder_model.h5")

print(predictions)

