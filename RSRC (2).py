#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Activation, MaxPool2D, Dropout, BatchNormalization
from numpy import asarray

import tensorflow_datasets as tfds

import sys

import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import shutil

import pathlib
from pathlib import Path
import PIL
import PIL.Image
import glob
import os, os.path, time


# In[2]:


print(tf.__version__)


# In[3]:


batch_size = 32
img_height = 128
img_width = 128


# In[4]:



path = pathlib.Path('ClassDataset')

data_dir = list(path.glob('100/*.jpg'))

PIL.Image.open(str(data_dir[1]))


# In[5]:


def show(image):
  plt.figure()
  plt.imshow(image)
  plt.axis('off')


# In[6]:


img_ds = tf.data.Dataset.list_files(str(path/'*/*.jpg'), shuffle=False)
img_ds = img_ds.shuffle(5500, reshuffle_each_iteration=False)

for f in img_ds.take(5):
  print(f.numpy())


# In[7]:


classes = np.array(sorted([item.name for item in path.glob('*')]))
print(classes)


# In[8]:


get_ipython().run_line_magic('autosave', '1000')


# In[9]:


new_path = Path('Rankset')


# In[10]:


image_ds = tf.data.Dataset.list_files(str(new_path/'*/*.jpg'))


# In[11]:


rank = np.array(sorted([item.name for item in new_path.glob('*[0-9]')]))
print(rank)


# In[12]:


val_size = int(5500 * 0.2)
train_ds = image_ds.skip(val_size)
val_ds = image_ds.take(val_size)


# In[13]:


print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())


# In[14]:


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    label = parts[-2]
    return int(label)


# In[15]:


def get_labels(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    label = parts[-2]
    return label


# In[16]:


def decode_img(file):
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.cast(img, tf.float32)
  return tf.image.resize(img, [img_height, img_width])


# In[17]:


def process_path(file_path):
  parts = tf.strings.split(filename, os.sep)
  image = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(image, channels = 3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [128, 128])
  return image


# In[18]:


def parse_image(filename):
  parts = tf.strings.split(filename, os.sep)
  label = get_label(filename)
  image = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(image, channels = 3)
  image = tf.image.resize(image, [128, 128])
  image = tf.image.convert_image_dtype(image, tf.float32)/255 
  return image, label


# In[19]:


file_path = next(iter(image_ds))
image, label = parse_image(file_path)

def show(image, label):
  plt.figure()
  plt.imshow(image)
  plt.title(label)
  plt.axis('off')

show(image, label)


# In[20]:


image_ds = image_ds.map(parse_image)


# In[ ]:





# In[21]:


for image, label in image_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy().shape)


# In[22]:


val_size = int(5500 * 0.2)
training_ds = image_ds.skip(val_size)
valid_ds = image_ds.take(val_size)
batch_train = training_ds.batch(32, drop_remainder = True)
batch_valid = valid_ds.batch(32, drop_remainder = True)


# In[23]:


for image, label in training_ds:
    label = label/25
    pixels = asarray(image)
    pixels = pixels.astype('float32')
    pixels /= 255.0
    image = image*(1.0/255)


# In[24]:


for image, label in valid_ds:
    pixels = asarray(image)
    pixels = pixels.astype('float32')
    pixels /= 255.0
    image = image*(1.0/255)
    label = label/25


# In[25]:


AUTOTUNE = tf.data.experimental.AUTOTUNE

training_ds = training_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)


# In[26]:


for image, label in training_ds.batch(32):
  print(image.shape)
  print(label.shape)
  break


# In[27]:


num_classes = 500
model = Sequential([
  layers.Conv2D(32, 3, padding='same', activation='relu', input_shape =(128,128,3)),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(256, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(512, activation='relu'),
  layers.Dense(num_classes)
])


# In[28]:


model.compile(optimizer = "adam" , loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])


# In[29]:


def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(32)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds


# In[30]:


batch_size = 32
training_ds = configure_for_performance(training_ds)
valid_ds = configure_for_performance(valid_ds)


# In[31]:


model.fit(
  training_ds,
  epochs=13
)


# In[32]:


model.fit(
  training_ds,
  epochs=2
)


# In[33]:


model.summary()


# In[47]:


prediction_path = Path('shanaya.jpg')

img = keras.preprocessing.image.load_img(
    prediction_path, target_size=(128, 128)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format([np.argmax(score)], 100 * np.max(score))
)


# In[37]:




predictions = model.predict(img_array)


# In[ ]:




