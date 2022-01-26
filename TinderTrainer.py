import tensorflow as tf
import tensorflow_datasets as tfds
import os
import path
import mediapipe as mp

##########
directory_url_train = 'C:\\Users\\manav\\Desktop\\full_images\\train'
file_names_train = os.listdir('C:\\Users\\manav\\Desktop\\full_images\\train'
data_dir_train = tf.keras.utils.get_file("train", origin-dataset)
##directory_url_test = 'C:\\Users\\manav\\Desktop\\full_images\\test'
##file_names_test = os.listdir('C:\\Users\\manav\\Desktop\\full_images\\test'
##data_dir_train = tf.keras.utils.get_file("test", origin-dataset)
                             #######


##data_dir =       

batch_width = 32
img_height = 180
img_width = 128

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
full_model = tf.keras.Sequential()


full_model.add(data_augmentation)
full_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(8,650,350,3)))
full_model.add(layers.MaxPooling2D((2, 2)))
full_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
full_model.add(layers.MaxPooling2D((2, 2)))
full_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
full_model.add(layers.Flatten())
full_model.add(layers.Dense(64, activation='relu'))
full_model.add(layers.Dense(10))

full_model.compile(optimizer='adam',
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])

full_history = face_model.fit(full_train_ds, epochs=30, 
                validation_data=full_val_ds, batch_size=8)

results = full_model.evaluate(full_test_ds, batch_size=8)

class_names = train_ds.class_names
print(class_names)

                             
full_model.save("full_model.h5")
