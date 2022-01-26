import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height = 350
img_width = 250
batch_size = 3

model = keras.Sequential([
    layers.input((350,250,3)),
    layers.Conv2D(16, 3, padding=same),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, 3, padding= same),
    layers.MaxPooling2D((2,2)),
    layers.Flatten()
    layers.Dense(10)
])

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
'train', labels = 'inferred', label_mode = "int", class_names=['1','2','3','4','5','6','7','8','9','10'],
batch_size = batch_size, image_size = (img_height, img_width), shuffle= True, seed=123, validation_split = 0.12, subset = "training'
    )

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
'train', labels = 'inferred', label_mode = "int", class_names=['1','2','3','4','5','6','7','8','9','10'],
batch_size = batch_size, image_size = (img_height, img_width), shuffle= True, seed=123, validation_split = 0.12, subset = "validation'
    )

def augment(x ,y):
    image = tf.image.random_brightness(x, max_delta = 0.05)
    return image, y

ds_train = ds_train.map(augment)

for epochs in range(10):
    for x, y in ds_train:
        pass

model.compile(
    optimizer=keras.optiizers.Adam()
    loss=[
        keras.losses.SparceCategoricalCrossentropy(from_logits  = True),], metrics = ["accuracy"]
model.fit(ds_train, epochs=10, verbose=2)
