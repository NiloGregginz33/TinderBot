import tensorflow as tf
from tensorflow import keras
import requests
import requests.auth
import cv2
import os
import pynder
from bs4 import BeautifulSoup as BSHTML
import urllib3


xAuth = "enter x_auth here for tinder you can find it on tinders website by checking"

##ans = input("Do you have an existing session ID? [y/n]")
##
##if ans is "Y" or ans is "y":
##    xAuth = input("Paste your new x-auth token here: ")
##
##else:
##    xAuth = xAuth
##    print("Using XAuth: ", xAuth)
    
session = pynder.Session(XAuthToken = xAuth)

users = session.nearby_users()

model = keras.models.load_model("facial_attractiveness.h5")

count = 0

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))

body_model = os.path.join(cv2_base_dir, 'data/haarcascade_fullbody.xml')

face_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

eye_model = os.path.join(cv2_base_dir, 'data/haarcascade_eye.xml')

body_detector = cv2.CascadeClassifier(body_model)

face_detector = cv2.CascadeClassifier(face_model)

eye_detector = cv2.CascadeClassifier(eye_model)

for user in users:

    photos = user.photos

    name = user.name

    filename = "temp.jpg"

    for photo in photos:

        print(photo)

        with open(filename) as f:
            
            f.write(response.content)
            
        f.close()

        image = cv2.imread(filename)

        gray = cv2.cvtColor(COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray)

        if len(faces) > 0:

            for face in faces:

                score = 0

                face_count = 0

                for (x, y, w, h) in face:

                    face_count += 1

                    roi = image[y:y+h, x:x+w]
                    
                    cv2.resize(roi,(128,128))

                    roi_arr = keras.preprocessing.image.img_to_array(roi_arr)

                    roi_arr = tf.expand_dims(roi_arr, 0)

                    predictions = model.predict(roi_arr)

                    score = np.argmax(predictions)

            print(np.argmax(score)/(50*face_count))


        final = np.argmax(score)/(50*face_count)

        if final >= 6.5:
            user.like()

        else:
            user.dislike

        time.sleep(3)

    

           
                    

                    

        

        

    

    
