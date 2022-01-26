import datetime
import requests
from geopy.geocoders import Nominatim
import cv2
import time
import os, os.path
#import tensorflow as tf
from random import randint
import time

#This is a tinder bot after all

TINDER_URL = "https://api.gotinder.com"
geolocator = Nominatim(user_agent="auto-tinder")
PROF_FILE = "unsorted/profiles.txt"

#I absolutely love haarcascades so thats why I'm using them they're built into opencv

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

#just in case if anyone wants to use it, this will create all the folders they need to create image labels

def foldermaker(folder):
    os.chdir(folder)
    for i in range(1, 11):
        os.mkdir(str(i))
    os.chdir('..')
    
def folderspawner():

    if not os.path.exists(PROF_FILE):
        with open(PROF_FILE, 'r+') as f:
            f.close()

    if not os.path.isdir("full_images"):
        os.mkdir("full_images")
        os.chdir("full_images")
        os.mkdir("test")
        foldermaker("test")
        os.mkdir("train")
        foldermaker("train")
        os.chdir("..")
    if not os.path.isdir("faces"):
        os.mkdir("faces")
        os.chdir("faces")
        os.mkdir("test")
        foldermaker("test")
        os.mkdir("train")
        foldermaker("train")
        os.chdir("..")
    

#freshens up the data
        
def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

#we use a class here because someone online had already created a more primitive Tinder controller so I figured I would modify that to train my ML models

class Person(object):

    def __init__(self, data, api):
        self._api = api

        self.id = data["_id"]
        self.name = data.get("name", "Unknown")

        self.bio = data.get("bio", "")
        self.distance = data.get("distance_mi", 0) / 1.60934

        self.birth_date = datetime.datetime.strptime(data["birth_date"], '%Y-%m-%dT%H:%M:%S.%fZ') if data.get(
            "birth_date", False) else None
        self.gender = ["Male", "Female", "Unknown"][data.get("gender", 2)]

        self.images = list(map(lambda photo: photo["url"], data.get("photos", [])))

        self.jobs = list(
            map(lambda job: {"title": job.get("title", {}).get("name"), "company": job.get("company", {}).get("name")}, data.get("jobs", [])))
        self.schools = list(map(lambda school: school["name"], data.get("schools", [])))

        if data.get("pos", False):
            self.location = geolocator.reverse(f'{data["pos"]["lat"]}, {data["pos"]["lon"]}')


    def __repr__(self):
        return f"{self.id}  -  {self.name} ({self.birth_date.strftime('%d.%m.%Y')})"


    def like(self):
        return self._api.like(self.id)

    def dislike(self):
        return self._api.dislike(self.id)

    #This is one of the only functions I had to modify to run the haarcascades and eventually OpenPose in the future to see if that improves my bot
    
    def download_images(self, folder=".", sleep_max_for=0):
        with open(PROF_FILE, "r") as f:
            lines = f.readlines()
        if self.id in lines:
            return
        with open(PROF_FILE, "a") as f:
            f.write(self.id+"\r\n")
            index = -1
        for image_url in self.images:
            index += 1
            req = requests.get(image_url, stream=True)
            if req.status_code == 200:
                os.chdir("unsorted")
                with open("{self.id}_{self.name}_{index}.jpeg", "wb+") as f:
                    f.write(req.content)
                    f.close()
                
                img = cv2.imread("{self.id}_{self.name}_{index}.jpeg")
                count = 1
                os.chdir("..")

                #it only works in gray scale because that's how a lot of the threshholding algorithms work I believe
                #we're just changing the size to be more human while also being a power of 2 since that will make some math easier later
                img = cv2.flip(img, 1)
                img = cv2.resize(img, (258, 512), interpolation = cv2.INTER_CUBIC)
                gray_face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = faceCascade.detectMultiScale(
                    gray_face,
                    scaleFactor=1.3,
                    minNeighbors=3,
                    minSize=(15, 15)
                )

                #so now we have the coordinates of just the faces, and we can start performing classification tasks easier
                if len(faces) != 0:
                    flag = True
                for (x,y,w,h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    face_image = img[y:y+h, x:x+w]
                    face_image = cv2.resize(face_image, (64, 64), interpolation=cv2.INTER_CUBIC)
                    
                    face_image = cv2.imshow("On a scale from 1-10, what would you rate this person's face? (ESC to quit)", face_image)
                    
                    j = cv2.waitKey(0)
                    if j == 27:         # wait for ESC key to exit
                        cv2.destroyAllWindows()
                        
                    ranking = input("What would you rank this players face? [1-10] : ")
                    self
                    if (randint(12,22) % 5 > 3):
                        with open("faces/" + "test/" + str(ranking)+ "//"+ str(self.id)+"_"+str(self.name)+"_"+str(index)+"_face_"+str(count)+".jpeg", "wb+") as f:
                            f.write(face_image)
                            count += 1
                            f.close()
                    else:
                        with open("faces/" + "train/" + str(ranking)+ "//"+ str(self.id)+"_"+str(self.name)+"_"+str(index)+"_face_"+str(count)+".jpeg", "wb+") as f:
                            f.write(face_image)
                            count += 1
                            f.close()

                #just a small user interface, I want anyone to be able to run this program fairly easily

                #we're also classifying them seperately because I want to try some sort of instance segmentation with openpose to determine body type
                
                image = cv2.imshow("On a scale from 1-10, what would you rate this person as a whole? (ESC to quir)", img)
                j = cv2.waitKey(0)
                if j == 27:         # wait for ESC key to exit
                    cv2.destroyAllWindows()
                    
                ranking = input("What would you rank this player? [1-10] : ")

                #just a way to randomly classify my testing and training samples and I wanted to see how training could go if i periodically change whether data is for test or for train in between training sessions
                
                if (randint(12, 22) % 5 > 3):
                    with open("full_images/" + "test/" + str(ranking)+ "//"+ str(self.id)+"_"+str(self.name)+"_"+str(index)+".jpeg", "wb+") as f:
                        f.write(req.content)
                else:
                    with open("full_images/" + "train/" + str(ranking)+ "//"+ str(self.id)+"_"+str(self.name)+"_"+str(index)+".jpeg", "wb+") as f:
                        f.write(req.content)
                f.close()

                f.close()
                time.sleep(float((randint(1, 9)/10)))
                
#this communicates with the api in order to return a list of people that we can actually start interacting with
                
class tinderAPI():

    def __init__(self, token):
        self._token = token

    def profile(self):
        data = requests.get(TINDER_URL + "/v2/profile?include=account%2Cuser", headers={"X-Auth-Token": self._token}).json()
        return Profile(data["data"], self)

    def matches(self, limit=10):
        data = requests.get(TINDER_URL + f"/v2/matches?count={limit}", headers={"X-Auth-Token": self._token}).json()
        return list(map(lambda match: Person(match["person"], self), data["data"]["matches"]))

    def like(self, user_id):
        data = requests.get(TINDER_URL + f"/like/{user_id}", headers={"X-Auth-Token": self._token}).json()
        return {
            "is_match": data["match"],
            "liked_remaining": data["likes_remaining"]
        }

    def dislike(self, user_id):
        requests.get(TINDER_URL + f"/pass/{user_id}", headers={"X-Auth-Token": self._token}).json()
        return True

    def nearby_persons(self):
        data = requests.get(TINDER_URL + "/v2/recs/core", headers={"X-Auth-Token": self._token}).json()
        return list(map(lambda user: Person(user["user"], self), data["data"]["results"]))


if __name__ == "__main__":
    skip_ranking = input("Do you want to rank players? [y/n]")
    if skip_ranking == "Y" or skip_ranking == "y":
        token = input("Log onto your Tinder profile from Google Chrome. Open inspect element, and under you should find an event log under the networks tabs which should have a log something along the lines of updates?locale=(your language). After clicking the log, and scrolling to the bottom you should see a variable named x-auth-token. Copy and paste that code here: ")
        api = tinderAPI(token)
        folderspawner()
        finished = 1
        count = 0

        #You can put however much data you want in here

        while True:
            persons = api.nearby_persons()
            for person in persons:
                person.download_images()
                count+=1
                if count % 5 == 0:
                    finished = input("Would you like to continue strengthening the algorithms? [0 for NO, 1 for YES] : ")
                    while not finished.isdigit():
                        finished = input("Would you like to continue strengthening the algorithms? [0 for NO, 1 for YES] : ")
                    if finished == 0:
                        break
                
    builder = tfds.ImageFolder("full_images/")
    full_train_ds = builder.as_dataset(split = "train", shuffle_files = True, deoders="UTF-8")
    full_test_ds = builder.as_dataset(split = "test", shuffle_files = True, deoders="UTF-8")

    
    builder = tfds.ImageFolder("faces/")
    face_train_ds = builder.as_dataset(split = "train", shuffle_files = True, deoders="UTF-8")
    face_test_ds = builder.as_dataset(split = "test", shuffle_files = True, decoders="UTF-8")
    
    
#I know there are 2 seperate models here, but this way I can understand the users preference's and maybe use a combination of pixel-wise segmentation algorithms and OpenPose for more important things like public health
##    choice = input("Would you like to train the model now or collect more data for more accurate results later? [(Y,y), (N,n)]")
##    if choice == "Y" or choice == "y":
##        face_train_ds = tfds.folder_dataset.ImageFolder(
##          "faces/",
##          validation_split=0.2,
##          subset="train",
##          seed=123,
##          image_size=(64, 64),
##          batch_size=8
##        )
##        full_train_ds = tfds.folder_dataset.ImageFolder(
##          "full_images/",
##          validation_split=0.2,
##          subset="train",
##          seed=123,
##          image_size=(258, 512),
##          batch_size=8
##        )
##        full_val_ds = tfds.folder_dataset.ImageFolder(
##          "full_images/",
##          validation_split=0.2,
##          subset="test",
##          seed=123,
##          image_size=(258, 512),
##          batch_size=8
##        )
##        
##        face_train_ds = tfds.folder_dataset.ImageFolder(
##          "faces/",
##          validation_split=0.2,
##          subset="test",
##          seed=123,
##          image_size=(64, 64),
##          batch_size=8
##        )
##
##        face_train_ds = configure_for_performance(train_ds)
##        full_val_ds = configure_for_performance(val_ds)
##        full_train_ds = configure_for_performance(train_ds)
##        face_val_ds = configure_for_performance(val_ds)
##
##        
##        data_augmentation = tf.keras.Sequential([
##            layers.experimental.preprocessing.RandomRotation(0.25),
##            layers.experimental.preprocessing.RandomContrast(0.25)
##        ])
##        
##        face_model = models.Sequential()
##        face_model.add(data_augmentation)
##        face_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(8,64,64,3)))
##        face_model.add(layers.MaxPooling2D((2, 2)))
##        face_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
##        face_model.add(layers.MaxPooling2D((2, 2)))
##        face_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
##        face_model.add(layers.Flatten())
##        face_model.add(layers.Dense(64, activation='relu'))
##        face_model.add(layers.Dense(10))
##
##        face_model.compile(optimizer='adam',
##                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
##                  metrics=['accuracy'])
##
##        face_history = face_model.fit(face_train_ds, epochs=30, 
##                        validation_data=face_test_ds)
##
##        face_model.save("face_model.h5")
##
##        #So the models right now are very basic image classifying algorithms, but Im hoping that with enough compute even with a smaller dataset than one I can find online this will be more tailored to the user
##
##        full_model = models.Sequential()
##        full_model.add(data_augmentation)
##        full_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(8,650,350,3)))
##        full_model.add(layers.MaxPooling2D((2, 2)))
##        full_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
##        full_model.add(layers.MaxPooling2D((2, 2)))
##        full_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
##        full_model.add(layers.Flatten())
##        full_model.add(layers.Dense(64, activation='relu'))
##        full_model.add(layers.Dense(10))
##
##        full_model.compile(optimizer='adam',
##                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
##                  metrics=['accuracy'])
##
##        full_history = face_model.fit(full_train_ds, epochs=30, 
##                        validation_data=full_test_ds)
##
##        full_model.save("full_model.h5")
##
##    else:
##        print("Have a nice day!")


