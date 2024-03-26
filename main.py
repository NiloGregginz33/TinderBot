# Manav Naik 6/2/2022

import pprint

from google.api_core.exceptions import InvalidArgument
from google.cloud import dialogflow_v2beta1 as dialogflow
import apiai

from tinder_ex.tinder.http import *
from tinder_ex.tinder.entities.user import *
from tinder_ex.tinder.entities.message import Message
from tinder_ex.tinder.tinder import TinderClient
from tinder_ex.tinder.http import *

import os
import datetime
from random import random, uniform
from time import sleep
import time
import PIL
from PIL import Image
import math

import cv2
import tensorflow as tf
import numpy as np
import pandas as pd

import openai
import csv

from collections import Counter

from google.api_core.client_options import ClientOptions


project_id = "blabber-ada4c"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'google.json'

openai.api_key= ""



pp = pprint.PrettyPrinter(indent=2)
# This function takes a message and a prompt and then uses gpt 2 to fill in the conversational gaps

def counting_messages(message_history, tinder_session):

    x = 0

    msg = ""

    girl_responses = 0

    flag = False

    if message_history != []:

        for message in message_history:

            if(message.author_id != tinder_session.get_self_user().id):

                x += 1

                girl_responses = 0

                flag = True

            else:

                flag = False

                x = 0

                girl_responses += 1


        return message_history[(len(message_history) - (x + 1))]

    else:

        return "default text"

    
def openai_request(q_prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=q_prompt,
        temperature=0.6,
        max_tokens=40,
        top_p=1,
        frequency_penalty=0.22,
        presence_penalty=0.1,
        stop=["\nAI: "]
    )
    final_response = str(response["choices"][0]["text"])
    try:
        return final_response
    except:
        return None

def dialogflow_request(text):
    """Returns the result of detect intent with texts as inputs.

    Using the same `session_id` between requests allows continuation
    of the conversation."""

    session_id = "random text"

    session_client = dialogflow.SessionsClient()

    session = session_client.session_path(project_id, session_id)
    
    print("Session path: {}\n".format(session))
        
    text_input = dialogflow.TextInput(text=text, language_code="en")

    query_input = dialogflow.QueryInput(text=text_input)

    knowledge_base_path = dialogflow.KnowledgeBasesClient.knowledge_base_path(
            project_id, "MTQ5NjQ5NzY2NzcxMjQzMDg5OTI"
        )

    query_params = dialogflow.QueryParameters(
            knowledge_base_names=[knowledge_base_path]
        )

    response = session_client.detect_intent(
            request={"session": session, "query_input": query_input, "query_params": query_params}
        )

    print(response)

    print("=" * 20)
    print("Query text: {}".format(response.query_result.query_text))
    print(
            "Detected intent: {} Technical name: {}(confidence: {})\n".format(
                response.query_result.intent.display_name,
                response.query_result.intent.name,
                response.query_result.intent_detection_confidence,
            )
        )


    if response.query_result.intent.display_name == "Default Fallback Intent":

        #knowledge_answers = response.query_result.knowledgeAnswers
        #answer = knowledge_answers.answers[0].answer
        #knowledge_confidence = knowledge_answers.answers[0].match_confidence

        fulfillment_text = openai_request(response.query_result.query_text)

    else:

        fulfillment_text = response.query_result.fulfillment_text
    
    print("Fulfillment text: {}\n".format(fulfillment_text))

    return fulfillment_text

# The messaging part of the code
# Pretty straight forward, we find the last message, compare its author id to our user id to determine who texted last when deciding to respond

def message_matches(question_bank, tinder_session):

    base_string = ""
    for match in tinder_session.load_all_matches():
        
        messages = match.message_history
        message = messages.load_all_messages()
        print(len(message))

        last_msg = []

        msgs = []

        for msg in message:
            last_msg.append(msg)
            msgs.append(msg.content)

        counter = intersection(msgs, question_bank)

        unasked = []

        for i in question_bank:
            if not i in counter:
                unasked.append(i)

        print(counter)

        if len(message) != 0:

            last_message = last_msg[-1]
            print(last_message)


            last_user_text = counting_messages(last_msg, tinder_session)

            print("test")

            print(last_user_text)

            print(last_message.author_id)

            print(tinder_session.get_self_user())
            
            if len(message) != 1 and len(message) != 0 and last_msg != [] and last_message.author_id != tinder_session.get_self_user().id and last_user_text.content != "thank you for interacting with this AI that acts like me and swipes on girls it thinks are hot. If you want to know something vulnerable about me, schedule a date with me, hear a joke, or know something more light hearted about me just ask but stop responding if you don't want a response." and last_user_text != last_msg[-1]:

                base_string += "\n AI: {}".format(str(last_user_text))
                base_string += "\n AI: {}".format(str(last_message.content))
                df_response = dialogflow_request(str(last_user_text.content))
                match.send_message(df_response)
                df_response = dialogflow_request(str(last_message.content))
                match.send_message(df_response)
                base_string += "\n AI: {}".format(df_response)
                question = unasked[floor(uniform(0,len(unasked)-1))]
                match.send_message(question)
                base_string += "\n AI: {}".format(question)
                print("working")
                print(last_user_text.content)
                
        elif len(message) == 0:
            match.send_message("hey")

def euclidean_distance(a, b):
    x1 = a[0];
    y1 = a[1]
    x2 = b[0];
    y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


def afk_mode(rec, photos):
    time.sleep(1 + uniform(2.5, 3.5))
    path = "temp_image"
    x = 0
    for x in range(0, len(photos) - 1):
        url = photos[x].url
        temp_match = photos[x]
        r = requests.get(url)
        file = open(path + "_" + str(x) + ".jpg", "wb+")
        file.write(r.content)
        file.close()
        image = cv2.imread(path + "_" + str(x) + ".jpg")
        img_original = image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        print(len(faces))

        if len(faces) != 0:
            face_x, face_y, face_w, face_h = faces[0]

            img = image[int(face_y):int(face_y + face_h), int(face_x):int(face_x + face_w)]
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            eyes = eye_detector.detectMultiScale(img_gray)

            index = 0

            if len(eyes) != 0 and len(eyes) != 1:
                
                for (eye_x, eye_y, eye_w, eye_h) in eyes:
                    if index == 0:
                        eye_1 = (eye_x, eye_y, eye_w, eye_h)
                        index = index + 1
                    elif index == 1:
                        eye_2 = (eye_x, eye_y, eye_w, eye_h)

                if eye_1[0] <= eye_2[0]:
                    left_eye = eye_1
                    right_eye = eye_2
                else:
                    left_eye = eye_2
                    right_eye = eye_1

                left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
                left_eye_x = left_eye_center[0];
                left_eye_y = left_eye_center[1]

                right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
                right_eye_x = right_eye_center[0];
                right_eye_y = right_eye_center[1]

                if left_eye_y <= right_eye_y:
                    point_3rd = (right_eye_x, left_eye_y)
                    direction = -1  # rotate same direction to clock
                    print("rotate to clock direction")
                else:
                    point_3rd = (left_eye_x, right_eye_y)
                    direction = 1  # rotate inverse direction of clock
                    print("rotate to inverse clock direction")

                a = euclidean_distance(left_eye_center, point_3rd)
                b = euclidean_distance(right_eye_center, left_eye_center)
                c = euclidean_distance(right_eye_center, point_3rd)

                if b != 0 and c != 0:
                    cos_a = (b * b + c * c - a * a) / (2 * b * c)
                    print("cos(a) = ", cos_a)

                    angle = np.arccos(cos_a)
                    print("angle: ", angle, " in radian")

                    angle = (angle * 180) / math.pi
                    print("angle: ", angle, " in degree")

                    if direction == -1:
                        angle = 90 - angle
                else:
                    angle = 0

                new_img = Image.fromarray(img_original)
                new_img = np.array(new_img.rotate(direction * angle))

                new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

                faces = face_detector.detectMultiScale(new_gray, 1.3, 5)

                if len(faces) != 0:

                    face_x, face_y, face_w, face_h = faces[0]

                    img = new_img[int(face_y):int(face_y + face_h), int(face_x):int(face_x + face_w)]

                else:
                    img = new_img

                img_resized = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
                cv2.imwrite("temp_img.jpg", img_resized)

                raw = tf.io.read_file("temp_img.jpg")
                image = tf.io.decode_image(raw, channels=3)

                image = tf.constant(image)
            
                input_data = np.array(image, dtype=np.float32)
                input_data = np.reshape(input_data,(1, 128, 128, 3))

                model = tf.keras.models.load_model('facial_attractiveness.h5')

                prediction = model.predict(input_data)

                output = np.argmax(prediction)

                print(output/50)

                scaled_output = output/50

                if scaled_output > 6.5:
                    rec.like()
                    print("user_liked")
                else:
                    rec.dislike()
                    print("user_disliked")

        else:
            rec.like()

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

if __name__ == "__main__":

    X_Auth_Token = ""

    cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))

    # X_Auth_Token = input("Log into tinder on your computer, click inspect element, and then scroll to the bar labeled network. Under there you will find a history of network requests being made. Find your XAuth token from a below network request and enter it here: ")
    tinder_session = TinderClient(X_Auth_Token, log_level=1, ratelimit=20)

    start_chat_log = "hey"

    question_bank = ["What's your favorite color?", "Do you have any pets?",
                     "Has anyone changed your life without knowing it?",
                     "What's your major?", "Whats your favorite food?",
                     "What do I need to watch as soon as i have the time?",
                     "What insecurity of yours holds you back the most?",
                     "What's the most pain you've ever been in that wasn't physical?",
                     "What lesson took you the longest to unlearn?", "What's been keeping you sane lately?",
                     "Are you missing anyone right now and do you think they're missing you too?",
                     "How would you describe the feeling of being in love in one word?",
                     "How can I be there for you during this chapter?",
                     "What have you tolerated from people in the past that i no longer have space for?"
                     ]

    while True:
        message_matches(question_bank, tinder_session)
        recs = tinder_session.get_recommendations()
        for rec in recs[:2]:
            print(rec)
            photos = rec.photos            
            afk_mode(rec, photos)
            time.sleep(1 + int(uniform(2, 3)))
            
        
