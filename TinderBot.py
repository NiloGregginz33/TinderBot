import pynder
import time
import random
import requests
import itertools
import kivy
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.stacklayout import StackLayout

Builder.load_string("""
<TinderScreen>
    BoxLayout:
        Button:
            id: X_Auth_publish
            text: "Submit your Xauth Key and Start the swiping session"
            on_click: app.update_XAuth()
        TextInput:
            id: X_Auth_input
            hint_text: "Enter Current X_Auth token by logging into Tinder.com, and inspecting the network page elements"
""")

class TinderScreen:    
    pass    



class TinderBot(MDApp):

    def scheduling():
        pass

    schedule = None
    users = None
    session = None
    user_num = 0
    xAuth = "5546071e-7c89-407c-a7ae-fb303526bfa5"

    def build(self):
        layout = TinderScreen()
        return layout
    
    def update_XAuth(self):
        self.xAuth = self.ids['X_Auth_input'].text()
        self.getSess()
        self.xAuth = string
        try:
            self.session = pynder.Session(XAuthToken=self.xAuth)
            self.user_num = 0
            self.users = session.nearby_users()
        except:
            print("unable to start sess")

    def scheduling(self):
        seconds = time.time()
        local_time = time.localtime(seconds)
        minutes = local_time.tm_min
        hours = local_time.tm_hour
        if int(hours) % 6 == 0 and int(minutes) == 30:
            time.sleep(random.randrange((0, 1800)))
            self.update_XAuth()
            return True
        else:
            print("Failed")
            return False

    def swiping(self):
        i = 0 
        for i in range(0, 50+random.randrange((-10, 10))):
            for user in itertools.islice(self.users, 5):
                time.sleep(2 + ((0.15+random.randrange(0.5, 0.8)) * random.randrange(0, 6)))
                self.user_num+=1
                if self.user_num % 7 == 0 and self.user_num % 4 == 0:
                    user.like()
                    print("Liked User")
                elif self.user_num %7 != 0:
                    user.like()
                    print("Liked User")
                elif self.user_num % 4 == 0 or self.user_num % 7 == 0:
                    user.dislike()
                    print("Disliked User")
                elif self.user_num % 10 == 0:
                    user.like()
                    print("Liked User")

                else:
                    continue
        print("Done With Swiping Session")


    schedule = self.scheduling()
if __name__ == '__main__':
    TinderBot().run()
