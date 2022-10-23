#!/usr/bin/env python3

import pyttsx3
engine = pyttsx3.init() # object creation

engine.setProperty('rate', 125)     # setting up new voice rate
engine.setProperty('volume',2.0)    # setting up volume level  between 0 and 1

name = 'Name'
engine.say("Hello" + name)

engine.runAndWait()
engine.stop()
