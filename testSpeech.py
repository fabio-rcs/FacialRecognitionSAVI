#!/usr/bin/env python3

import pyttsx3
engine = pyttsx3.init("espeak")
#engine = pyttsx3.init()
engine.say("Hello!")
engine.runAndWait()
