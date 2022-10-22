#!/usr/bin/env python3

""" Falta organizar estes imports 
	Talvez meter isto por classes para ser mais organizado"""

import tkinter as tk                # python 3
from tkinter import font as tkfont
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import os
import time
import numpy as np
import cv2

# -----------------------------------------------------------
# Start mode selection, with or without database
# -----------------------------------------------------------
BaseD_ON = False
BaseD_OFF = False 

def make_something(value):
	global BaseD_OFF
	global BaseD_ON
	if value == True:
		BaseD_ON = True
		BaseD_OFF = False
		root.destroy()
		#print('On ' + str(BaseD_ON) + ' OFF ' + str(BaseD_OFF))
	else: 
		BaseD_ON = False
		BaseD_OFF = True 
		root.destroy()
		#print('On ' + str(BaseD_ON) + ' OFF ' + str(BaseD_OFF))
	
root = tk.Tk()
frm = tk.Frame(root)
frm.grid()

tk.Label(frm, text="Escolha tipo de inicio").grid(column=0, row=0)
tk.Label(frm, text="").grid(column=0, row=1)

tk.Button(frm, text="BaseD ON", command = lambda *args: make_something(True)).grid(column=0, row=2)
tk.Button(frm, text="BaseD OFF", command = lambda *args: make_something(False)).grid(column=0, row=3)

root.mainloop()

# -----------------------------------------------------------
# View database
# -----------------------------------------------------------
if BaseD_ON == True:
	dir_images = './Images'
	images_names = os.listdir(dir_images)
	num_total_img = np.array(images_names).size

	id = 0
	plt.figure(figsize=(2, 16))	
	for image in images_names:
		id += 1
		img = cv2.imread(str('./Images/'+image))
		plt.subplot(num_total_img,1,id), plt.imshow(img)
		plt.title(image.rsplit('.',1)[0]), plt.xticks([]), plt.yticks([])
		plt.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
	
	plt.show()

	
