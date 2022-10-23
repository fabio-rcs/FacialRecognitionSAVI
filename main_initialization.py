#!/usr/bin/env python3

import cv2
import os
import numpy as np
import tkinter as tk              
from matplotlib import pyplot as plt
from MakeSomething import Make_Something
# -----------------------------------------------------------
# Start mode selection, with or without database
# -----------------------------------------------------------

root = tk.Tk()
frm = tk.Frame(root)
frm.grid()

def Value(value):
	global BaseD_ON
	global BaseD_OFF
	value = value
	BaseD_ON,BaseD_OFF = Make_Something(root,value).make_something()

tk.Label(frm, text="Escolha tipo de inicio").grid(column=0, row=0)
tk.Label(frm, text="").grid(column=0, row=1)

tk.Button(frm, text="BaseD ON", command = lambda *args: Value(0), activebackground='green').grid(column=0, row=2)
tk.Button(frm, text="BaseD OFF", command = lambda *args: Value(1), activebackground='green').grid(column=0, row=3)
tk.Button(frm, text="Espaço Branco", command = lambda *args: Value(2), activebackground='green').grid(column=0, row=4)

root.mainloop()

# -----------------------------------------------------------
# View database
# -----------------------------------------------------------

if BaseD_ON == True:
	dir_images = './Images'
	images_names = os.listdir(dir_images)
	num_total_img = np.array(images_names).size

	# ------------------------------------------
    # Plot with the people in the database
    # ------------------------------------------
	id = 0
	plt.figure("DataBase",figsize=(3, 16))
	for image in images_names:
		id += 1
		img = cv2.imread(str('./Images/'+image))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		plt.subplot(num_total_img,1,id), plt.imshow(img)
		plt.title(image.rsplit('.',1)[0]), plt.xticks([]), plt.yticks([])
		plt.subplots_adjust(top=0.95, bottom=0.08, right=0.95, hspace=0.25)
	
	plt.show()