#!/usr/bin/env python3

import cv2
import os
import numpy as np     
import tkinter as tk         
from matplotlib import pyplot as plt
from MakeSomething import Make_Something

class Initialization:

	def __init__(self):	
		self.DB_Orig = False
		self.DB_RealT = False
		self.DB_Reset = False
		
	def app(self):
		# -----------------------------------------------------------
		# Start mode selection, with or without database (app)
		# -----------------------------------------------------------
		# initialization (app)
		root = tk.Tk()
		frm = tk.Frame(root)
		frm.grid()

		# command function (app)
		def Value(value):
			value = value
			self.DB_Orig,self.DB_RealT = Make_Something(root,value).make_something()

		# title (app)
		tk.Label(frm, text="Inicial setup").grid(column=0, row=0)
		tk.Label(frm, text="").grid(column=0, row=1)

		# Buttons (app)
		tk.Button(frm, text="With original database", command = lambda *args: Value(0), activebackground='green').grid(column=0, row=2)
		tk.Button(frm, text="With database", command = lambda *args: Value(1), activebackground='green').grid(column=0, row=3)
		tk.Button(frm, text="Reset database", command = lambda *args: Value(2), activebackground='green').grid(column=0, row=4)

		root.mainloop()
		return self.DB_Orig, self.DB_RealT, self.DB_Reset

	def view_database(self, dir_image, dir_image_backup):
		# -----------------------------------------------------------
		# View database
		# -----------------------------------------------------------

		if self.DB_Orig == True:
			dir_images = dir_image_backup
			
		else:
			dir_images = dir_image
		
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
			# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			plt.subplot(num_total_img,1,id), plt.imshow(img)
			plt.title(image.rsplit('.',1)[0]), plt.xticks([]), plt.yticks([])
			plt.subplots_adjust(top=0.95, bottom=0.08, right=0.95, hspace=0.25)
			
		plt.show()

