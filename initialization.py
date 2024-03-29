#!/usr/bin/env python3

import cv2
import os
import pickle
import shutil
import numpy as np     
import tkinter as tk         
from matplotlib import pyplot as plt
from MakeSomething import Make_Something
import shutil
class Initialization:

	def __init__(self,dir_db,dir_db_backup,dir_image,dir_image_backup):	
		self.dir_db = dir_db
		self.dir_db_backup = dir_db_backup
		self.dir_image = dir_image
		self.dir_image_backup = dir_image_backup
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
			self.DB_Orig,self.DB_RealT,self.DB_Reset = Make_Something(root,value).make_something()

		# title (app)
		tk.Label(frm, text="Inicial setup").grid(column=0, row=0)
		tk.Label(frm, text="").grid(column=0, row=1)

		# Buttons (app)
		tk.Button(frm, text="With original database", command = lambda *args: Value(0), activebackground='green').grid(column=0, row=2)
		tk.Button(frm, text="With database", command = lambda *args: Value(1), activebackground='green').grid(column=0, row=3)
		tk.Button(frm, text="Reset database", command = lambda *args: Value(2), activebackground='green').grid(column=0, row=4)

		root.mainloop()
	
		return self.DB_Orig, self.DB_RealT, self.DB_Reset


	def open_database(self):
		
		known_face_encodings = []
		known_face_names = []
		if self.DB_Orig:
			with open(self.dir_db_backup, 'rb') as f:
				known_face_names, known_face_encodings = pickle.load(f)
		if self.DB_RealT:
			with open(self.dir_db, 'rb') as f:
				known_face_names, known_face_encodings = pickle.load(f)
		if self.DB_Reset:
			with open(self.dir_db, 'wb') as f:
				pickle.dump([],f)
		
		return known_face_names, known_face_encodings
		

	def save_database(self,known_face_names,known_face_encodings):
		# if self.DB_Reset or self.DB_RealT:
		with open(self.dir_db, 'wb') as f:
			pickle.dump((known_face_names, known_face_encodings),f)
		

	def select_diretory(self):
	
		if self.DB_Orig:
			for filename in os.listdir(self.dir_image):
				filepath = os.path.join(self.dir_image, filename)
				try:
					shutil.rmtree(filepath)
				except OSError:
					os.remove(filepath)
			files = os.listdir(self.dir_image_backup)
			for file in files:
				shutil.copy2(os.path.join(self.dir_image_backup,file), self.dir_image)

		if self.DB_RealT:
			self.dir_image = self.dir_image
		
		if self.DB_Reset:
			for filename in os.listdir(self.dir_image):
				filepath = os.path.join(self.dir_image, filename)
				try:
					shutil.rmtree(filepath)
				except OSError:
					os.remove(filepath)

	def view_database(self):
		known_names = []
		try:
			with open(self.dir_db, 'rb') as f:
				known_names, _ = pickle.load(f)
		except:
			pass
		
		images_names = os.listdir(self.dir_image)
		num_total_img = np.array(images_names).size

		# ------------------------------------------
		# Plot with the people in the database
		# ------------------------------------------
		while num_total_img < 1:
			images_names = os.listdir(self.dir_image)
			num_total_img = np.array(images_names).size

		if num_total_img >= 1 and num_total_img == len(known_names):
			id = 0
			plt.figure("DataBase",figsize=(3, 16))
			for image in images_names:
				id += 1
				img = cv2.imread(self.dir_image + '/' + image)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				plt.subplot(num_total_img,1,id), plt.imshow(img)
				plt.title(known_names[int(image.rsplit('.',1)[0])-1]), plt.xticks([]), plt.yticks([])
				plt.subplots_adjust(top=0.95, bottom=0.08, right=0.95, hspace=0.25)
			plt.show(block=False)
			known_names_update = known_names
			try:
				with open(self.dir_db, 'rb') as f:
					known_names_update, _ = pickle.load(f)
			except:
				pass
			while (len(known_names_update) != len(known_names)):
				try:
					with open(self.dir_db, 'rb') as f:
						known_names_update, _ = pickle.load(f)
				except:
					pass
			plt.pause(3)
			plt.close()

