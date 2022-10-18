#!/usr/bin/env python3
import tkinter as tk                # python 3
from tkinter import font as tkfont

BaseD_ON = False
BadeD_OFF = False 

def make_something(value):
	if value == True:
		BaseD_ON = True
		BaseD_OFF = False
		root.destroy()
		print('On ' + str(BaseD_ON) + ' OFF ' + str(BaseD_OFF))
	else: 
		BaseD_ON = False
		BaseD_OFF = True
		root.destroy()
		print('On ' + str(BaseD_ON) + ' OFF ' + str(BaseD_OFF))
	
root = tk.Tk()
frm = tk.Frame(root)
frm.grid()

tk.Label(frm, text="Escolha tipo de inicio").grid(column=0, row=0)
tk.Label(frm, text="").grid(column=0, row=1)

tk.Button(frm, text="BaseD ON", command = lambda *args: make_something(True)).grid(column=0, row=2)
tk.Button(frm, text="BaseD OFF", command = lambda *args: make_something(False)).grid(column=0, row=3)

root.mainloop()  


