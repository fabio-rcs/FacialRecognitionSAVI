#!/usr/bin/env python3

class Make_Something:

    def __init__(self,root,value):
        
        self.root = root
        self.value = value
        self.DB_Orig = False
        self.DB_RealT = False
        self.DB_Reset = False


    def make_something(self):
        
        if self.value == 0:
            self.DB_Orig = True
            self.DB_RealT = False
            self.DB_Reset = False
                        
        if self.value == 1: 
            self.DB_Orig = False
            self.DB_RealT = True
            self.DB_Reset = False 
        
        if self.value == 2:
            self.DB_Orig = False
            self.DB_RealT = False
            self.DB_Reset = True
            
        self.root.destroy()
        return self.DB_Orig, self.DB_RealT, self.DB_Reset