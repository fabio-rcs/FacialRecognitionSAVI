#!/usr/bin/env python3

class Make_Something:

    def __init__(self,root,value):
        
        self.root = root
        self.value = value
        self.BaseD_OFF = False
        self.BaseD_ON = False


    def make_something(self):
        
        if self.value == 0:
            self.BaseD_ON = True
            self.BaseD_OFF = False
                        
        elif self.value == 1: 
            self.BaseD_ON = False
            self.BaseD_OFF = True 

        else:
            self.BaseD_ON = False
            self.BaseD_OFF = False
        
        #print('On ' + str(self.BaseD_ON) + ' OFF ' + str(self.BaseD_OFF))
        self.root.destroy()
        return self.BaseD_ON, self.BaseD_OFF