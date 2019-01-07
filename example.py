# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 14:07:13 2018

@author: piesauce
"""


class Example:
    """
    Class that represents a typical training, dev, and test example.
    """
    
    def __init__(self, X, Y, Z=None):
        """
        Initialization of an example.
        X = input text.
        Y = output label
        Z = private information
        """
        self.X = X
        self.Y = Y
        self.Z = Z
        
    def get_X(self):
        return self.X
    
    def get_Y(self):
        return self.Y
    
    def get_Z(self):
        return self.Z
    
    def get_training_example(self):
        return self.X, self.Y
    
    def get_adv_training_example(self):
        return self.X, self.Z
    
