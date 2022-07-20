# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 15:40:03 2021

@author: explo
"""

import numpy as np


# create random subset of n rows
def generateRandomRows(array, n):
    shuffled = np.random.shuffle(array)
    return shuffled[0: n, :]
    
