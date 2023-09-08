#instructions
#1 - pick someone from your group to share their screen and type code
#2 - add all group member names below as first initial, last name
#3 - complete the breakout task
#4 - make sure the code runs and submit if requested

#member 1
#member 2
#member 3
#member 4

import matplotlib.pyplot as plt
import math
import numpy as np
from numpy import arange

def initial_func(x,c, L, decayL):
  for i in range(0,len(x)):
     dis = abs(x[i] - L/2)
     c[i] = math.exp(-dis/decayL)
  
  return c