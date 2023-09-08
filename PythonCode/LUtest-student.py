#instructions
#1 - pick someone from your group to share their screen and type code
#2 - add all group member names below as first initial, last name
#3 - complete the breakout task
#4 - make sure the code runs and submit if requested

#member 1
#member 2
#member 3
#member 4


#Breakout Task
#write code to verify that the L matrix and the U matrix below do in fact multiply to yield A, where
# A = [ 1 2 -1]
#     [ 2 1 -2]
#     [-3 1  1]
#
# If you have time, generalize this in the form of a function that takes any two matrices as inputs and checks
#  their product against a third matrix 

import numpy as np


L = np.array([[1,0, 0], [2, 1,0], [-3, -7.0/3.0,  1]])
U = np.array([[1,2,-1], [0,-3,0], [0,         0, -2]])
