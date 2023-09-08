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
# Extend the code below to calculate the backward and forward errors 
# backward error = norm of (b - Ax_a)
# forward error = norm of ( x - x_a )
# your code should print both errors out to the screen



import numpy as np
import math

def inf_norm(x):
    max = 0.0
    for i in range(0,len(x)):
        if (abs(x[i]) > max):
            max = abs(x[i])
    return max


A = np.array([[1, 1], [ 1.0001, 1]])
xa = np.array([-1, 3.0001])
x = np.array([1, 1])
b = np.array([2, 2.0001])


def backward_error(A, xa, b):
    return inf_norm(b - np.matmul(A, xa))

def forward_error(x, xa):
    return inf_norm(x - xa)

print("backward error: ", backward_error(A, xa, b))
print("forward error: ", forward_error(x, xa))