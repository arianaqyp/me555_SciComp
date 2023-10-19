#code for discrete catenary

from numpy.linalg import inv,solve,norm
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from math import tan,sin,cos


def Feval(T,t1,t2,t3):
  F = array([[ T*(tan(t1) - tan(t2)) -16.0  ], \
             [ T*(tan(t2) + tan(t3)) -20.0  ], \
             [ -4.0*sin(t1) -6.0*sin(t2) + 5.0*sin(t3) + 3 ], \
             [ 4.0*cos(t1) + 6.0*cos(t2) + 5.0*cos(t3) - 12 ] ])
  return F

def DF(T1,t1,t2,t3):
  Jcob = array([[ tan(t1) - tan(t2), T/(cos(t1)**2) , -T/(cos(t2)**2), 0], \
                [ tan(t2) + tan(t3), 0,  T/(cos(t2)**2), T/(cos(t3)**2)], \
                [ 0, -4.0*cos(t1), -6.0*cos(t2), 5.0*cos(t3)], \
                [ 0, -4.0*sin(t1), -6.0*sin(t2), -5.0*sin(t3)]])
  return Jcob


def PlotConfig(t1,t2,t3,ax):
  x = np.zeros(4)
  y = np.zeros(4)
  x[1] = x[0] + 4.0*cos(t1)
  y[1] = y[0] - 4.0*sin(t1)
  x[2] = x[1] + 6.0*cos(t2)
  y[2] = y[1] - 6.0*sin(t2)
  x[3] = x[2] + 5.0*cos(t3)
  y[3] = y[2] + 5.0*sin(t3)
  ax.plot(x,y,x,y,'o')
  return 

theta = 45
theta_rad = theta*3.1415/180.0

#initial guesses 

t1 = 45*3.1415/180.0
t2 = t1/2.0
t3 = 45*3.1415/180



T = 15

error = 1
tol = 1.0e-12

iter = 0

fig = plt.gcf()
ax = fig.gca()

while (error > tol) & (iter < 11) :
   PlotConfig(t1,t2,t3,ax)
   iter = iter+1
   Resid = -Feval(T,t1,t2,t3)
   if (iter == 1):
       Resid0 = Resid
    
   error = norm(Resid)/norm(Resid0)
   
   DeltaX = solve(DF(T,t1,t2,t3),Resid)
   #print(DeltaX)
   T = T + DeltaX[0,0]
   t1 = t1 + DeltaX[1,0]
   t2 = t2 + DeltaX[2,0]
   t3 = t3 + DeltaX[3,0]

  


#plt.plot(x,y,x,y,'o')
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.show()