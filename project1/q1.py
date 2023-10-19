# %%
import matplotlib.pyplot as plt
import math
import numpy as np
from numpy import arange, array
from numpy.linalg import inv,solve

# %%
def initial_func(x,c, L, decayL):
  for i in range(0,len(x)):
     dis = abs(x[i] - L/2)
     c[i] = math.exp(-dis/decayL)
  
  return c

# %%
L = 100.0
N = 1000

#diffusion constant
D = 1000

#velocity
v = 100.0

decayL = L/8

#time steps
n = 1000
Tfinal = 1.0
dt = Tfinal/n

dx = L/N

# %%
x = arange(0.0,L+dx,dx)
c0 = np.zeros_like(x)

#initialize right side
b = np.zeros_like(x)
initial_func(x,c0,L,decayL)

# %%
s = (N+1,N+1)
Amat = np.zeros(s)

#set up first and last rows
Amat[0,0] = -1/dx
Amat[0,1] = 1/dx

Amat[N,N-1] = -1/dx
Amat[N,N] = 1/dx

#diffusive piece
for i in range (1,len(Amat)-1):
   Amat[i,i] = 1.0 + 2*D*dt/(dx*dx)
   Amat[i,i-1] = -D*dt/(dx*dx)
   Amat[i,i+1] = -D*dt/(dx*dx)

# %%
# we want to write a function to calculate the mass of the contaminant in the domain
# integrate the concentration over the domain
def mass(c,dx):
    """
    Computes mass at a particular time step 
    """

    # integrate the concentration over the domain
    mass = 0
    for i in range (1,len(c)-1):
        mass = mass + c[i]*dx
    
    mass = mass + c[0]*dx/2 + c[len(c)-1]*dx/2

    return mass

# %%
#initialize b
for i in range (1,len(c0)-1):
   b[i] = c0[i]

#initialize c
c = np.zeros_like(x)

t = 0
tsteps = np.zeros(n+1)

mass_ss_list = [mass(c0,dx)]
# print("t = ", 0, " ", "mass = ", mass(c0,dx,L))

for k in range (1,n+1):
    t = t + dt
    tsteps[k] = t
    c = solve(Amat,b)

    # computes mass at every 100 time steps
    if k%10 == 0:
        mass_ss_list.append(mass(c,dx))
        # print("t = ", round(t, 3), "mass = ", mass(c,dx,L))
    for i in range (1,len(c)-1):
       b[i] = c[i]

# %%
# plot a graph of mass vs time and take also plot the mean
mass_mean = np.mean(mass_ss_list)
print("mean mass = ", mass_mean)
print("mass at t = 0 = ", mass_ss_list[0])
print("max deviation from mean = ", max(abs(mass_ss_list - mass_mean)))

iter = np.arange(0,n+1,10)
# print(iter)

fig = plt.figure(figsize=(4.5,3))
plt.plot(iter, mass_ss_list, 'r-', label='mass')
plt.plot(iter, mass_mean*np.ones(len(iter)), 'b--', label='mean')
plt.grid()
plt.xlabel('timestep')
plt.ylabel('mass')
plt.legend()
plt.xlim(0,1000)
plt.ylim(mass_mean-0.1,mass_mean+0.1)
plt.title('Mass vs Time @ every 10 timesteps', fontsize=10)


