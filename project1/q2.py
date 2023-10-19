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
def solve_c_mass(v, dx, dt, N, D, L, n):
    # define auxiliary variables
    sigma = D*dt/(dx*dx)
    lmda = v*dt/dx

    x = arange(0.0,L+dx,dx)
    c0 = np.zeros_like(x)

    #initialize right side
    b = np.zeros_like(x)
    decayL = L/8
    initial_func(x,c0,L,decayL)

    s = (N+1,N+1)
    Amat = np.zeros(s)

    #set up first and last rows
    Amat[0,0] = -D/dx - v
    Amat[0,1] = D/dx

    Amat[N,N-1] = -D/dx - v
    Amat[N,N] = D/dx

    #diffusive piece
    for i in range (1,len(Amat)-1):
        Amat[i,i] = 2*sigma - lmda + 1
        Amat[i,i-1] = -sigma
        Amat[i,i+1] = -sigma + lmda

    #initialize b
    for i in range (1,len(c0)-1):
        b[i] = c0[i]

    #initialize c
    c = np.zeros_like(x)

    t = 0
    tsteps = np.zeros(n+1)

    #store initial mass
    initial_m = mass(c0,dx)

    for k in range (1,n+1):
        t = t + dt
        tsteps[k] = t
        c = solve(Amat,b)

        for i in range (1,len(c)-1):
            b[i] = c[i]    

    final_m = mass(c,dx)
    
    return initial_m, final_m

# %%
#diffusion constant
D = 1000

#time steps
n = 1000
Tfinal = 1.0
dt = Tfinal/n

# domain
L = 100.0

### change parameters here
v = 10
N_list = [100]

for N in N_list:
    dx = L/N
    print("dx = ", dx)
    print(initial_m)
    print(final_m)
    initial_m, final_m = solve_c_mass(v, dx, dt, N, D, L, n)
    print("rel error (%) = ", abs(final_m - initial_m)/initial_m * 100, "%")
    print("")

# %%
#diffusion constant
D = 1000

#time steps
n = 1000
Tfinal = 1.0
dt = Tfinal/n

# domain
L = 100.0

### change parameters here
v = 100
N_list = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
rel_err_list = []

for N in N_list:
    dx = L/N
    print("dx = ", dx)
    # print(initial_m)
    # print(final_m)
    initial_m, final_m = solve_c_mass(v, dx, dt, N, D, L, n)
    rel_err = abs(final_m - initial_m)/initial_m * 100
    rel_err_list.append(rel_err)
        # if error is less than 2 percent, break
    if rel_err < 2:
        break
    print("rel error (%) = ", rel_err, "%")
    print("")


# %%
print("N_list = ", N_list[:len(rel_err_list)])
dx_list = [L/N for N in N_list[:len(rel_err_list)]]
print("dx_list = ", dx_list)
print("rel_err_list = ", rel_err_list)

# %%
# plot a graph of the relative error vs dx
plt.figure(figsize = (4.5, 2.7))
plt.plot(dx_list, rel_err_list, 'o-', markersize = 3, linewidth = 1)
plt.xlabel("dx, spatial discretization")
plt.ylabel("relative error (%)")
plt.title("Relative error vs dx")
plt.grid(True)

# %%
def solve_c_mass(v, dx, dt, N, D, L, n):
    # define auxiliary variables
    sigma = D*dt/(dx*dx)
    lmda = v*dt/dx

    x = arange(0.0,L+dx,dx)
    c0 = np.zeros_like(x)

    #initialize right side
    b = np.zeros_like(x)
    decayL = L/8
    initial_func(x,c0,L,decayL)

    s = (N+1,N+1)
    Amat = np.zeros(s)

    #set up first and last rows
    Amat[0,0] = -D/dx - v
    Amat[0,1] = D/dx

    Amat[N,N-1] = -D/dx - v
    Amat[N,N] = D/dx

    #diffusive piece
    for i in range (1,len(Amat)-1):
        Amat[i,i] = 2*sigma - lmda + 1
        Amat[i,i-1] = -sigma
        Amat[i,i+1] = -sigma + lmda

    #initialize b
    for i in range (1,len(c0)-1):
        b[i] = c0[i]

    #initialize c
    c = np.zeros_like(x)

    t = 0
    tsteps = np.zeros(n+1)

    #store initial mass
    initial_m = mass(c0,dx)

    for k in range (1,n+1):
        t = t + dt
        tsteps[k] = t
        c = solve(Amat,b)
        
        # for plotting
        if (k % 25) == 0:
            # plt.clf()
            plt.plot(x,c)
            # plt.suptitle("Time = %1.3f" % t)
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("c")
        plt.xlim(0,100)
        # legend
        # plt.legend(loc = "upper right")

        for i in range (1,len(c)-1):
            b[i] = c[i]    

    final_m = mass(c,dx)
    
    return initial_m, final_m

# %%
# we want to plot out to show the difference between the two cases

#diffusion constant
D = 1000

#time steps
n = 1000
Tfinal = 1.0
dt = Tfinal/n

# domain
L = 100.0

### change parameters here
v = 10
N = 100
dx = L/N
print("dx = ", dx)
plt.figure(figsize = (4.5, 3))
plt.title("v = 10")
initial_m, final_m = solve_c_mass(v, dx, dt, N, D, L, n)
plt.show()

### change parameters here
v = 100
N = 100
dx = L/N
print("dx = ", dx)
plt.figure(figsize = (4.5, 3))
plt.title("v = 100")
initial_m, final_m = solve_c_mass(v, dx, dt, N, D, L, n)
plt.show()


