# %%
#code for discrete catenary
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import tensorflow as tf
import time

# %%
# definition of variables and initial guess
P = tf.Variable([0,-16, -20], dtype=tf.float64)
ell = tf.Variable([4.0,6.0,5.0], dtype=tf.float64)
a = tf.Variable(6.0, dtype=tf.float64)
d_vertical = tf.Variable(3, dtype=tf.float64)
    
# x = tf.Variable([50, 50*np.pi/180, 50*np.pi/180/2, -50*np.pi/180], dtype=tf.float64)
x = tf.Variable([15, 0.45, 0.1,-0.2], dtype=tf.float64)

# %%
def Feval(x, ell, a, P, d_vertical):
    N = x.shape[0]-1  #length of T and theta
    T = x[0]
    theta = x[1:]

    F = [tf.Variable(0.0, dtype=tf.float64) for _ in range(N+1)]
    
    for i in range(N-1):
        F[i] = F[i] + T*(tf.tan(theta[i])-tf.tan(theta[i+1])) + P[i+1] 
        F[N-1] = F[N-1] + ell[i]*tf.cos(theta[i])
        F[N]  =  F[N] + ell[i]*tf.sin(-theta[i])
    
    F[N-1] = F[N-1] + ell[N-1]*tf.cos(theta[N-1]) - 2*a
    F[N] = F[N] + ell[N-1]*tf.sin(-theta[N-1]) + d_vertical

    # Convert the list of tensors back into a single tensor
    F = tf.stack(F)
    
    return F

# %%
# calculate the jacobian matrix with autodiff
with tf.GradientTape() as tape:
    # forward pass
    F = Feval(x, ell, a, P, d_vertical)
    print("F is: \n", F)
# get the gradient of F with respect to x
J = tape.jacobian(F, x)
print("J is: \n", J)

# %%
# function for FD
def J_FD(x, eps, h, ell=ell, a=a, P=P, d_vertical=d_vertical):
    """
    x: point at which to evaluate the Jacobian
    eps: perturbation to x (step size)
    h: arbitrary vector
    """
    # have everything in float64
    x = x.numpy().astype(np.float64)
    eps = np.float64(eps)
    h = h.astype(np.float64)
    ell = ell.numpy().astype(np.float64)
    a = a.numpy().astype(np.float64)
    P = P.numpy().astype(np.float64)
    d_vertical = d_vertical.numpy().astype(np.float64)

    N = x.shape[0]-1
    J = np.zeros((N+1, N+1), dtype=np.float64)
    for i in range(N+1):
        x_pert = x + eps*h[i]
        F_pert = Feval(x_pert, ell, a, P, d_vertical)
        F = Feval(x, ell, a, P, d_vertical)
        J[:,i] = (F_pert - F)/eps
    return J

# %%
print("J_FD is: \n", J_FD(x, 1e-6, np.eye(4)))

# %%
# eps from 1e-1 to 1e-12
eps_array = np.logspace(-1, -12, 12)
print("eps_array is: \n", eps_array)

err_list = []

for i in range(len(eps_array)):
    eps = eps_array[i]
    J_FD_ = J_FD(x, eps, np.eye(4))
    err = norm(J - J_FD_, ord ="fro") / norm(J, ord="fro")
    err_list.append(err)

# %%
print("Error_list is: \n", err_list)
# plot error
plt.loglog(eps_array, err_list, '-o')
plt.xlabel('eps')
plt.ylabel('Frobenius norm error')
plt.title('Error of Jacobian')
plt.grid(True)
plt.show()

# %%
def PlotConfig_up(theta_array, ell_array, ax, label):
    """
    theta_array: array of angles (which corresponds to N segments); N-1 nodes
    """
    # plot the configuration
    x = np.zeros(theta_array.shape[0]+1)
    y = np.zeros(theta_array.shape[0]+1)

    x[0] = 0
    y[0] = 0

    for i in range(theta_array.shape[0]):
        x[i+1] = x[i] + ell_array[i]*np.cos(theta_array[i])
        y[i+1] = y[i] - ell_array[i]*np.sin(theta_array[i])
    
    # ax.plot(x,y, x, y, 'o-', label=label)
    ax.plot(x,y, 'o-', label=label)
    return

# %%
# AD Jacobian
# Redefine variables and initial guess
# definition of variables and initial guess
P = tf.Variable([0,-16, -20], dtype=tf.float64)
ell = tf.Variable([4.0,6.0,5.0], dtype=tf.float64)
a = tf.Variable(6.0, dtype=tf.float64)
d_vertical = tf.Variable(3, dtype=tf.float64)

# We concatenate T, Theta 1, Theta 2... 
# x = tf.Variable([50, 50*np.pi/180, 50*np.pi/180/2, -50*np.pi/180], dtype=tf.float64)
x = tf.Variable([15, 0.45, 0.1,-0.2], dtype=tf.float64)


# plot the first
fig = plt.gcf()
# we are plotting two different configurations (initial and final and label them)
ax = fig.gca()
plt_1 = PlotConfig_up(x[1:], ell, ax, label='Initial Config')

# param for Newton's method
err_tol = 1e-12
err = 1
max_iter = 100

# initialize
iter = 0
while (err > err_tol) & (iter < max_iter):
    iter += 1
    print("Iter %d: " % iter)

    # calculate the jacobian matrix with autodiff
    start = time.time()

    with tf.GradientTape() as tape:
        # forward pass
        # Resid = -Feval(x, ell, a, P, d_vertical)
        # use assign_sub to update Resid in place
        Resid = Feval(x, ell, a, P, d_vertical)
    # get the gradient of F with respect to x
    J = tape.jacobian(Resid, x)
    # print("J is: \n", J)

    end = time.time()
    print("Time for autodiff: ", end-start)

    if iter == 1:
        # Resid0 = Resid
        Resid0 = Resid
    
    err = tf.norm(Resid)/tf.norm(Resid0)

    # Update x using the Newton-Raphson method
    # use x.assign_sub to update x in place
    x.assign_add(tf.squeeze(tf.linalg.solve(J, tf.expand_dims(-Resid, 1))))
    # print("x is: \n", x)

    if iter == max_iter:
        print("Maximum number of iterations reached")
        print("x is: \n", x)
        print("J is: \n", J)
        print("Residual is: \n", Resid)
        print("Error is: \n", err)
    if err < err_tol:
        print("Converged to tolerance")
        print("x is: \n", x)
        print("J is: \n", J)
        print("Residual is: \n", Resid)
        print("Error is: \n", err)
        break

# plot the final configuration
plt_2 = PlotConfig_up(x[1:], ell, ax, label='Final Config')
plt.title('Initial and final Config after %d Iter with AD' % iter)
plt.grid(True)
plt.xlabel('x(m)')
plt.ylabel('y(m)')
plt.legend()
plt.show()

# %%
# Redefine variables and initial guess
# definition of variables and initial guess
P = tf.Variable([0,-16, -20], dtype=tf.float64)
ell = tf.Variable([4.0,6.0,5.0], dtype=tf.float64)
a = tf.Variable(6.0, dtype=tf.float64)
d_vertical = tf.Variable(3, dtype=tf.float64)

# We concatenate T, Theta 1, Theta 2... 
# x = tf.Variable([50, 50*np.pi/180, 50*np.pi/180/2, -50*np.pi/180], dtype=tf.float64)
x = tf.Variable([15, 0.45, 0.1,-0.2], dtype=tf.float64)

# implementation of Newton's method (With FD Jacobian
# plot the first
fig = plt.gcf()
# we are plotting two different configurations (initial and final and label them)
ax = fig.gca()
plt_1 = PlotConfig_up(x[1:], ell, ax, label='Initial Config')

# param for Newton's method
err_tol = np.float64(1e-12)
err = np.float64(1)
max_iter = 100
eps = np.float64(1e-8)
h = np.eye(4)

# initialize
iter = 0
fig = plt.gcf()
ax = fig.gca()

while (err > err_tol) & (iter < max_iter):
    # # plot the configuration
    # PlotConfig_up(x[1:], ell, ax)
    iter += 1
    print("Iter %d: " % iter)

    # forward pass (for evaluation)
    Resid = np.float64(Feval(x, ell, a, P, d_vertical))

    if iter == 1:
        Resid0 = Resid
    
    # calculate the error
    err = np.float64(norm(Resid))/np.float64(norm(Resid0))
    # err = norm(Resid)/norm(Resid0)
    print("Error is: ", err)

    # calculate the jacobian matrix with FD
    start = time.time()
    J = J_FD(x, eps, h)     # x = T,t1,t2,t3
    end = time.time()
    print("Time for FD: ", end-start)

    # solve the linear system
    delta_x = np.linalg.solve(J, -Resid)

    # update x
    T = x[0] + delta_x[0]
    theta = x[1:] + delta_x[1:]
    x = tf.Variable(np.concatenate(([T], theta)), dtype=tf.float64)

    # print("x is: \n", x)
    # print("F is: \n", Resid)
    # print("J is: \n", J)

    if iter == max_iter:
        print("Maximum number of iterations reached")
        print("x is: \n", x)
        print("Residual is: \n", Resid)
        print("Error is: \n", err)
    if err < err_tol:
        print("Converged to tolerance")
        print("x is: \n", x)
        print("Residual is: \n", Resid)
        print("Error is: \n", err)
        break

# plot the final configuration
plt_2 = PlotConfig_up(x[1:], ell, ax, label='Final Config')
plt.title('Initial and final Config after %d Iter with FD' % iter)
plt.grid(True)
plt.xlabel('x(m)')
plt.ylabel('y(m)')
plt.legend()
plt.show()


