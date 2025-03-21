# Carter Colton

#%% Problem 12.3 

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

# Physical constants
alpha = 0.1

# Make the grid
N = 500
L = 10
h = L/N
x = np.linspace(h/2,L-h/2,N)

# Initial Gaussian centered on the computing region
ymax = 2
y = ymax * np.exp(-(x-.5*L)**2)

# Time range
tau = 0.5
tfinal = 100
t = np.arange(0,tfinal,tau)

# Initialize the parts of the A and B matrices that
# do not depend on ybar and load them into At and Bt.
# Make them be sparse so the code will run fast.
At = np.zeros((N,N))
Bt = np.zeros((N,N))

# Function to wrap the column index
def jwrap(j):
    if (j < 0):
        return j + N
    if (j >= N):
        return j - N
    return j

# load the matrices with the terms that don't depend on ybar
h3 = h**3
for j in range(N):
    At[j,jwrap(j-1)] =-0.5*alpha/h3
    At[j,j]          = 0.5/tau + 1.5*alpha/h3
    At[j,jwrap(j+1)] = 0.5/tau - 1.5*alpha/h3
    At[j,jwrap(j+2)] = 0.5*alpha/h3

    Bt[j,jwrap(j-1)] = 0.5*alpha/h3
    Bt[j,j]          = 0.5/tau - 1.5*alpha/h3
    Bt[j,jwrap(j+1)] = 0.5/tau + 1.5*alpha/h3
    Bt[j,jwrap(j+2)] =-0.5*alpha/h3


plt.figure(1)
skip = 1
for n in range(len(t)):
    # Predictor step
    A = np.copy(At)
    B = np.copy(Bt)

    # load ybar, then add its terms to A and B
    ybar = np.copy(y)
    for j in range(N):
        tmp = 0.25*(ybar[jwrap(j+1)] + ybar[j])/h
        A[j,j]          = A[j,j] - tmp
        A[j,jwrap(j+1)] = A[j,jwrap(j+1)] + tmp
        B[j,j]          = B[j,j] + tmp
        B[j,jwrap(j+1)] = B[j,jwrap(j+1)] - tmp

    # do the predictor solve
    r = B@y
    yp = la.solve(A,r)

    # corrector step
    A = np.copy(At)
    B = np.copy(Bt)

    # average current and predicted y values to correct ybar
    ybar=.5*(y+yp)
    for j in range(N):
        tmp = 0.25*(ybar[jwrap(j+1)] + ybar[j])/h
        A[j,j]          = A[j,j] - tmp
        A[j,jwrap(j+1)] = A[j,jwrap(j+1)] + tmp
        B[j,j]          = B[j,j] + tmp
        B[j,jwrap(j+1)] = B[j,jwrap(j+1)] - tmp

    # do the final corrected solve
    r = B@y
    y = la.solve(A,r)

    if (n % skip == 0):
        plt.clf()
        plt.plot(x,y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('time={:1.3f}'.format(t[n]))
        plt.ylim(0,3)
        plt.pause(.1)
        
#%% Problem 12.3 b tau = 0.1

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

# Physical constants
alpha = 0.1

# Make the grid
N = 500
L = 10
h = L/N
x = np.linspace(h/2,L-h/2,N)

# Initial Gaussian centered on the computing region
ymax = 2
y = ymax * np.exp(-(x-.5*L)**2)

# Time range
tau = 0.1
tfinal = 10
t = np.arange(0,tfinal,tau)

# Initialize the parts of the A and B matrices that
# do not depend on ybar and load them into At and Bt.
# Make them be sparse so the code will run fast.
At = np.zeros((N,N))
Bt = np.zeros((N,N))

# Function to wrap the column index
def jwrap(j):
    if (j < 0):
        return j + N
    if (j >= N):
        return j - N
    return j

# load the matrices with the terms that don't depend on ybar
h3 = h**3
for j in range(N):
    At[j,jwrap(j-1)] =-0.5*alpha/h3
    At[j,j]          = 0.5/tau + 1.5*alpha/h3
    At[j,jwrap(j+1)] = 0.5/tau - 1.5*alpha/h3
    At[j,jwrap(j+2)] = 0.5*alpha/h3

    Bt[j,jwrap(j-1)] = 0.5*alpha/h3
    Bt[j,j]          = 0.5/tau - 1.5*alpha/h3
    Bt[j,jwrap(j+1)] = 0.5/tau + 1.5*alpha/h3
    Bt[j,jwrap(j+2)] =-0.5*alpha/h3


plt.figure(1)
skip = 1
for n in range(len(t)):
    # Predictor step
    A = np.copy(At)
    B = np.copy(Bt)

    # load ybar, then add its terms to A and B
    ybar = np.copy(y)
    for j in range(N):
        tmp = 0.25*(ybar[jwrap(j+1)] + ybar[j])/h
        A[j,j]          = A[j,j] - tmp
        A[j,jwrap(j+1)] = A[j,jwrap(j+1)] + tmp
        B[j,j]          = B[j,j] + tmp
        B[j,jwrap(j+1)] = B[j,jwrap(j+1)] - tmp

    # do the predictor solve
    r = B@y
    yp = la.solve(A,r)

    # corrector step
    A = np.copy(At)
    B = np.copy(Bt)

    # average current and predicted y values to correct ybar
    ybar=.5*(y+yp)
    for j in range(N):
        tmp = 0.25*(ybar[jwrap(j+1)] + ybar[j])/h
        A[j,j]          = A[j,j] - tmp
        A[j,jwrap(j+1)] = A[j,jwrap(j+1)] + tmp
        B[j,j]          = B[j,j] + tmp
        B[j,jwrap(j+1)] = B[j,jwrap(j+1)] - tmp

    # do the final corrected solve
    r = B@y
    y = la.solve(A,r)

    if (n % skip == 0):
        plt.clf()
        plt.plot(x,y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('time={:1.3f},tau={:1.3f}'.format(t[n],tau))
        plt.ylim(0,3)
        plt.pause(.1)
        
#%% tau = 0.02

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

# Physical constants
alpha = 0.1

# Make the grid
N = 500
L = 10
h = L/N
x = np.linspace(h/2,L-h/2,N)

# Initial Gaussian centered on the computing region
ymax = 2
y = ymax * np.exp(-(x-.5*L)**2)

# Time range
tau = 0.02
tfinal = 10
t = np.arange(0,tfinal,tau)

# Initialize the parts of the A and B matrices that
# do not depend on ybar and load them into At and Bt.
# Make them be sparse so the code will run fast.
At = np.zeros((N,N))
Bt = np.zeros((N,N))

# Function to wrap the column index
def jwrap(j):
    if (j < 0):
        return j + N
    if (j >= N):
        return j - N
    return j

# load the matrices with the terms that don't depend on ybar
h3 = h**3
for j in range(N):
    At[j,jwrap(j-1)] =-0.5*alpha/h3
    At[j,j]          = 0.5/tau + 1.5*alpha/h3
    At[j,jwrap(j+1)] = 0.5/tau - 1.5*alpha/h3
    At[j,jwrap(j+2)] = 0.5*alpha/h3

    Bt[j,jwrap(j-1)] = 0.5*alpha/h3
    Bt[j,j]          = 0.5/tau - 1.5*alpha/h3
    Bt[j,jwrap(j+1)] = 0.5/tau + 1.5*alpha/h3
    Bt[j,jwrap(j+2)] =-0.5*alpha/h3


plt.figure(1)
skip = 10
for n in range(len(t)):
    # Predictor step
    A = np.copy(At)
    B = np.copy(Bt)

    # load ybar, then add its terms to A and B
    ybar = np.copy(y)
    for j in range(N):
        tmp = 0.25*(ybar[jwrap(j+1)] + ybar[j])/h
        A[j,j]          = A[j,j] - tmp
        A[j,jwrap(j+1)] = A[j,jwrap(j+1)] + tmp
        B[j,j]          = B[j,j] + tmp
        B[j,jwrap(j+1)] = B[j,jwrap(j+1)] - tmp

    # do the predictor solve
    r = B@y
    yp = la.solve(A,r)

    # corrector step
    A = np.copy(At)
    B = np.copy(Bt)

    # average current and predicted y values to correct ybar
    ybar=.5*(y+yp)
    for j in range(N):
        tmp = 0.25*(ybar[jwrap(j+1)] + ybar[j])/h
        A[j,j]          = A[j,j] - tmp
        A[j,jwrap(j+1)] = A[j,jwrap(j+1)] + tmp
        B[j,j]          = B[j,j] + tmp
        B[j,jwrap(j+1)] = B[j,jwrap(j+1)] - tmp

    # do the final corrected solve
    r = B@y
    y = la.solve(A,r)

    if (n % skip == 0):
        plt.clf()
        plt.plot(x,y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('time={:1.3f},tau={:1.3f}'.format(t[n],tau))
        plt.ylim(0,3)
        plt.pause(.1)
        
#%% Problem 12.4

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

# Physical constants
alpha = 0.1

# Make the grid
N = 500
L = 10
h = L/N
x = np.linspace(h/2,L-h/2,N)

# Initial Gaussian centered on the computing region
ymax = 0.001
y = ymax * np.exp(-(x-.5*L)**2)

# Time range
tau = 0.1
tfinal = 10
t = np.arange(0,tfinal,tau)

# Initialize the parts of the A and B matrices that
# do not depend on ybar and load them into At and Bt.
# Make them be sparse so the code will run fast.
At = np.zeros((N,N))
Bt = np.zeros((N,N))

# Function to wrap the column index
def jwrap(j):
    if (j < 0):
        return j + N
    if (j >= N):
        return j - N
    return j

# load the matrices with the terms that don't depend on ybar
h3 = h**3
for j in range(N):
    At[j,jwrap(j-1)] =-0.5*alpha/h3
    At[j,j]          = 0.5/tau + 1.5*alpha/h3
    At[j,jwrap(j+1)] = 0.5/tau - 1.5*alpha/h3
    At[j,jwrap(j+2)] = 0.5*alpha/h3

    Bt[j,jwrap(j-1)] = 0.5*alpha/h3
    Bt[j,j]          = 0.5/tau - 1.5*alpha/h3
    Bt[j,jwrap(j+1)] = 0.5/tau + 1.5*alpha/h3
    Bt[j,jwrap(j+2)] =-0.5*alpha/h3


plt.figure(1)
skip = 1
for n in range(len(t)):
    # Predictor step
    A = np.copy(At)
    B = np.copy(Bt)

    # load ybar, then add its terms to A and B
    ybar = np.copy(y)
    for j in range(N):
        tmp = 0.25*(ybar[jwrap(j+1)] + ybar[j])/h
        A[j,j]          = A[j,j] - tmp
        A[j,jwrap(j+1)] = A[j,jwrap(j+1)] + tmp
        B[j,j]          = B[j,j] + tmp
        B[j,jwrap(j+1)] = B[j,jwrap(j+1)] - tmp

    # do the predictor solve
    r = B@y
    yp = la.solve(A,r)

    # corrector step
    A = np.copy(At)
    B = np.copy(Bt)

    # average current and predicted y values to correct ybar
    ybar=.5*(y+yp)
    for j in range(N):
        tmp = 0.25*(ybar[jwrap(j+1)] + ybar[j])/h
        A[j,j]          = A[j,j] - tmp
        A[j,jwrap(j+1)] = A[j,jwrap(j+1)] + tmp
        B[j,j]          = B[j,j] + tmp
        B[j,jwrap(j+1)] = B[j,jwrap(j+1)] - tmp

    # do the final corrected solve
    r = B@y
    y = la.solve(A,r)

    if (n % skip == 0):
        plt.clf()
        plt.plot(x,y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('time={:1.3f}'.format(t[n]))
        plt.ylim(-5*10**(-4),10*10**(-4))
        plt.pause(.1)
        
#%% Problem 12.4 b

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

# Physical constants
alpha = 0.01

# Make the grid
N = 500
L = 10
h = L/N
x = np.linspace(h/2,L-h/2,N)

# Initial Gaussian centered on the computing region
ymax = 2
y = ymax * np.exp(-(x-.5*L)**2)

# Time range
tau = 0.01
tfinal = 10
t = np.arange(0,tfinal,tau)

# Initialize the parts of the A and B matrices that
# do not depend on ybar and load them into At and Bt.
# Make them be sparse so the code will run fast.
At = np.zeros((N,N))
Bt = np.zeros((N,N))

# Function to wrap the column index
def jwrap(j):
    if (j < 0):
        return j + N
    if (j >= N):
        return j - N
    return j

# load the matrices with the terms that don't depend on ybar
h3 = h**3
for j in range(N):
    At[j,jwrap(j-1)] =-0.5*alpha/h3
    At[j,j]          = 0.5/tau + 1.5*alpha/h3
    At[j,jwrap(j+1)] = 0.5/tau - 1.5*alpha/h3
    At[j,jwrap(j+2)] = 0.5*alpha/h3

    Bt[j,jwrap(j-1)] = 0.5*alpha/h3
    Bt[j,j]          = 0.5/tau - 1.5*alpha/h3
    Bt[j,jwrap(j+1)] = 0.5/tau + 1.5*alpha/h3
    Bt[j,jwrap(j+2)] =-0.5*alpha/h3


plt.figure(1)
skip = 1
for n in range(len(t)):
    # Predictor step
    A = np.copy(At)
    B = np.copy(Bt)

    # load ybar, then add its terms to A and B
    ybar = np.copy(y)
    for j in range(N):
        tmp = 0.25*(ybar[jwrap(j+1)] + ybar[j])/h
        A[j,j]          = A[j,j] - tmp
        A[j,jwrap(j+1)] = A[j,jwrap(j+1)] + tmp
        B[j,j]          = B[j,j] + tmp
        B[j,jwrap(j+1)] = B[j,jwrap(j+1)] - tmp

    # do the predictor solve
    r = B@y
    yp = la.solve(A,r)

    # corrector step
    A = np.copy(At)
    B = np.copy(Bt)

    # average current and predicted y values to correct ybar
    ybar=.5*(y+yp)
    for j in range(N):
        tmp = 0.25*(ybar[jwrap(j+1)] + ybar[j])/h
        A[j,j]          = A[j,j] - tmp
        A[j,jwrap(j+1)] = A[j,jwrap(j+1)] + tmp
        B[j,j]          = B[j,j] + tmp
        B[j,jwrap(j+1)] = B[j,jwrap(j+1)] - tmp

    # do the final corrected solve
    r = B@y
    y = la.solve(A,r)

    if (n % skip == 0):
        plt.clf()
        plt.plot(x,y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('time={:1.3f},tau={:1.3f}'.format(t[n],tau))
        plt.ylim(-1,3)
        plt.pause(.1)
        
#%% Problem 12.5

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

# Physical constants
alpha = 0.135

# Make the grid
N = 500
L = 10
h = L/N
x = np.linspace(h/2,L-h/2,N)
k = 1.1
x0 = L/2

# Time range
tau = 0.01
tfinal = 10
t = np.arange(0,tfinal,tau)

# Initial Gaussian centered on the computing region
y = (12*alpha*k**2)/(np.cosh(k*(x-x0-4*t[0]*alpha*k**2)))**2

# Initialize the parts of the A and B matrices that
# do not depend on ybar and load them into At and Bt.
# Make them be sparse so the code will run fast.
At = np.zeros((N,N))
Bt = np.zeros((N,N))

# Function to wrap the column index
def jwrap(j):
    if (j < 0):
        return j + N
    if (j >= N):
        return j - N
    return j

# load the matrices with the terms that don't depend on ybar
h3 = h**3
for j in range(N):
    At[j,jwrap(j-1)] =-0.5*alpha/h3
    At[j,j]          = 0.5/tau + 1.5*alpha/h3
    At[j,jwrap(j+1)] = 0.5/tau - 1.5*alpha/h3
    At[j,jwrap(j+2)] = 0.5*alpha/h3

    Bt[j,jwrap(j-1)] = 0.5*alpha/h3
    Bt[j,j]          = 0.5/tau - 1.5*alpha/h3
    Bt[j,jwrap(j+1)] = 0.5/tau + 1.5*alpha/h3
    Bt[j,jwrap(j+2)] =-0.5*alpha/h3


plt.figure(1)
skip = 10
for n in range(len(t)):
    # Predictor step
    A = np.copy(At)
    B = np.copy(Bt)

    # load ybar, then add its terms to A and B
    ybar = np.copy(y)
    for j in range(N):
        tmp = 0.25*(ybar[jwrap(j+1)] + ybar[j])/h
        A[j,j]          = A[j,j] - tmp
        A[j,jwrap(j+1)] = A[j,jwrap(j+1)] + tmp
        B[j,j]          = B[j,j] + tmp
        B[j,jwrap(j+1)] = B[j,jwrap(j+1)] - tmp

    # do the predictor solve
    r = B@y
    yp = la.solve(A,r)

    # corrector step
    A = np.copy(At)
    B = np.copy(Bt)

    # average current and predicted y values to correct ybar
    ybar=.5*(y+yp)
    for j in range(N):
        tmp = 0.25*(ybar[jwrap(j+1)] + ybar[j])/h
        A[j,j]          = A[j,j] - tmp
        A[j,jwrap(j+1)] = A[j,jwrap(j+1)] + tmp
        B[j,j]          = B[j,j] + tmp
        B[j,jwrap(j+1)] = B[j,jwrap(j+1)] - tmp

    # do the final corrected solve
    r = B@y
    y = la.solve(A,r)

    if (n % skip == 0):
        plt.clf()
        plt.plot(x,y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('time={:1.3f},tau={:1.3f},alpha={:1.3f}'.format(t[n],tau,alpha))
        plt.ylim(-1,3)
        plt.pause(.1)
        
#%% Problem 12.5 c

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

# Physical constants
alpha = 0.135

# Make the grid
N = 500
L = 10
h = L/N
x = np.linspace(h/2,L-h/2,N)
k = 1.1
x0 = L/2

# Time range
tau = 0.01
tfinal = 20
t = np.arange(0,tfinal,tau)

# Initial Gaussian centered on the computing region
y = (12*alpha*k**2)/(np.cosh(k*(x-x0-4*t[0]*alpha*k**2)))**2

# Initialize the parts of the A and B matrices that
# do not depend on ybar and load them into At and Bt.
# Make them be sparse so the code will run fast.
At = np.zeros((N,N))
Bt = np.zeros((N,N))

# Function to wrap the column index
def jwrap(j):
    if (j < 0):
        return j + N
    if (j >= N):
        return j - N
    return j

# load the matrices with the terms that don't depend on ybar
h3 = h**3
for j in range(N):
    At[j,jwrap(j-1)] =-0.5*alpha/h3
    At[j,j]          = 0.5/tau + 1.5*alpha/h3
    At[j,jwrap(j+1)] = 0.5/tau - 1.5*alpha/h3
    At[j,jwrap(j+2)] = 0.5*alpha/h3

    Bt[j,jwrap(j-1)] = 0.5*alpha/h3
    Bt[j,j]          = 0.5/tau - 1.5*alpha/h3
    Bt[j,jwrap(j+1)] = 0.5/tau + 1.5*alpha/h3
    Bt[j,jwrap(j+2)] =-0.5*alpha/h3


plt.figure(1)
skip = 30
for n in range(len(t)):
    # Predictor step
    A = np.copy(At)
    B = np.copy(Bt)

    # load ybar, then add its terms to A and B
    ybar = np.copy(y)
    for j in range(N):
        tmp = 0.25*(ybar[jwrap(j+1)] + ybar[j])/h
        A[j,j]          = A[j,j] - tmp
        A[j,jwrap(j+1)] = A[j,jwrap(j+1)] + tmp
        B[j,j]          = B[j,j] + tmp
        B[j,jwrap(j+1)] = B[j,jwrap(j+1)] - tmp

    # do the predictor solve
    r = B@y
    yp = la.solve(A,r)

    # corrector step
    A = np.copy(At)
    B = np.copy(Bt)

    # average current and predicted y values to correct ybar
    ybar=.5*(y+yp)
    for j in range(N):
        tmp = 0.25*(ybar[jwrap(j+1)] + ybar[j])/h
        A[j,j]          = A[j,j] - tmp
        A[j,jwrap(j+1)] = A[j,jwrap(j+1)] + tmp
        B[j,j]          = B[j,j] + tmp
        B[j,jwrap(j+1)] = B[j,jwrap(j+1)] - tmp

    # do the final corrected solve
    r = B@y
    y = la.solve(A,r)

    if (n % skip == 0):
        plt.clf()
        plt.plot(x,y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('time={:1.3f},tau={:1.3f}'.format(t[n],tau))
        plt.ylim(-1,3)
        plt.pause(.1)
print(10/15.3)
print(4*alpha*k**2)
#%% Problem 12.6

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

# Physical constants
alpha = 0.1

# Make the grid
N = 500
L = 10
h = L/N
x = np.linspace(h/2,L-h/2,N)
k = 1.5
x0 = 3*L/4

k1 = 2
x01 = L/4

# Time range
tau = 0.01
tfinal = 20
t = np.arange(0,tfinal,tau)

# Initial Gaussian centered on the computing region
y = (12*alpha*k**2)/(np.cosh(k*(x-x0-4*t[0]*alpha*k**2)))**2 + (12*alpha*k1**2)/(np.cosh(k1*(x-x01-4*t[0]*alpha*k1**2)))**2
y1 = (12*alpha*k1**2)/(np.cosh(k1*(x-x01-4*t[0]*alpha*k1**2)))**2
y2 = y+y1

# Initialize the parts of the A and B matrices that
# do not depend on ybar and load them into At and Bt.
# Make them be sparse so the code will run fast.
At = np.zeros((N,N))
Bt = np.zeros((N,N))

# Function to wrap the column index
def jwrap(j):
    if (j < 0):
        return j + N
    if (j >= N):
        return j - N
    return j

# load the matrices with the terms that don't depend on ybar
h3 = h**3
for j in range(N):
    At[j,jwrap(j-1)] =-0.5*alpha/h3
    At[j,j]          = 0.5/tau + 1.5*alpha/h3
    At[j,jwrap(j+1)] = 0.5/tau - 1.5*alpha/h3
    At[j,jwrap(j+2)] = 0.5*alpha/h3

    Bt[j,jwrap(j-1)] = 0.5*alpha/h3
    Bt[j,j]          = 0.5/tau - 1.5*alpha/h3
    Bt[j,jwrap(j+1)] = 0.5/tau + 1.5*alpha/h3
    Bt[j,jwrap(j+2)] =-0.5*alpha/h3


plt.figure(1)
skip = 10
for n in range(len(t)):
    # Predictor step
    A = np.copy(At)
    B = np.copy(Bt)
    
    A1 = np.copy(At)
    B1 = np.copy(Bt)

    # load ybar, then add its terms to A and B
    ybar = np.copy(y)
    for j in range(N):
        tmp = 0.25*(ybar[jwrap(j+1)] + ybar[j])/h
        A[j,j]          = A[j,j] - tmp
        A[j,jwrap(j+1)] = A[j,jwrap(j+1)] + tmp
        B[j,j]          = B[j,j] + tmp
        B[j,jwrap(j+1)] = B[j,jwrap(j+1)] - tmp
    
    ybar1 = np.copy(y1)
    for j in range(N):
        tmp1 = 0.25*(ybar1[jwrap(j+1)] + ybar1[j])/h
        A1[j,j]          = A1[j,j] - tmp1
        A1[j,jwrap(j+1)] = A1[j,jwrap(j+1)] + tmp1
        B1[j,j]          = B1[j,j] + tmp1
        B1[j,jwrap(j+1)] = B1[j,jwrap(j+1)] - tmp1

    # do the predictor solve
    r = B@y
    r1 = B1@y1
    
    yp = la.solve(A,r)
    yp1 = la.solve(A1,r1)

    # corrector step
    A = np.copy(At)
    B = np.copy(Bt)
    
    A1 = np.copy(At)
    B1 = np.copy(Bt)

    # average current and predicted y values to correct ybar
    ybar=.5*(y+yp)
    for j in range(N):
        tmp = 0.25*(ybar[jwrap(j+1)] + ybar[j])/h
        A[j,j]          = A[j,j] - tmp
        A[j,jwrap(j+1)] = A[j,jwrap(j+1)] + tmp
        B[j,j]          = B[j,j] + tmp
        B[j,jwrap(j+1)] = B[j,jwrap(j+1)] - tmp
        
    ybar1=.5*(y1+yp1)
    for j in range(N):
        tmp1 = 0.25*(ybar1[jwrap(j+1)] + ybar1[j])/h
        A1[j,j]          = A1[j,j] - tmp1
        A1[j,jwrap(j+1)] = A1[j,jwrap(j+1)] + tmp1
        B1[j,j]          = B1[j,j] + tmp1
        B1[j,jwrap(j+1)] = B1[j,jwrap(j+1)] - tmp1

    # do the final corrected solve
    r = B@y
    y = la.solve(A,r)
    
    r1 = B1@y1
    y1 = la.solve(A1,r1)
    

    if (n % skip == 0):
        plt.clf()
        plt.plot(x,y,'r')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('time={:1.3f},tau={:1.3f}'.format(t[n],tau))
        plt.ylim(-1,3)
        plt.pause(.1)

