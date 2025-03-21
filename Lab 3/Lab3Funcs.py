# Carter Colton Functions

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

N=100;

def force(x):
    b = np.zeros(N+1) # load b with zeros
    b[0] = 0;
    b[N] = 0;
    n = 1
    f=0;

    for n in range(1,N): # right hand side
        b[n] = f
        n = n + 1
        if x[n]>=0.8 and x[n]<=1:
            f=-0.73
        else:
            f=0
    return b
            
def steadySol(f,h,w,T,mu):
    A = np.zeros((N+1,N+1)) # load a matrix A with zeros
    s = np.array([T/(h**2),-2*T/(h**2) + mu*w**2,T/(h**2)])
    c=1;
    d=0;

    A[0,0] = 1; # Set the endpoints
    A[N,N] = 1;

    for i in A[1:N,0:N]: # build our matrix
        A[c,d] = s[0]
        A[c,d+1] = s[1]
        A[c,d+2] = s[2]
        c = c + 1
        d = d + 1

    g = la.solve(A,f) # solve for y
    
    return g

