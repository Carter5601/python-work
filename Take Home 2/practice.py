# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:17:29 2023

@author: carte
"""

for i in A[0:N,0:N]: # build our matrix
    A[c,d] = 0
    A[c,c-1] = (1/(r[c]*h)) - (1/(2*h**2))
    A[c,c] = (rho*c)/(tau) + 1/(h**2)
    A[c,c+1] = -(1/(r[c]*h)) - (1/(2*h**2))
    c = c + 1
    d = d + 1


for i in b[0:N,0:N]: # build our matrix
    b[j,q]= 0
    b[j,j-1] = -1/(r[q]*h) + (1/(2*h**2))
    b[j,j] =  (rho*c)/(tau) - 1/(h**2)
    b[j,j+1] =  1/(r[q]*h) + (1/(2*h**2))
    j=j+1
    q=q+1