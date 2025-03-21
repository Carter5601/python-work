# Carter Colton

#%% Problem 11.1

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=10;
N=400 # the number of grid points
a=0
b=L
tau=0.001
c=1;
d=0;
gamma=0.1;
kappa=1;
M=1;
kb=1;


x,h = np.linspace(a,b,N+1,retstep = True) # make the grid

v = np.zeros((N+1,N+1));
rho=np.zeros(N+1);

vp = np.zeros((N+1,N+1));
rhop = np.zeros(N+1);

T = 1 + np.exp(-200*(x/L - 1/2)**2)
T[0] = 1;

F = (gamma-1)*M*kappa/(kb)
#D1 = (v[c,c] + v[c+1,c])/(4*h) + (2*F)/((p[c] + p[c+1])*h**2)
#D2 = -(gamma-1)*(v[c,c+1] + v[c+1,c+1] - v[c,c-1] - v[c+1,c-1])/(4*h) - (2*F)/((p[c] + p[c+1])*h**2)
#D3 = -(v[c,c] + v[c+1,c])/(4*h) + (2*F)/((p[c] + p[c+1])*h**2)

    
A = np.zeros((N+1,N+1),dtype=np.complex_) # load a matrix A with zeros

j=1

for i in A[1:N,0:N]: # build our matrix
    D1 = (v[j] + vp[j])/(4*h) + 2*F/(rho[j]+rhop[j])/h**2
    D2 = (-(gamma-1) * (v[j+1] + vp[j+1] - v[j-1] - vp[j-1] )/(4*h)
       - 4*F/(rho[j] + rhop[j])/h**2 )
    D3 = -(v[j] + vp[j])/(4*h) + 2*F/(rho[j]+rhop[j])/h**2
    A[c,c-1] = -0.5*D1
    A[c,c] = 1/tau - 0.5*D2
    A[c,c+1] = -0.5*D3
    j = j+1

b = np.zeros((N+1,N+1),dtype=np.complex_) # load b with zeros

for i in b[1:N,0:N]: # build our matrix
    D1 = (v[j] + vp[j])/(4*h) + 2*F/(rho[j]+rhop[j])/h**2
    D2 = (-(gamma-1) * (v[j+1] + vp[j+1] - v[j-1] - vp[j-1] )/(4*h)
      - 4*F/(rho[j] + rhop[j])/h**2 )
    D3 = -(v[j] + vp[j])/(4*h) + 2*F/(rho[j]+rhop[j])/h**2
    b[j,j-1] = 0.5*D1
    b[j,j] = 1/tau + 0.5*D2
    b[j,j+1] = 0.5*D3
    

        
#%% Problem 11.2

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps

L=10;
N=400 # the number of grid points
a=0
b=L
tau=0.001
c=1;
d=0;
gamma=0.1;
kappa=1;
M=1;
v=np.zeros(N+1);
kB=1;
mu=1;


x,h = np.linspace(a,b,N+1,retstep = True) # make the grid
vavg = np.zeros(N+1);

v = 1 + np.exp(-200*(x/L - 1/2)**2)
v[0] = 1;

E0 = np.zeros_like(v)
Tp = np.zeros_like(v)
vbar = np.zeros_like(v)
vbarp = np.zeros_like(v)

#E0 = -(kb/(M*(p[c]+p[c+1]))) * (((p[c]+p[c+1])*(T[c,c+1]*T[c+1,c+1]) - (p[c]+p[c+1])*(T[c,c-1]*T[c+1,c-1]))/(8*h))
#E1 = (vavg[c] + vavg[c+1])/(4*h) + (4*mu)/(3*(p[c]+p[c+1])*h**2)
#E2 = -(8*mu)/(3*(p[c]+p[c+1])*h**2)
#E3 = -(vavg[c] + vavg[c+1])/(4*h) + (4*mu)/(3*(p[c]+p[c+1])*h**2)

    
A = np.zeros((N+1,N+1),dtype=np.complex_) # load a matrix A with zeros

j = 1

for i in A[1:N,0:N]: # build our matrix
    E0[j] = (-kB/4/h/M/(rho[j]+rhop[j]) *
              ( (rho[j+1] + rhop[j+1]) * (T[j+1] + Tp[j+1])
               - (rho[j-1] + rhop[j-1]) * (T[j-1] + Tp[j-1])) )
    E1 = (vbar[j] + vbarp[j])/(4*h)+8*mu/3/h**2/(rho[j]+rhop[j])
    E2 =-16*mu/3/h**2/(rho[j]+rhop[j])
    E3 =-(vbar[j] + vbarp[j])/(4*h) +8*mu/3/h**2/(rho[j]+rhop[j])
    A[j,j-1] = -0.5*E1
    A[j,j] = 1/tau - 0.5*E2
    A[j,j+1] = -0.5*E3
    
    j = j + 1

b = np.zeros((N+1,N+1),dtype=np.complex_) # load b with zeros

for i in b[1:N,0:N]: # build our matrix
    E0[j] = (-kB/4/h/M/(rho[j]+rhop[j]) *
            ( (rho[j+1] + rhop[j+1]) * (T[j+1] + Tp[j+1])
             - (rho[j-1] + rhop[j-1]) * (T[j-1] + Tp[j-1])) )
    E1 = (vbar[j] + vbarp[j])/(4*h)+8*mu/3/h**2/(rho[j]+rhop[j])
    E2 =-16*mu/3/h**2/(rho[j]+rhop[j])
    E3 =-(vbar[j] + vbarp[j])/(4*h) +8*mu/3/h**2/(rho[j]+rhop[j])
    b[j,j-1] = 0.5*E1
    b[j,j] = 1/tau + 0.5*E2
    b[j,j+1] = 0.5*E3
        
#%% Problem 11.3

import Lab11Funcs as S
import matplotlib.pyplot as plt
import numpy as np
# System Parameters
L = 10.0 # Length of tube
T0 = 293. # Ambient temperature
rho0 = 1.3 # static density (sea level)
# speed of sound
c = np.sqrt(S.gamma * S.kB * T0 / S.M)
# cell-center grid with ghost points
N = 100
h = L/N
x = np.linspace(-.5*h,L+.5*h,N+2)
# initial distributions

rho = rho0 * np.ones_like(x)
T = T0 * np.ones_like(x)
v = np.exp(-200*(x/L-0.5)**2) * c/100
tau = 1e-4
tfinal = 0.1
t = np.arange(0,tfinal,tau)
skip = 5 #input(' Steps to skip between plots - ')
for n in range(len(t)):
# Plot the current values before stepping
    if n % skip == 0:
        plt.clf()
        plt.subplot(3,1,1)
        plt.plot(x,rho)
        plt.ylabel('rho')
        plt.ylim(1.28, 1.32)
        plt.title('time={:1.3f}'.format(t[n]))
        plt.subplot(3,1,2)
        plt.plot(x,T)
        plt.ylabel('T')
        plt.ylim(292,294)
        plt.subplot(3,1,3)
        plt.plot(x,v)
        plt.ylabel('v')
        plt.ylim(-4,4)
        plt.xlabel('x')
        plt.pause(0.1)
    # 1. Predictor step for rho
    rhop = S.Srho(rho,v,v,tau,h)
    # 2. Predictor step for T
    Tp = S.ST(T,v,v,rho,rhop,tau,h)
    # 3. Predictor step for v
    vp = S.Sv(v,v,v,rho,rhop,T,Tp,tau,h)
    # 4. Corrector step for rho
    rhop = S.Srho(rho,v,vp,tau,h)
    # 5. Corrector step for T
    
    Tp = S.ST(T,v,vp,rho,rhop,tau,h)
    # 6. Corrector step for v
    v = S.Sv(v,v,vp,rho,rhop,T,Tp,tau,h)
    # Now put rho and T at the same time-level as v
    rho = rhop
    T = Tp
    
#%% Problem 11.4

import Lab11Funcs as S
import matplotlib.pyplot as plt
import numpy as np
# System Parameters
L = 10.0 # Length of tube
T0 = 293 # Ambient temperature
rho0 = 1.3 # static density (sea level)
# speed of sound
c = np.sqrt(S.gamma * S.kB * T0 / S.M)
# cell-center grid with ghost points
N = 100
h = L/N
x = np.linspace(-.5*h,L+.5*h,N+2)
print('This is the speed of the density waves')
print(5/.015)
# initial distributions

rho = rho0 * np.ones_like(x)
T = T0 * np.ones_like(x)
v0 = c/100;
v = v0*np.exp(-200*(x/L-0.5)**2)
tau = 1e-4
tfinal = 0.1
t = np.arange(0,tfinal,tau)
skip = 5 #input(' Steps to skip between plots - ')
for n in range(len(t)):
# Plot the current values before stepping
    if n % skip == 0:
        plt.clf()
        plt.subplot(3,1,1)
        plt.plot(x,rho)
        plt.ylabel('rho')
        plt.ylim(1.28, 1.32)
        plt.title('time={:1.3f}'.format(t[n]))
        plt.subplot(3,1,2)
        plt.plot(x,T)
        plt.ylabel('T')
        plt.ylim(292,294)
        plt.subplot(3,1,3)
        plt.plot(x,v)
        plt.ylabel('v')
        plt.ylim(-4,4)
        plt.xlabel('x')
        plt.pause(0.1)
    # 1. Predictor step for rho
    rhop = S.Srho(rho,v,v,tau,h)
    # 2. Predictor step for T
    Tp = S.ST(T,v,v,rho,rhop,tau,h)
    # 3. Predictor step for v
    vp = S.Sv(v,v,v,rho,rhop,T,Tp,tau,h)
    # 4. Corrector step for rho
    rhop = S.Srho(rho,v,vp,tau,h)
    # 5. Corrector step for T
    
    Tp = S.ST(T,v,vp,rho,rhop,tau,h)
    # 6. Corrector step for v
    v = S.Sv(v,v,vp,rho,rhop,T,Tp,tau,h)
    # Now put rho and T at the same time-level as v
    rho = rhop
    T = Tp
    
#%% Problem 11.4

import Lab11Funcs as S
import matplotlib.pyplot as plt
import numpy as np
# System Parameters
L = 10.0 # Length of tube
T0 = 293 # Ambient temperature
rho0 = 1.3 # static density (sea level)
# speed of sound
c = np.sqrt(S.gamma * S.kB * T0 / S.M)
# cell-center grid with ghost points
N = 50
h = L/N
x = np.linspace(-.5*h,L+.5*h,N+2)
# initial distributions

rho = rho0 * np.ones_like(x)
T = T0 * np.ones_like(x)
v0 = c/100;
v = v0*np.exp(-200*(x/L-0.5)**2)
tau = h/(2*c)
tfinal = 0.1
t = np.arange(0,tfinal,tau)
skip = 5 #input(' Steps to skip between plots - ')
for n in range(len(t)):
# Plot the current values before stepping
    if n % skip == 0:
        plt.clf()
        plt.subplot(3,1,1)
        plt.plot(x,rho)
        plt.ylabel('rho')
        plt.ylim(1.28, 1.32)
        plt.title('time={:1.3f}'.format(t[n]))
        plt.subplot(3,1,2)
        plt.plot(x,T)
        plt.ylabel('T')
        plt.ylim(292,294)
        plt.subplot(3,1,3)
        plt.plot(x,v)
        plt.ylabel('v')
        plt.ylim(-4,4)
        plt.xlabel('x')
        plt.pause(0.1)
    # 1. Predictor step for rho
    rhop = S.Srho(rho,v,v,tau,h)
    # 2. Predictor step for T
    Tp = S.ST(T,v,v,rho,rhop,tau,h)
    # 3. Predictor step for v
    vp = S.Sv(v,v,v,rho,rhop,T,Tp,tau,h)
    # 4. Corrector step for rho
    rhop = S.Srho(rho,v,vp,tau,h)
    # 5. Corrector step for T
    
    Tp = S.ST(T,v,vp,rho,rhop,tau,h)
    # 6. Corrector step for v
    v = S.Sv(v,v,vp,rho,rhop,T,Tp,tau,h)
    # Now put rho and T at the same time-level as v
    rho = rhop
    T = Tp
    
#%% Problem 11.4 c Make mu and kappa 0

import Lab11Funcs4d as S
import matplotlib.pyplot as plt
import numpy as np
# System Parameters
L = 10.0 # Length of tube
T0 = 293 # Ambient temperature
rho0 = 1.3 # static density (sea level)
# speed of sound
c = np.sqrt(S.gamma * S.kB * T0 / S.M)
# cell-center grid with ghost points
N = 100
h = L/N
x = np.linspace(-.5*h,L+.5*h,N+2)
# initial distributions

rho = rho0 * np.ones_like(x)
T = T0 * np.ones_like(x)
v0 = c/10;
v = v0*np.exp(-200*(x/L-0.5)**2)
tau = 1e-4
tfinal = 0.1
t = np.arange(0,tfinal,tau)
skip = 5 #input(' Steps to skip between plots - ')
for n in range(len(t)):
# Plot the current values before stepping
    if n % skip == 0:
        plt.clf()
        plt.subplot(3,1,1)
        plt.plot(x,rho)
        plt.ylabel('rho')
        plt.ylim(1.28, 1.32)
        plt.title('time={:1.3f}'.format(t[n]))
        plt.subplot(3,1,2)
        plt.plot(x,T)
        plt.ylabel('T')
        plt.ylim(292,294)
        plt.subplot(3,1,3)
        plt.plot(x,v)
        plt.ylabel('v')
        plt.ylim(-4,4)
        plt.xlabel('x')
        plt.pause(0.1)
    # 1. Predictor step for rho
    rhop = S.Srho(rho,v,v,tau,h)
    # 2. Predictor step for T
    Tp = S.ST(T,v,v,rho,rhop,tau,h)
    # 3. Predictor step for v
    vp = S.Sv(v,v,v,rho,rhop,T,Tp,tau,h)
    # 4. Corrector step for rho
    rhop = S.Srho(rho,v,vp,tau,h)
    # 5. Corrector step for T
    
    Tp = S.ST(T,v,vp,rho,rhop,tau,h)
    # 6. Corrector step for v
    v = S.Sv(v,v,vp,rho,rhop,T,Tp,tau,h)
    # Now put rho and T at the same time-level as v
    rho = rhop
    T = Tp

#%% Problem 11.4 d Make mu and kappa not zero

import Lab11Funcs as S
import matplotlib.pyplot as plt
import numpy as np
# System Parameters
L = 10.0 # Length of tube
T0 = 293 # Ambient temperature
rho0 = 1.3 # static density (sea level)
# speed of sound
c = np.sqrt(S.gamma * S.kB * T0 / S.M)
# cell-center grid with ghost points
N = 100
h = L/N
x = np.linspace(-.5*h,L+.5*h,N+2)
# initial distributions

rho = rho0 * np.ones_like(x)
T = T0 * np.ones_like(x)
v0 = c/10;
v = v0*np.exp(-200*(x/L-0.5)**2)
tau = 1e-4
tfinal = 0.1
t = np.arange(0,tfinal,tau)
skip = 5 #input(' Steps to skip between plots - ')
for n in range(len(t)):
# Plot the current values before stepping
    if n % skip == 0:
        plt.clf()
        plt.subplot(3,1,1)
        plt.plot(x,rho)
        plt.ylabel('rho')
        plt.ylim(1.28, 1.32)
        plt.title('time={:1.3f}'.format(t[n]))
        plt.subplot(3,1,2)
        plt.plot(x,T)
        plt.ylabel('T')
        plt.ylim(292,294)
        plt.subplot(3,1,3)
        plt.plot(x,v)
        plt.ylabel('v')
        plt.ylim(-4,4)
        plt.xlabel('x')
        plt.pause(0.1)
    # 1. Predictor step for rho
    rhop = S.Srho(rho,v,v,tau,h)
    # 2. Predictor step for T
    Tp = S.ST(T,v,v,rho,rhop,tau,h)
    # 3. Predictor step for v
    vp = S.Sv(v,v,v,rho,rhop,T,Tp,tau,h)
    # 4. Corrector step for rho
    rhop = S.Srho(rho,v,vp,tau,h)
    # 5. Corrector step for T
    
    Tp = S.ST(T,v,vp,rho,rhop,tau,h)
    # 6. Corrector step for v
    v = S.Sv(v,v,vp,rho,rhop,T,Tp,tau,h)
    # Now put rho and T at the same time-level as v
    rho = rhop
    T = Tp

