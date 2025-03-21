# Carter Colton

# This is all my work for Problem 1.1

x = 20
x=x+1

# Variables
a=2
b=4
c=a*b

a1=input('What is your age?')

# Integers 
A=20
B=15
C=a+b
D=A/B

i=A//B
r=A%B
e=A*B
f=C**4

# Float Variables
fp=20.
fn=20
fpn=float(fp)

fl=0.1
inte=3*fl
y=1.23e15

# Functions
print(abs(a))
print(divmod(r,a))
b%c
float(b)
print(int(fl))
print(round(fl))

# Boolean
q=True

# Strings
s='This is a string'
t="Don't worry"

# Formatting Printed Values
a=22
b=3.5
print('I am {:d} years old and my GPA is: {:5.2f}'. format(a,b))

#This style also works
joe_string="My GPA is {:5.2f} and I am {:d} years old."
print(joe_string.format(b,a))

# Functions and Libraries 
x=3.14
y=round(x)

x=3.14
y=2
b,c=divmod(x,y)

import numpy

x=numpy.pi #Get the value of pi
y=numpy.sin(x) #Find the sine of the pi radians
z=numpy.sqrt(x) #Take the sqaure root of pi

import numpy as np

x=np.sqrt(5.2) #Take the square root of 5.2
y=np.pi #Get the value of pi
z=np.sin(34) #Find the sine of 34 radians

import scipy.special as sps
a = 5.5
b = 6.2
c = np.pi/2
d = a * np.sin(b * c)
e = a * sps.j0(3.5 * np.pi/4)

#Numpy Arrays

s=((x-5)**3)
a = np.array([1,2,3,4,5,6,7,8,9,10])
b = np.arange(0,10,.1)
x = np.linspace(0,10,10)
x,dx = np.linspace(0,10,10,retstep = True) #Ask what step size results

x = np.array([2,3,5.2,2,6.7,3]) # Create the array x
y = np.array([4,8,9.8,2.1,8.2,4.5]) # Create the array y
c = x**2 # Square the elements of x
d = x + 3 # Add 3 to every element of x
e = x * 5 # Multipy every element of x by 5
f = x + y # Add the elements of x to the elements y
g = x * y # Multiply the elements of x by the elements of y

x = np.array([2,3,5.2,2,6.7,3])
y = x

x = np.array([2,3,5.2,2,6.7,3])
y = np.copy(x)

#Functions of Arrays
x = np.array([2,3,5.2,2,6.7])
c = np.sin(x)

#Accessing Elements of Arrays
a = np.array([1,2,3,4,5,6,7,8,9,10])
x = a[1]

b=a[1:4]
c=a[0:6:2]

b = a[-1]
c = a[1:-2]

b = a[1:]
c = a[:3]

#Making x-y Plots
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0,10,0.01)
y = x**2
plt.figure(1)
plt.plot(x,y,'ro')

#Culmination of it all
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,2.*np.pi,100)
y1 = np.sin(2*x)
y2 = np.cos(3*x)
plt.figure(1)
plt.plot(x,y1,'r.',x,y2,'b')
plt.legend(['Sine of x','Cosine of x'])
plt.xlabel('x')
plt.title('Two Trig Functions')
plt.figure(2)
plt.plot(y1,y2)
plt.title('Something Fancy')


