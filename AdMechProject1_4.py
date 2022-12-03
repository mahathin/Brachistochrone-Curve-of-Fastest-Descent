# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 13:00:48 2022

@author: Mahathi
"""

#importing required libraries
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

import time

"""
Let us consider 2 points A(x1, y1) and B(x2, y2).
We want to find a trajetory that minimises the time taken from A to B when released from rest.
Since frame are relative, let A be at the origin (0,0).
"""

#Final Point
x2 = 100
y2 = 0
X = np.sqrt(x2**2 + y2**2)

#Gravity influencing the trajectory
g = 9.8

"""
In order to obtain this path let us begin by considering N piecewise linear paths.
Let the projection of each piecewise linear segment be fixed. 
This allows us to compute the increment in x for each segment.
"""

N = 50 #N is the number of segments 
dx = x2/N #dx is the increment of x in each segment

"""
Given N segments, we have len(x) points that presently define our trajectory.
Let us initialise the positions of the len(x) points
"""

# intialising x coordinates of the len(x) points
# these coordinates will remain unchanged throughout
x = np.arange(0, X+dx, dx)

#let us define an initial curve
def path(x):
    return 0.001*x**2 - 0.1*x

#the time taken in each piecewise linear segment of a path is given by
def pathtime(dx_, y_):

    T = 0
    for i in range(1, len(y_)):
        dy_ = y_[i] - y_[i - 1]
        T += np.sqrt(2 * (1 + (dx_/dy_)**2)/(g)) * (abs(np.sqrt(abs(y_[i])) - np.sqrt(abs(y_[i - 1]))))
        
    return T

#let us now modify our initial path
def modpath(y_):
    y0 = np.zeros(len(y_))
    for i in range(len(y_)):
        c = rnd.uniform(0.999, 1.001)
        y0[i] = c*y_[i]
        y0[0] = 0
        y0[-1] = 0

    return y0


#comparing times of the modified path with that of the initial path and accepting or rejecting new path
def checking():
    y = path(x)
    graph.plot(x,y, ls='dotted', linewidth = 2, color = 'black', label = 'initial path')
    for i in range(0, 2000000):
        
        newy = modpath(y) 
        
        t1 = pathtime(dx,newy)
        t2 = pathtime(dx,y)
        
        if t1 <= t2:
            for i in range(len(x)):
                y[i] = newy[i]
        elif t1 > t2:
            for i in range(len(x)):
                y[i] = y[i]
    
    graph.plot(x, newy, ls='dashed', linewidth=2, color='orange', label = 'numerical fastest descent')
    pathtime(dx,newy)
    print("Computational time = ", pathtime(dx, newy))
    print("low", min(newy))
            
#Analytical solution which is a cycloid as derived from Euler-Lagrange equation
def analytical():
    theta = np.linspace(0, 2*np.pi, N)
    r = x2/(2*np.pi)
    ax =  r*(theta - np.sin(theta))
    ay = r*(np.cos(theta) - 1)
    graph.plot(ax, ay, linewidth=3, color='#653700', label = 'theoretical fastest descent')
    analytical_time = 2*np.pi*np.sqrt(r/g)
    print("Analytical time = ", analytical_time)
 
#plotting
img = plt.imread("Cliff.png")
fig, graph = plt.subplots()   
graph.imshow(img, extent=[-110, 190, -90, 60])
plt.title("Arriving at the Curve of Fastest Descent Numerically")
plt.xlabel("Horizontal distance(in metres)")
plt.ylabel("Vertical distance(in metres)")

 
#function call
starttime = time.time()
analytical()
checking()
runtime = (time.time() - starttime)
print(runtime)

plt.legend(fontsize='x-small')
plt.savefig("admech1")