"""
===========================
The double pendulum problem
===========================

This animation illustrates the double pendulum problem.

Double pendulum formula translated from the C code at
http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

from collections import deque
import scipy
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin

import matplotlib.animation as animation

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
L = L1 + L2  # maximal length of the combined pendulum
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg
t_stop = 15  # how many seconds to simulate
history_len = 250  # how many trajectory points to display


def derivs(t, state):
    dydx = np.zeros_like(state)

    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
    dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                + M2 * G * sin(state[2]) * cos(delta)
                + M2 * L2 * state[3] * state[3] * sin(delta)
                - (M1+M2) * G * sin(state[0]))
               / den1)

    dydx[2] = state[3]

    den2 = (L2/L1) * den1
    dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                + (M1+M2) * G * sin(state[0]) * cos(delta)
                - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                - (M1+M2) * G * sin(state[2]))
               / den2)

    return dydx


# Define the RK4 integration function
def rk4(func, t, y, h):
    k1 = h * func(t, y)
    k2 = h * func(t + 0.5 * h, y + 0.5 * k1)
    k3 = h * func(t + 0.5 * h, y + 0.5 * k2)
    k4 = h * func(t + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# create a time array from 0..t_stop sampled at 0.02 second steps
dt = 0.01
t = np.arange(0, t_stop, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 120.0
w1 = 0.0
th2 = -10.0
w2 = 0.0

# initial state
state = np.radians([th1, w1, th2, w2])

# integrate the ODE using Euler's method

y = np.empty((len(t), 4))
y[0] = state
for i in range(1, len(t)):
    y[i] = y[i - 1] + derivs(t[i - 1], y[i - 1]) * dt

# A more accurate estimate could be obtained e.g. using scipy:
#
#y = scipy.integrate.solve_ivp(derivs, t[[0, -1]], state, t_eval=t).y.T

x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])

x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1

#####
ysci = scipy.integrate.solve_ivp(derivs, t[[0, -1]], state, t_eval=t).y.T
x1sci = L1*sin(ysci[:, 0])
y1sci = -L1*cos(ysci[:, 0])

x2sci = L2*sin(ysci[:, 2]) + x1sci
y2sci = -L2*cos(ysci[:, 2]) + y1sci

#####


# Integrate the ODE using RK4
yrk = np.empty((len(t), 4))
yrk[0] = state
for i in range(1, len(t)):
    yrk[i] = rk4(derivs, t[i - 1], yrk[i - 1], dt)

# Calculate the pendulum positions
xrk1 = L1*sin(yrk[:, 0])
yrk1 = -L1*cos(yrk[:, 0])

xrk2 = L2*sin(yrk[:, 2]) + xrk1
yrk2 = -L2*cos(yrk[:, 2]) + yrk1

######
fig = plt.figure(figsize=(5, 4))

ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
ax.set_aspect('equal')
ax.grid()


line, = ax.plot([], [], 'o-', lw=2, color='blue', label='Euler')
linesci, = ax.plot([],[],'o-',lw=2, color='red', label='SciPy')#####
linerk, = ax.plot([],[],'o-',lw=2, color='green', label='RK4')######
trace, = ax.plot([], [], '.-', lw=1, ms=2, color='blue')
tracesci, = ax.plot([], [], '.-', lw=1, ms=2, color='red')######
tracerk, = ax.plot([], [], '.-', lw=1, ms=2, color='green')######

time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)
history_xsci, history_ysci = deque(maxlen=history_len), deque(maxlen=history_len)
history_xrk, history_yrk = deque(maxlen=history_len), deque(maxlen=history_len)

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    thisxsci = [0, x1sci[i], x2sci[i]]#
    #thisysci = [0, x1sci[i], x2sci[i]]# to make pretty thing
    thisysci = [0, y1sci[i], y2sci[i]]

    thisxrk = [0, xrk1[i], xrk2[i]]
    thisyrk = [0, yrk1[i], yrk2[i]]

    if i == 0:
        history_x.clear()
        history_y.clear()
        history_xsci.clear()#
        history_ysci.clear()#
        history_xrk.clear()#
        history_yrk.clear()#

    history_x.appendleft(thisx[2])
    history_y.appendleft(thisy[2])
    #history_x.appendleft(thisxsci[2])
    #history_y.appendleft(thisysci[2])
    history_xsci.appendleft(thisxsci[2])#
    history_ysci.appendleft(thisysci[2])#
    history_xrk.appendleft(thisxrk[2])#
    history_yrk.appendleft(thisyrk[2])#

    line.set_data(thisx, thisy)
    #linesci.set_data(thisxsci, thisysci)#
    #trace.set_data(history_x, history_y)###this puts it in front of sci
    linesci.set_data(thisxsci, thisysci)#
    linerk.set_data(thisxrk, thisyrk)#
    tracesci.set_data(history_xsci, history_ysci)#
    trace.set_data(history_x, history_y)
    tracerk.set_data(history_xrk, history_yrk)#
    time_text.set_text(time_template % (i*dt))
    return line, trace, linesci, tracesci, linerk, tracerk, time_text

ax.legend()
'''
ani = animation.FuncAnimation(
    fig, animate, len(y), interval=dt*1000, blit=True)
plt.show()
'''
animation_writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Hayden Church'))
#ani.save("double_pendulum_animation.mp4", writer=animation_writer)

#animation_writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Hayden Church'))
ani = animation.FuncAnimation(
    fig, animate, len(y), interval=dt*1000, blit=True)

#plt.show()
ani.save("pend_presentation.mp4", writer=animation_writer)
#ani.save("double_pendulum_animation.mp4")
plt.show()
# Render the animation
#plt.show()