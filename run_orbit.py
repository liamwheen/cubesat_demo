#!/usr/bin/env python
from animation import Animate
from control import Controller
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

q = Quaternion(1,0,0,0)
bod_mass = 4 #kg
bod_width = 0.1 #metres
bod_depth = 0.1
bod_height = 0.3
body_cm = [0,0,0]
wheel_mass = 0.0166
wheel_rad = 0.011515
wheel_height = 0.02
wheel_cms = [[0,0,0]]*3
#wheel_cms = [[0.02,0.1,0.04],[-0.01,0.01,0.04],[-0.01,-0.01,0.04]]
#wheel_axes = [[1,0,1],[0,1,1],[-1,0,1],[0,-1,1]]
wheel_axes = [[1,0,0],[0,1,0],[0,0,1]]

def latlon2euc(latlon, alt):
    lat,lon = latlon
    lat *= np.pi/180
    lon *= np.pi/180
    x = np.outer(np.cos(lon), np.cos(lat)).T*alt
    y = np.outer(np.sin(lon), np.cos(lat)).T*alt
    z = np.outer(np.ones(np.size(lon)), np.sin(lat)).T*alt
    if len(x)==1:
        x,y,z = *x[0],*y[0],*z[0]
    return np.array([x,y,z])

def calc_q_move(sat_pos):
    euc_vol_pos = latlon2euc(vol_pos, earth_rad)
    euc_sat_pos = latlon2euc(sat_pos, sat_alt)
    sat2vol = euc_vol_pos - euc_sat_pos
    sat2vol /= np.linalg.norm(sat2vol) #normalise
    rotated_z = q.rotate(np.array([0,0,1]))
    perp_vec = np.cross(rotated_z,sat2vol)
    angle = np.arccos(min(np.dot(sat2vol, rotated_z),1))
    q_move = Quaternion(axis=perp_vec, angle=angle)
    return q_move

body_params = {'mass':bod_mass, 'width':bod_width, 'depth':bod_depth,
        'height':bod_height, 'cm':body_cm}
wheel_params = {'mass':wheel_mass, 'radius':wheel_rad, 'height':wheel_height,
        'cm':wheel_cms, 'axis':wheel_axes}

earth_rad = 6371 #km
sat_alt = earth_rad+300

vol_pos = (-6,0)
sat_pos0 = (6, 0)
#number of iterations that the controller's cpu can run
att_update = 0.01
time_sf = 10
w0 = [0,0,0]
q0 = [1,0,0,0]
rate = 20 #frames/calculations per second
controller = Controller(rate, body_params, q0=[1,0,0,0], w0=[0,0,0])
anim_obj = Animate(controller, time_sf, real=False)
my_anim = anim_obj.make_animation()
plt.show()
