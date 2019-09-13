import numpy as np
from pyquaternion import Quaternion
from decimal import Decimal
from scipy.integrate import odeint

class Controller:

    def __init__(self, rate, body_params, q0=[1,0,0,0], w0=[0,0,0]):
        self.earth_rad = 1
        self.sat_alt = 0#self.earth_rad+300
        self.rate = rate
        self.delta = round(1/rate,2)
        self.max_speed =  0.06 #max rotation speed of body, rad/s
        self.q_cur = Quaternion(q0).unit
        self.w_cur = w0
        self.q_err_sum = np.array([0,0,0], 'f')
        self.kp,self.kd = 0.000002,0.004
        self.axis_err_vals = np.array([0,0,0])
        self.wheel_speeds = np.array([0,0,0])
        self.max_torq = np.array([2.4e-4, 2.4e-4, 2.54e-4])
        mass, width, depth, height = [body_params[key] for key in ['mass','width', 'depth', 'height']]
        A_bod = self.calc_cross_prod_mat(body_params['cm'])
        I_bod =  np.diag([1/12 * mass * (depth**2 + height**2),  1/12 *\
            mass * (depth**2 + height**2), 1/12 * mass * (depth**2 + width**2)])
        self.I_tot = I_bod + mass*A_bod
        self.t = 0
        
    def set_vol_pos(self, click_euc):
        self.vol_pos = np.array(click_euc)
        
    def calc_cross_prod_mat(self, r_cm):
        rx, ry, rz = r_cm
        return np.array([[ry**2+rz**2, -rx*ry, -rx*rz],
                         [-rx*ry, rx**2+rz**2, -ry*rz],
                         [-rx*rz, -ry*rz, rx**2+ry**2]])
    def calc_q_move(self):
        sat2vol = self.vol_pos
        #sat2vol /= np.linalg.norm(sat2vol) #normalise
        #print('sat2vol = ',sat2vol)
        #print('co sat2vol = ',sat2vol)
        rotated_z = self.q_cur.rotate(np.array([0,0,1]))
        #print('co rot_z = ',rotated_z)
        perp_vec = np.cross(rotated_z,sat2vol)
        #perp_vec /=  np.linalg.norm(perp_vec)
        angle = np.arccos(min(np.dot(sat2vol, rotated_z),1))
        if np.linalg.norm(perp_vec)==0: perp_vec = self.q_cur.rotate(np.array([1,0,0]))
        self.q_move = Quaternion(axis=perp_vec, angle=angle)
        
    def calc_w_tar(self):
        theta_tar = self.q_move.angle #angle to move
        #print('theta_tar: ',theta_tar)
        #coef = min(theta_tar/self.max_theta, 1) #severity of movement
        coef = np.tanh(theta_tar)
        #needs to be transformed into body coords as q_move is in inertial
        self.w_tar = coef*self.max_speed*self.q_cur.inverse.rotate(self.q_move.axis)

    def calc_torq(self):
        q_err = self.q_cur.inverse.rotate(self.q_move.vector)
        #q_err[np.where(abs(q_err)>0.05)] = 0.05*np.sign(q_err[np.where(abs(q_err)>0.05)])
        w_err = self.w_tar-self.w_cur
        w_err[2]=0
        torq = - self.kp*q_err - self.kd*w_err
        #input('~~~~~~~~~~~~\n')
        #print('q_err: ',q_err)
        #print('w_err: ',w_err)
        #torq = np.minimum(self.max_torq,torq)
        #torq = np.maximum(-self.max_torq,torq)
        #print(torq)
        self.wheel_speeds = np.vstack((self.wheel_speeds,
            2e3*np.array(self.w_cur)))
        return torq

    def solve_step(self, torq, q0, w0):
        sol = odeint(self.ddt,[*q0, *w0], [0,self.delta], (torq,))[-1]
        q, w_bod = sol[:4], sol[4:]
        q_quat = Quaternion(q)
        return q_quat.unit, w_bod

    def ddt(self, q_w, t, torq):
        q, w = q_w[:4], q_w[4:]
        q = Quaternion(q).unit
        q_dot = 0.5 * q * Quaternion([0, *w])
        w_dot = np.linalg.inv(self.I_tot)@(
                - np.cross(w,self.I_tot@w) - torq)
        return [*q_dot.elements.tolist(), *w_dot]

    def update(self):
        self.calc_q_move()
        self.calc_w_tar()
        self.torq = self.calc_torq()
        self.q_cur, self.w_cur = self.solve_step(self.torq,
                self.q_cur, self.w_cur)
        self.t+=self.delta
        return self.q_cur
