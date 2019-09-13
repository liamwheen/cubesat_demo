from collections import namedtuple
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from control import Controller
from matplotlib import rc
#rc('text', usetex=True)
#rc('font',size=6)

class Animate:

    def __init__(self, controller, time_sf, real=False):
        
        self.cid = 0
        self.click_count = 0
        self.controller = controller
        self.rate = controller.rate
        self.time_sf = time_sf
        self.iter = 0
        self.earth_rad = 1
        self.sat_pos = [0,0,0]
        self.sat_scale = 0.03#60/rescale
        self.vertices = self.sat_scale*np.reshape(np.mgrid[-1:2:2,-1:2:2,-3:4:6].T, (8,3))
        self.centre_vertices = self.vertices
        self.sat_axes = self.sat_scale*np.diag([2,2,6])
        self.sat_axes = self.sat_scale*np.array([0,0,30])
        self.centre_sat_axes = self.sat_axes
        self.orbit_fig = plt.figure(figsize=(9,9))
        self.orbit_ax = p3.Axes3D(self.orbit_fig)
        self.orbit_ax.set_xlim3d([-1,1])
        self.orbit_ax.set_ylim3d([-1,1])
        self.orbit_ax.set_zlim3d([-1,1])
        
        self.bod_ax = self.orbit_fig.add_subplot(221,projection='3d')
        for item in [self.orbit_fig, self.bod_ax]:
            item.patch.set_visible(False)
        self.bod_plot, = self.bod_ax.plot([],[])
        self.vertices_init = np.reshape(np.mgrid[-1:2:2,-1:2:2,-3:4:6].T, (8,3))
        self.edges = [(j,k) for j in range(8) for k in range(j,8)
                if
                abs(np.linalg.norm(self.vertices[j]-self.vertices[k])-2)<0.01 or
                abs(np.linalg.norm(self.vertices[j]-self.vertices[k])-6)<0.01]

        #error plot
        self.err_ax = self.orbit_fig.add_subplot(444)
        self.err_ax.set_ylabel('Error (Deg) ', rotation='horizontal')
        self.err_ax.yaxis.set_label_coords(1.1,1.1)
        self.err_ax.yaxis.tick_right()
        self.err_ax.set_xlabel('Time (s)', x=0.55)
        self.err_ax.grid(True,'major','y')
        self.err_plot, self.cont_err_plot = self.err_ax.plot([],[],[],[],alpha=0.6)
        self.err_vals = []
        self.err_ax.set_xticklabels([''])
        #self.err_ax.set_yticklabels([''])
        self.cont_err_vals = []

        #wheels
        self.wheel_speed_plot = self.orbit_fig.add_subplot(4,4,16)
        self.wheel_speed_plot.set_ylabel('Wheel Speeds (rpm)', rotation='horizontal')
        self.wheel_speed_plot.yaxis.set_label_coords(0.7,1.1)
        self.wheel_speed_plot.yaxis.tick_right()
        self.wheel_speed_plot.set_xlabel('Time (s)', x=0.55)
        self.wheel_speed_plot.grid(True,'major','y')
        self.wheel_speed_plot.set_xticklabels([''])
        #self.wheel_speed_plot.set_yticklabels([''])
        self.wheel1, self.wheel2, self.wheel3 = self.wheel_speed_plot.plot([],[],[],[],[],[],alpha=0.8)
        
        #view
        elev = 0#10
        azim = 0#volcano[1]+10
        self.orbit_ax.view_init(elev, azim)
        self.bod_ax.view_init(elev, azim)
        self.q = self.controller.q_cur
        self.sat_trace = self.sat_pos
        self.orientate()
        self.edges = [(j,k) for j in range(8) for k in range(j,8)
                if
                abs(np.linalg.norm(self.vertices[j]-self.vertices[k])-2*self.sat_scale)<0.01*self.sat_scale or
                abs(np.linalg.norm(self.vertices[j]-self.vertices[k])-6*self.sat_scale)<0.01*self.sat_scale]
        #self.init_fig()
        #self.tau = np.array([0,0,0])
        #self.set_all_data()
        res = 200
        lons = np.linspace(-180, 180, res*2)
        lats = np.linspace(-90, 90, res)[::-1]
        x,y,z = self.latlon2euc((lats,lons), self.earth_rad)
        self.orbit_ax.plot_wireframe(x, y, z, rstride=10, cstride=10,
                alpha=0.5, linewidth=1)
        self.vol_plot = self.orbit_ax.plot([],[],[],'ko', alpha=0.7)[0]
        self.click_euc = [0,0,1]
        self.volcano_pos = self.click_euc
        self.controller.set_vol_pos(self.click_euc)
        self.vol_plot.set_data([self.volcano_pos[0]],[self.volcano_pos[1]])
        self.vol_plot.set_3d_properties([self.volcano_pos[2]])
        
    def latlon2euc(self, latlon, alt):
        lat,lon = latlon
        lat *= np.pi/180
        lon *= np.pi/180
        x = np.outer(np.cos(lon), np.cos(lat)).T*alt
        y = np.outer(np.sin(lon), np.cos(lat)).T*alt
        z = np.outer(np.ones(np.size(lon)), np.sin(lat)).T*alt
        if len(x)==1:
            x,y,z = *x[0],*y[0],*z[0]
        return np.array([x,y,z])

    def init_fig(self):
        Artists = namedtuple("Artists",
                ("sat_axes_plot","edges_plot","sat_trace_plot",
                    "tau_plot", "bod_edges_plot","verts_plot"))
        self.artists = Artists([self.orbit_ax.plot([],[],[],'k')[0] for i in range(1)],
                               [self.orbit_ax.plot([],[],[], lw=1, color='firebrick', alpha=0.7)[0] for i in range(12)],
                               self.orbit_ax.plot([],[],[],'m', alpha=0.5)[0],
                               self.bod_ax.plot([],[],[],'b', lw=2)[0],
                               [self.bod_ax.plot([],[],[],'g-', lw=2)[0] for i in range(12)],
                               self.bod_ax.plot([],[],[],'bo',markersize=3)[0])

        self.orbit_ax.set_xticklabels([''])
        self.orbit_ax.set_yticklabels([''])
        self.orbit_ax.set_zticklabels([''])
        self.orbit_ax.zaxis.set_rotate_label(False)
        self.orbit_ax.yaxis.set_rotate_label(False)
        self.orbit_ax.xaxis.set_rotate_label(False)
        self.bod_ax.axis(self.sat_scale*np.array([-3,3,-3,3]))
        self.bod_ax.set_zlim3d(self.sat_scale*np.array((-3,3)))
        self.bod_ax.set_xticklabels([''])
        self.bod_ax.set_yticklabels([''])
        self.bod_ax.set_zticklabels([''])
        self.bod_ax.zaxis.set_rotate_label(False)
        self.bod_ax.yaxis.set_rotate_label(False)
        self.bod_ax.xaxis.set_rotate_label(False)
        return #[self.orbit_ax, self.bod_ax]

    def orientate(self):
        self.vertices = np.stack([self.q.rotate(vert) for vert in
            self.centre_vertices])
        self.sat_vertices = self.vertices
        self.sat_axes = np.array([self.q.rotate(v).tolist() for v in
            [self.centre_sat_axes]])

    def onclick(self, event):
        #self.orbit_fig.set_dpi(100)
        ymean = 823/1600
        zmean = 818/1600
        yscale = 433/1600
        zscale = 434/1600
        width, height = self.orbit_fig.get_size_inches()
        width*=self.orbit_fig.dpi
        height*=self.orbit_fig.dpi
        if event.dblclick:
           #if self.click_count == 1:
           #    #still not sure why this is not removing the first plot....
           #    print('removing')
           #    self.vol_plot.set_data([],[])
           #    self.vol_plot.set_3d_properties([])
           #    plt.draw()
           #    self.click_count+=1
            y, z = event.x, event.y
            y = (y-ymean*width)/(yscale*width)
            z = (z-zmean*height)/(zscale*height)
            hyp = (y**2+z**2)**0.5
            if hyp <= 1:
                x = (1-hyp**2)**0.5
                lat = np.arcsin(z)*180/np.pi+self.orbit_ax.elev
                lon = np.arctan2(y, x)*180/np.pi+self.orbit_ax.azim
                self.click_euc = self.latlon2euc((lat,lon),1)
                self.volcano_pos = self.click_euc
                self.controller.set_vol_pos(self.click_euc)
                self.vol_plot.set_data([self.volcano_pos[0]],[self.volcano_pos[1]])
                self.vol_plot.set_3d_properties([self.volcano_pos[2]])
           #if self.click_count== 0: 
           #    plt.draw()
           #    for i in range(1):
           #        self.artists.sat_axes_plot[i].remove()
           #    for count in range(12):
           #        self.artists.edges_plot[count].remove()
           #        self.artists.bod_edges_plot[count].remove()
           #    self.artists.verts_plot.remove()
           #    self.click_count+=1
           #    self.orbit_fig.canvas.mpl_disconnect(self.cid)

    def frame_iter(self):
        while True:
            connection_id = self.orbit_fig.canvas.mpl_connect('button_press_event', self.onclick)
            self.q  = self.controller.update()
            self.tau = self.controller.torq*15000*self.sat_scale
            ang_err = 0.5*(self.calc_angle_err()+self.controller.q_move.angle)
            self.err_vals += [ang_err]
            ang_err = self.controller.q_move.angle
            self.cont_err_vals += [ang_err]
            self.iter+=1
            if self.iter%self.time_sf==0: yield self.q, self.sat_pos
            #yield self.q, self.sat_pos
        return
    
    def set_all_data(self):
        for i in range(1):
            self.artists.sat_axes_plot[i].set_data([0,self.sat_axes[i,0]],[0,self.sat_axes[i,1]])
            self.artists.sat_axes_plot[i].set_3d_properties([0,self.sat_axes[i,2]])

        for count, (j,k) in enumerate(self.edges):
            self.artists.edges_plot[count].set_data(self.sat_vertices[[j,k],0],
                    self.sat_vertices[[j,k],1])
            self.artists.edges_plot[count].set_3d_properties(self.sat_vertices[[j,k],2])
        
        for count, (j,k) in enumerate(self.edges):
            self.artists.bod_edges_plot[count].set_data(self.vertices[[j,k],0],
                    self.vertices[[j,k],1])
            self.artists.bod_edges_plot[count].set_3d_properties(self.vertices[[j,k],2])

        self.artists.tau_plot.set_data([0, self.tau[0]], [0, self.tau[1]])
        self.artists.tau_plot.set_3d_properties([0, self.tau[2]])

        self.artists.verts_plot.set_data(self.vertices[:,0],self.vertices[:,1])
        self.artists.verts_plot.set_3d_properties(self.vertices[:,2])
        
        self.sat_trace = np.vstack((self.sat_trace, self.sat_pos))
        self.artists.sat_trace_plot.set_data(self.sat_trace[:,0],self.sat_trace[:,1])
        self.artists.sat_trace_plot.set_3d_properties(self.sat_trace[:,2])
        t = self.controller.t
        self.wheel_speed_plot.set_xlim([max(0,t-200), max(t,1)])
        self.wheel_speed_plot.set_ylim([-100,100]) 
        if t>0:
            self.wheel1.set_data(np.linspace(max(0,t-200),t,min(3000,len(self.controller.wheel_speeds))),self.controller.wheel_speeds[-3000:,0])
            self.wheel2.set_data(np.linspace(max(0,t-200),t,min(3000,len(self.controller.wheel_speeds))),self.controller.wheel_speeds[-3000:,1])
            self.wheel3.set_data(np.linspace(max(0,t-200),t,min(3000,len(self.controller.wheel_speeds))),self.controller.wheel_speeds[-3000:,2])
            self.err_plot.set_data(np.linspace(max(0,t-200),t,min(3000,len(self.err_vals))),
                    180*np.array(self.err_vals[-3000:])/np.pi)
            self.cont_err_plot.set_data(np.linspace(max(0,t-200),t,min(3000,len(self.err_vals))),
                    180*np.array(self.cont_err_vals[-3000:])/np.pi)
            self.err_plot.axes.axis([max(0,t-200), max(t,1), 0,
                max(180*max(self.err_vals[-3000:])/np.pi,1)])
            #self.cont_err_plot.axes.axis([max(0,t-200), max(t,1), 0,
            #    max(210*max(self.err_vals[len(self.err_vals[-3000:])//3:])/np.pi,1)])

    def update_artists(self, frames):
        self.bod_ax.view_init(self.orbit_ax.elev, self.orbit_ax.azim)
        self.orientate()
        self.set_all_data()
        return #[self.orbit_ax, self.bod_ax]
    
    def calc_angle_err(self):
        sat2vol = np.array(self.volcano_pos)
        rotated_z = self.q.rotate(np.array([0,0,1]))
        angle = np.arccos(min(np.dot(sat2vol,
                rotated_z),1))
        return angle
    
    def make_animation(self):
        anim = animation.FuncAnimation(fig=self.orbit_fig, func=self.update_artists,
                frames=self.frame_iter, init_func=self.init_fig, 
                interval=1, repeat=False, blit=False)
        return anim
