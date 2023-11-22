from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter
import numpy as np
from copy import deepcopy

class Track():
    def __init__(self,dt):
        self.ekf = ExtendedKalmanFilter(dim_x=2, dim_z=2)
        # make an imperfect starting guess
        self.ekf.x = np.array([0.1, 0.1, 0.1])

        self.ekf.F = np.eye(3) + np.array([[0, 0, 0],
                                            [0, 0, 0],
                                            [0, 0, 0]]) * dt
        
        range_std = 0.05 # meters
        theta_std = 0.2 # 
        phi_std = 0.2
        self.ekf.R = np.diag([range_std**2,theta_std**2,phi_std**2])
        self.ekf.Q = Q_discrete_white_noise(2, dt=dt, var=0.1)
        self.ekf.P *= 5
    
    def _HJacobian(self,x):
        Px = x[0]
        Py = x[1]
        Pz = x[2]
        H = np.array([[Px/(Px**2 + Py**2 + Pz**2)**0.5, Py/(Px**2 + Py**2 + Pz**2)**0.5, Pz/(Px**2 + Py**2 + Pz**2)**0.5],
                      [- Py / (Px**2 + Py**2 + Pz**2),  Px / (Px**2 + Py**2 + Pz**2)   , ]])
        return H
    
    def _Hx(self,x):
        Px = x[0]
        Py = x[1]
        Pz = x[2]
        Hx = np.array([(Px**2 + Py**2 + Pz**2)**0.5,np.arctan(Py/ Px),np.arctan(Pz/ (Px**2 + Py**2)**0.5)])
        return Hx
    
    def update(self,z):
        # self.test_update(z, self._HJacobian, self._Hx)
        self.ekf.update(z, self._HJacobian, self._Hx)
        self.ekf.predict()
        return self.ekf.x