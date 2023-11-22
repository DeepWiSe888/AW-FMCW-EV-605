import os
import time
import numpy as np
from scipy import signal
from scipy import ndimage
# import cv2

import threading
from pyqtgraph.Qt import QtGui, QtWidgets
import pyqtgraph.opengl as gl
import pyqtgraph as pg

from libs.recv import *
from libs.conf import *
from libs.utils import *
from libs.plot import PlotData
from libs.track import Track


def polar_to_cartesian(pdata):
    (x_shape,y_shape) = pdata.shape
    angle_map = np.zeros((y_shape * 2,y_shape))
    x_idxs = [i for i in range(-1 * y_shape,y_shape,1)]
    y_idxs = [i for i in range(0,y_shape,1)]

    # 映射关系
    for i,x_idx in enumerate(x_idxs):
        for j,y_idx in enumerate(y_idxs):
            if y_idx == 0  and x_idx == 0:
                R = y_idx
                the = 90
            else:
                R = int(np.math.sqrt(x_idx*x_idx + y_idx * y_idx))
                the = int(np.math.acos(x_idx / R ) / np.pi * 180)
            if the >=x_shape:
                the = x_shape - 1
            elif the < 0:
                the = 0
            if R >=y_shape:
                R = y_shape - 1
            elif R < 0:
                R = 0
            angle_map[i,j] = pdata[the,R]
    # 均值滤波
    angle_map = signal.convolve2d(angle_map,np.ones((3,3)) / 9,'same')
    return angle_map


def polar2cart(polar_data, order=3):
    from scipy.ndimage.interpolation import map_coordinates as mp
    (x_shape,y_shape) = polar_data.shape
    x = np.linspace(-y_shape,y_shape,y_shape*2 + 1)
    y = np.linspace(0,y_shape,y_shape + 1)
    
    theta_step = 1
    range_step = 1
    X, Y = np.meshgrid(x, y)

    Tc = np.degrees(np.arctan2(Y, X)).ravel()
    Rc = (np.sqrt(X**2 + Y**2)).ravel()

    Tc = Tc / theta_step
    Rc = Rc / range_step

    coords = np.vstack((Tc, Rc))
    polar_data = np.vstack((polar_data, polar_data[-1,:]))
    cart_data = mp(polar_data, coords, order=order, mode='constant', cval=np.nan)
    return cart_data.reshape(len(y), len(x)).T
    

class FallDetetion(object):
    def __init__(self):
        ekf = Track()
        traget_position = ekf.update(z)
        pass    
    
    def monitor(self,raw_data):
        #remove the background
        iq_data = raw_data - np.mean(raw_data,0)
        
        
        iq_abs = np.sum(np.abs(np.mean(iq_data,1)),0)

        max_p = np.where(iq_abs == np.max(iq_abs))[0][0]
        
        p_start = 0
        p_end = 0
        if max_p - 2 < 0:
            p_start = 0
            p_end = 5
        elif max_p + 3 > len(iq_abs) - 1:
            p_start = len(iq_abs) - 1 - 5
            p_end = len(iq_abs) - 1
        else:
            p_start = max_p - 2
            p_end = max_p + 3
            
        azimuth_data = iq_data[:,1:3,p_start:p_end]
        azimuth_capon = capon(azimuth_data)
        azimuth_tmp = np.sum(azimuth_capon,1)
        azimuth_rad = np.where(azimuth_tmp == np.max(azimuth_tmp))[0][0]
        
        elevation_data = iq_data[:,0:2,p_start:p_end]
        elevation_capon = capon(elevation_data)
        elevation_tmp = np.sum(elevation_capon,1)
        elevation_rad = np.where(elevation_tmp == np.max(elevation_tmp))[0][0]
        
        azimuth_angle = (azimuth_rad - 90) * (np.pi / 180)
        elevation_angle = (elevation_rad - 90) * (np.pi / 180)
        
        target_r = max_p * RANGE_RESOLUTION
        target_x = target_r * np.sin(azimuth_angle) * np.cos(elevation_angle)
        target_y = target_r * np.cos(azimuth_angle) * np.cos(elevation_angle)
        target_z = target_r * np.sin(elevation_angle)
        
        print("peak:{},x:{:.2f},y:{:.2f},z:{:.2f}".format(max_p,target_x,target_y,target_z))
    
    


def main():
    s = SerialCollect(port='COM8',baudrate= 921600)
    recv_thd = threading.Thread(target=s.recv_data)
    recv_thd.setDaemon(True)
    recv_thd.start()
    plot_data = PlotData()
    framelist = []
    
    try:
        while True:
            if Cache.empty():
                time.sleep(0.001)
                continue
            else:
                # print(Cache.qsize())
                data_dict = Cache.get()
                adc_tmp = data_dict['data']
                adc = np.zeros((num_tx * num_rx ,num_chirps_per_frame,num_samples_per_chirp),dtype=np.complex_)
                try:
                    for i in range(num_tx * num_rx):
                        adc[i,:,:] = np.reshape(adc_tmp[num_chirps_per_frame * num_samples_per_chirp * i:num_chirps_per_frame * num_samples_per_chirp * (i + 1)],(num_chirps_per_frame,num_samples_per_chirp))
                except:
                    pass
                
                (n_rx,_,_) = adc.shape
                rx_iq = np.zeros((n_rx,num_range_bins),dtype=np.complex_)
                for rx in range(n_rx):
                    tmp_adc = np.mean(adc[rx,:,:],0)
                    iq = range_fft(tmp_adc,num_range_nfft)
                    iq = iq[:num_range_bins]
                    rx_iq[rx,:] = iq
                
                #plot data
                framelist.append(rx_iq)
                if len(framelist) >= 40:
                    frames = np.array(framelist)
                    x = frames[:,:,OFFSET:MAX_BIN]
                    
                    #remove the background
                    iq_data = x - np.mean(x,0)
                    
                    
                    iq_abs = np.sum(np.abs(np.mean(iq_data,1)),0)
    
                    max_p = np.where(iq_abs == np.max(iq_abs))[0][0]
                    
                    p_start = 0
                    p_end = 0
                    if max_p - 2 < 0:
                        p_start = 0
                        p_end = 5
                    elif max_p + 3 > len(iq_abs) - 1:
                        p_start = len(iq_abs) - 1 - 5
                        p_end = len(iq_abs) - 1
                    else:
                        p_start = max_p - 2
                        p_end = max_p + 3
                        
                    azimuth_data = iq_data[:,1:3,p_start:p_end]
                    azimuth_capon = capon(azimuth_data)
                    azimuth_tmp = np.sum(azimuth_capon,1)
                    azimuth_rad = np.where(azimuth_tmp == np.max(azimuth_tmp))[0][0]
                    
                    elevation_data = iq_data[:,0:2,p_start:p_end]
                    elevation_capon = capon(elevation_data)
                    elevation_tmp = np.sum(elevation_capon,1)
                    elevation_rad = np.where(elevation_tmp == np.max(elevation_tmp))[0][0]
                    
                    azimuth_angle = (azimuth_rad - 90) * (np.pi / 180)
                    elevation_angle = (elevation_rad - 90) * (np.pi / 180)
                    
                    target_r = max_p * RANGE_RESOLUTION
                    target_x = target_r * np.sin(azimuth_angle) * np.cos(elevation_angle)
                    target_y = target_r * np.cos(azimuth_angle) * np.cos(elevation_angle)
                    target_z = target_r * np.sin(elevation_angle)
                    
                    print("peak:{},x:{:.2f},y:{:.2f},z:{:.2f}".format(max_p,target_x,target_y,target_z))
                    
                    
                    azimuth_data = iq_data[:,1:3,:]
                    azimuth_capon = capon(azimuth_data)
                    
                    elevation_data = iq_data[:,0:2,:]
                    elevation_capon = capon(elevation_data)
                    
                    
                    angle_result = np.dot(azimuth_capon,elevation_capon.T)
                    angle_result = azimuth_capon * elevation_capon
                    
                    
                    
                    
                    iq_data = np.mean(iq_data,1)
                    iq_abs = np.abs(iq_data)
                    iq_bin_sum = np.sum(iq_abs,0)

                    #determine the target bin
                    iq_bin = np.mean(np.abs(iq_data),0)
                    bin_offset = 0
                    bin_idx = np.where(np.max(iq_bin[bin_offset:]) <= iq_bin[bin_offset:])[0][0]
                    bin_idx += bin_offset
                    org_wave = iq_data[:,bin_idx]

                    #fft
                    fft_data = np.fft.fft(iq_data,n = NFFT,axis=0)
                    fft_shift_data = np.fft.fftshift(fft_data,axes=0)
                    fft_abs = np.abs(fft_shift_data)
                    
                    
                    #max position
                    # (theta_idx,range_idx) = np.where(capon_map_conv == np.max(capon_map_conv))
                    # theta_idx = theta_idx[0]
                    # range_idx = range_idx[0]
                    # z = np.array([range_idx * RANGE_RESOLUTION, (theta_idx - 90) * (np.pi / 180)])
                    
                    # traget_position = ekf.update(z)
                    
                    # print('{:.2f} {:.2f},{:.2f} {:.2f},{:.2f} {:.2f}'.format(z[0],z[1],z[0] * np.cos(z[1]),z[0] * np.sin(z[1]),
                    #                                                          traget_position[0],traget_position[1]))

                    
                    xy_data = polar_to_cartesian(azimuth_capon)
                    # xy_data = polar2cart(capon_map_conv)
                    
                    
                    xy_data = np.flip(xy_data,axis=1)
                    xy_data = np.flip(xy_data,axis=0)
                    # xy_data = xy_data.T
                    curve_list = [iq_bin_sum,[org_wave.real,org_wave.imag],iq_abs[:,bin_idx],
                                  iq_abs.T,fft_abs.T,xy_data]
                    
                    plot_data.update(curve_list)
                        
                    
                    framelist = framelist[10:]

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
