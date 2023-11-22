import os
import time
import numpy as np
import serial
import struct
from scipy import signal
from queue import Queue

import threading
from pyqtgraph.Qt import QtGui, QtWidgets
import pyqtgraph.opengl as gl
import pyqtgraph as pg

from libs.recv import *
from libs.conf import *
from libs.utils import *


brpm,arpm = signal.butter(3,[0.1,0.5],'bandpass',analog=False,output='ba',fs=fps)
bbpm,abpm = signal.butter(5,[1,2],'bandpass',analog=False,output='ba',fs=fps)

class SignalProcess():
    def __init__(self) -> None:
        
        self.rpm_list = []
        self.bpm_list = []
        
        pass
    
    def match_filter(self,data):
        h = np.conj(data[::-1])
        data = np.convolve(data,h)
        return data
    
    def calcu_rpm(self,x):
        breathe_filter = signal.filtfilt(brpm,arpm,x,axis=0)
        breathe_fft = np.abs(np.fft.fft(breathe_filter,FPS))
        breathe_fft_max = np.max(breathe_fft[:int(len(breathe_fft)/2)])
        breathe_fft_max_index = (np.where(breathe_fft[:int(len(breathe_fft)/2)] == breathe_fft_max))[0][0]
        rpm = breathe_fft_max_index * fps/FPS * 60
        return rpm

    def calcu_bpm(self,x):
        filter_data = signal.filtfilt(bbpm,abpm,x,axis=0)
        # filter_data = self.match_filter(filter_data)
        bpm_fft = np.abs(np.fft.fft(filter_data,FPS))
        bpm_fft_max = np.max(bpm_fft[:int(len(bpm_fft)/2)])
        bpm_fft_max_index = (np.where(bpm_fft[:int(len(bpm_fft)/2)] == bpm_fft_max))[0][0]
        bpm = bpm_fft_max_index * fps/ FPS * 60
        return bpm
    
    def get_vital(self):
        rpm_result = np.array(self.rpm_list)
        if len(self.rpm_list) > 10:
            rpm_result = signal.convolve(self.rpm_list,np.array([1,1,1,1,1,1,1]),'same') / 7
            rpm_result = rpm_result[4:-4]
        rpm_result = rpm_result.astype(np.int32)
        bpm_result = np.array(self.bpm_list)
        if len(self.bpm_list) > 10:
            bpm_result = signal.convolve(self.bpm_list,np.array([1,1,1,1,1,1,1]),'same') / 7
            bpm_result = bpm_result[4:-4]
        bpm_result = bpm_result.astype(np.int32)
        return rpm_result,bpm_result
    
    def process(self,data):
        remove_dc_data = data - np.mean(data,0)
        
        time_range_map = np.abs(remove_dc_data)
        
        bin_data = np.mean(time_range_map,0)
        peak_idx = np.where(np.max(bin_data) <= bin_data)[0][0]
        
        peak_start = 0
        peak_end = 0
        if peak_idx - 1 < 0:
            peak_start = 0
            peak_end = 3
        elif peak_idx + 2 >= len(bin_data):
            peak_start = len(bin_data) - 4
            peak_end = len(bin_data) - 1
        else:
            peak_start = peak_idx - 1
            peak_end = peak_idx + 2 
            
        # peak_data = remove_dc_data[:,peak_start:peak_end] 
        # vital_data = np.sum(peak_data,1)  
        peak_data = remove_dc_data[:,peak_idx]
        vital_data = peak_data
        vital_data_abs = np.abs(vital_data)
        
        rpm = self.calcu_rpm(vital_data)
        bpm = self.calcu_bpm(vital_data)
        
        if len(self.rpm_list) > 10 and np.abs(rpm - np.mean(self.rpm_list[-10:])) > 10:
            self.rpm_list.append(self.rpm_list[-1])
        else:
            self.rpm_list.append(rpm)
        if len(self.rpm_list) > 120:
            self.rpm_list = self.rpm_list[1:]
            
        if len(self.bpm_list) > 10 and np.abs(bpm - np.mean(self.bpm_list[-10:])) > 20:
            self.bpm_list.append(self.bpm_list[-1])
        else:
            self.bpm_list.append(bpm)
        if len(self.bpm_list) > 120:
            self.bpm_list = self.bpm_list[1:]
        
        return time_range_map,bin_data,vital_data_abs                

def main():
    global curve1,curve21,curve22,curve23,curve3,curve41,curve42,curve43,curve51,curve52,curve53,curve6

    # Set graphical window, its title and size
    win = pg.GraphicsLayoutWidget(show=True)
    win.resize(1500,600)
    win.setWindowTitle('plot radar data')
    win.setBackground("w")

    pg.setConfigOptions(antialias=True)

    colorMap = pg.colormap.get("CET-D1")
    # range_ticks = [i for i in range(MAX_BIN - OFFSET + 1) if i % 10 == 0]
    # time_ticks = [i for i in range(FRAMES + 1) if i % FPS == 0]
    
    
    p1 = win.addPlot(title='time-range')
    curve1 = pg.ImageItem()
    p1.addItem(curve1)
    p1.setLabels(left='time(s)',bottom='range(m)')
    bar1 = pg.ColorBarItem( values=(0,1), colorMap=colorMap) 
    bar1.setImageItem(curve1)

    p2 = win.addPlot(title="range bin")
    p2.addLegend()
    curve21 = p2.plot(pen='r',name="rx1")
    curve22 = p2.plot(pen='g',name="rx2")
    curve23 = p2.plot(pen='b',name="rx3")
    p2.setLabels(left='amplitude', bottom='range(m)')

    # p3 = win.addPlot(title="max bin amplitude")
    # curve3 = p3.plot(pen='g')
    # p3.setLabels(left='amplitude', bottom='time')


    win.nextRow()
    p4 = win.addPlot(title="rpm list")
    p4.addLegend()
    curve41 = p4.plot(pen='r',name="rx1")
    curve42 = p4.plot(pen='g',name="rx2")
    curve43 = p4.plot(pen='b',name="rx3")
    p4.setLabels(left='rpm',bottom='time(s)')
    p4.setYRange(5,32)
    
    p5 = win.addPlot(title="bpm list")
    p5.addLegend()
    curve51 = p5.plot(pen='r',name="rx1")
    curve52 = p5.plot(pen='g',name="rx2")
    curve53 = p5.plot(pen='b',name="rx3")
    p5.setLabels(left='bpm',bottom='time(s)')
    p5.setYRange(60,120)
    


    
    s = SerialCollect(port='COM8')
    recv_thd = threading.Thread(target=s.recv_data)
    recv_thd.setDaemon(True)
    recv_thd.start()
    p = SignalProcess()

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
                num_chirps = 4
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
                
                if len(framelist) >= 20 * fps:
                    frames = np.array(framelist)
                    
                    for rx in range(n_rx):
                        x = frames[:,rx,:]
                        time_range_map,bin_data,vital_data_abs = p.process(x)
                        rpm_list,bpm_list = p.get_vital()

                        #update plot data
                        if rx == 0:
                            curve1.setImage(time_range_map.T)
                            curve21.setData(bin_data)
                            # curve3.setData(vital_data_abs)
                            curve41.setData(rpm_list)
                            curve51.setData(bpm_list)
                        elif rx == 1:
                            curve22.setData(bin_data)
                            curve42.setData(rpm_list)
                            curve52.setData(bpm_list)
                        else:
                            curve23.setData(bin_data)
                            curve43.setData(rpm_list)
                            curve53.setData(bpm_list)
                        
                        print('rpm:{},bpm:{}'.format(rpm_list[-1],bpm_list[-1]))

                    #update plot immediate
                    QtWidgets.QApplication.processEvents()
                    
                    framelist = framelist[1*fps:]

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
