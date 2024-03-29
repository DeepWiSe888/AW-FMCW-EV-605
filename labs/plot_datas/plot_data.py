import os
import time
import numpy as np

import threading
from pyqtgraph.Qt import QtGui, QtWidgets
import pyqtgraph.opengl as gl
import pyqtgraph as pg

from libs.recv import *
from libs.conf import *
from libs.utils import *


def main():
    global curve1,curve2,curve3,curve4,curve5,curve6

    # Set graphical window, its title and size
    win = pg.GraphicsLayoutWidget(show=True)
    win.resize(1500,600)
    win.setWindowTitle('plot radar data')
    win.setBackground("w")

    pg.setConfigOptions(antialias=True)

    colorMap = pg.colormap.get("CET-D1")
    range_ticks = [i for i in range(MAX_BIN - OFFSET + 1) if i % 10 == 0]
    time_ticks = [i for i in range(FRAMES + 1) if i % FPS == 0]

    p1 = win.addPlot(title="range bin")
    curve1 = p1.plot(pen='r')
    p1.setLabels(left='amplitude', bottom='range(m)')
    ax1 = p1.getAxis('bottom')
    ax1.setTicks([[(v, '{:.2f}'.format((v + OFFSET) * RANGE_RESOLUTION + RANGE_SATRT)) for v in range_ticks]])

    p2 = win.addPlot(title="max bin iq")
    curve2 = p2.plot(pen='g')
    p2.setLabels(bottom='i',left='q')

    p3 = win.addPlot(title="max bin time series")
    curve3 = p3.plot(pen='b')
    p3.setLabels(left='amplitude',bottom='time(s)')
    ax3 = p3.getAxis('bottom')
    ax3.setTicks([[(v, '{}'.format(int(v / FPS) )) for v in time_ticks]])


    win.nextRow()
    p4 = win.addPlot(title='time-range')
    curve4 = pg.ImageItem()
    p4.addItem(curve4)
    p4.setLabels(left='time(s)',bottom='range(m)')
    bar4 = pg.ColorBarItem( values=(0,1), colorMap=colorMap)
    bar4.setImageItem(curve4)  
    ax4 = p4.getAxis('bottom')
    ax4.setTicks([[(v, '{:.2f}'.format((v + OFFSET) * RANGE_RESOLUTION + RANGE_SATRT)) for v in range_ticks]]) 
    ax4 = p4.getAxis('left')
    ax4.setTicks([[(v, '{}'.format(int(v / FPS) )) for v in time_ticks]])

    p5 = win.addPlot(title='time-range remove the background')
    curve5 = pg.ImageItem()
    p5.addItem(curve5)
    p5.setLabels(left='time(s)',bottom='range(m)')
    bar5 = pg.ColorBarItem( values=(0,1), colorMap=colorMap)
    bar5.setImageItem(curve5) 
    ax5 = p5.getAxis('bottom')
    ax5.setTicks([[(v, '{:.2f}'.format((v + OFFSET) * RANGE_RESOLUTION + RANGE_SATRT)) for v in range_ticks]]) 
    ax5 = p5.getAxis('left')
    ax5.setTicks([[(v, '{}'.format(int(v / FPS) )) for v in time_ticks]])

    p6 = win.addPlot(title='range-doppler')
    curve6 = pg.ImageItem()
    p6.addItem(curve6)
    p6.setLabels(left='doppler(Hz)',bottom='range(m)')
    bar6 = pg.ColorBarItem(values=(0,1), colorMap=colorMap)
    bar6.setImageItem(curve6)  
    ax6 = p6.getAxis('bottom')
    ax6.setTicks([[(v, '{:.2f}'.format((v + OFFSET) * RANGE_RESOLUTION + RANGE_SATRT)) for v in range_ticks]])
    ax6 = p6.getAxis('left')
    freq_ticks = [i for i in range(NFFT + 1) if i % FPS == 0]
    ax6.setTicks([[(v, '{}'.format(int((v - NFFT / 2) * FPS / NFFT)) ) for v in freq_ticks]])

    s = SerialCollect(port='COM8')
    recv_thd = threading.Thread(target=s.recv_data)
    recv_thd.setDaemon=True
    recv_thd.start()

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
                if len(framelist) >= FRAMES:
                    frames = np.array(framelist)
                    x = frames[:,OFFSET:MAX_BIN]
                    x = np.mean(frames,1)
                    
                    #remove the background
                    iq_data = x - np.mean(x,0)
                    iq_abs = np.abs(iq_data)
                    iq_bin_sum = np.sum(iq_abs,0)

                    #determine the target bin
                    iq_bin = np.mean(np.abs(iq_data),0)
                    bin_offset = 0
                    bin_idx = np.where(np.max(iq_bin[bin_offset:]) <= iq_bin[bin_offset:])[0][0]
                    bin_idx += bin_offset
                    print(bin_idx)
                    org_wave = iq_data[:,bin_idx]

                    #fft
                    fft_data = np.fft.fft(iq_data,n = NFFT,axis=0)
                    fft_shift_data = np.fft.fftshift(fft_data,axes=0)
                    fft_abs = np.abs(fft_shift_data)

                    #update plot data
                    curve1.setData(iq_bin_sum)
                    curve2.setData(org_wave.real,org_wave.imag)
                    curve3.setData(iq_abs[:,bin_idx])
                    curve4.setImage(np.abs(x).T)
                    curve5.setImage(iq_abs.T)
                    curve6.setImage(fft_abs.T)
                        
                        
                       
                    #update plot immediate
                    QtWidgets.QApplication.processEvents()
                    
                    framelist = framelist[1*40:]

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
