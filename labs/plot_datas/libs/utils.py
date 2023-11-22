import numpy as np
import struct
from scipy import signal,ndimage
try:
    from conf import *
except:
    from libs.conf import *

def parse_pack(pack):
    cursor = 0
    cursor = cursor + len(flag)
    
    # sec
    sec_pack_tmp = pack[cursor:cursor + 8]
    sec = list(struct.unpack('q',sec_pack_tmp))[0]
    cursor = cursor + 8

    # usec
    usec_pack_tmp = pack[cursor:cursor + 8]
    usec = list(struct.unpack('q',usec_pack_tmp))[0]
    usec = usec / 1000000
    cursor = cursor + 8

    # rx
    rx_pack_tmp = pack[cursor:cursor + 4]
    rx = list(struct.unpack('i',rx_pack_tmp))[0]
    cursor = cursor + 4

    # tx
    tx_pack_tmp = pack[cursor:cursor + 4]
    tx = list(struct.unpack('i',tx_pack_tmp))[0]
    cursor = cursor + 4

    # frame no
    fno_pack_tmp = pack[cursor:cursor + 4]
    fno = list(struct.unpack('i',fno_pack_tmp))[0]
    cursor = cursor + 4

    # DATA
    adc_data_pack_tmp = pack[cursor:]
    adc_data = list(struct.unpack('{}f'.format(len(adc_data_pack_tmp)//4),adc_data_pack_tmp))

    pack_dict = {'fno':fno,'tx':tx,'rx':rx,'t':sec + usec,'data':adc_data}
    return pack_dict


def range_fft(data,range_fft_n):
    data =  data * np.hanning(len(data))
    data_fft = np.fft.fft(data,range_fft_n)
    return data_fft

def gen_steering_vec(angle_one_sided,ant_cnt):
    angle_res = 1
    angle_points = round(2 * angle_one_sided / angle_res + 1)
    steering_vector = np.zeros((angle_points, ant_cnt), dtype=np.complex_)
    for ii in range(angle_points):
        for jj in range(ant_cnt):
            mag = -1 * np.pi * jj * np.sin((-angle_one_sided + ii * angle_res) * np.pi / 180)
            real = np.cos(mag)
            imag = np.sin(mag)
            steering_vector[ii, jj] = real + 1j * imag
    return steering_vector

def forward_backward_avg(Rxx):
    assert np.size(Rxx, 0) == np.size(Rxx, 1)

    M = np.size(Rxx, 0)
    Rxx = np.matrix(Rxx)

    J = np.eye(M)
    J = np.fliplr(J) 
    J = np.matrix(J)
    
    Rxx_FB = (Rxx + J * np.conj(Rxx) * J) / 2
    return np.array(Rxx_FB)

def capon(data):
    (frame_cnt_n,ant_cnt,bin_cnt) = data.shape
    steering_vector = gen_steering_vec(90,ant_cnt)
    capon_map = np.zeros((181,bin_cnt),dtype=np.complex_)
    for i in range(bin_cnt):
        X = data[:,:,i]
        X = X.T
        R = X @ np.conj(X.T)
        R = forward_backward_avg(R)
        R = np.divide(R, frame_cnt_n)
        R_INV = np.linalg.inv(R)
        tmp = R_INV @ steering_vector.T
        # np.einsum('ij,ij->i', steering_vector.conj(), tmp.T) == np.sum(steering_vector.conj() * tmp.T,1)
        mvdr_result = np.reciprocal(np.einsum('ij,ij->i', steering_vector.conj(), tmp.T))
        capon_map[:,i] = np.array(abs(mvdr_result))
    capon_map = np.real(capon_map)
    conv_k = np.ones((5,5)) / 25
    capon_map_conv = signal.convolve2d(capon_map,conv_k,'same')
    capon_map_conv = ndimage.gaussian_filter(capon_map_conv,sigma=7)
    return capon_map_conv
    