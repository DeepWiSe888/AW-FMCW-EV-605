close all
clc;clear;

addpath libs;
conf;
% param config
param.num_tx = num_tx;
param.num_rx = num_rx;
param.num_chirps_per_frame = num_chirps_per_frame;
param.num_samples_per_chirp = num_samples_per_chirp;
param.range_fft_n = num_range_nfft;
param.num_range_bins = num_range_bins;
param.doppler_fft_n = num_doppler_nfft;
param.start_idx = 1;
param.end_idx = num_range_bins;

%read file
filename = '../plot_datas/datas/1658751164026.dat';
%read from file
framelist = read_frame(filename);

%range fft,size(frame_cnt,num_tx*num_rx,chirp,bin_cnt)
range_data = struct_to_cube(framelist,param);

% loop related
win_size = fps;
step = round(fps / 8);

figure(1)
set(gcf,'position',[18.6,270.6,1086.4,477.6])
colormap jet

F = [];
F_CNT = 1;

for i = 1:step: length(range_data) - win_size
    clf;
    %get data
    range_data_tmp = range_data(:,i:i+ win_size - 1,:);
    
    %remove dc
    range_data_tmp = range_data_tmp - mean(range_data_tmp,2);

    %time range
    time_range = squeeze(mean(range_data_tmp,1));

    %doppler
    non_coherent = abs(doppler_fft(time_range,num_doppler_nfft));
    
    %azimuth and elevation
    azimuth_data = range_data_tmp(2:3,:,:);
    
    elevation_data = zeros(2,win_size,num_range_bins);
    elevation_data(1,:,:) = range_data_tmp(3,:,:);
    elevation_data(2,:,:) = range_data_tmp(1,:,:);
    
    range_azimuth_conv = capon(azimuth_data);
    range_elevation_conv = capon(elevation_data);
    
    
    cla
    subplot(221)
    imagesc(abs(time_range))
    %view(1,1)    
    
    subplot(222)
    imagesc(non_coherent)
    
    subplot(223)
    imagesc(range_azimuth_conv)
    %view(1,1) 
    
    subplot(224)
    imagesc(range_elevation_conv)
    
    cdata(1,:,:,:) = getframe(gcf).cdata;
    F = [F;cdata];
    
    pause(0.01)
end

write2video('yingfeiling',F,8);













