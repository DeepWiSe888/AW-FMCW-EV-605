close all;
 %
delete(instrfindall);
if exist('s','var')
    delete(s);
end
s = serialport("COM8",921600);
configureTerminator(s,"CR/LF")

addpath libs;
conf;
% param config
param.fps = fps;
param.num_tx = num_tx;
param.num_rx = num_rx;
param.num_chirps_per_frame = num_chirps_per_frame;
param.num_samples_per_chirp = num_samples_per_chirp;
param.range_fft_n = num_range_nfft;
param.num_range_bins = num_range_bins;
param.doppler_fft_n = num_doppler_nfft;
param.start_idx = 1;
param.end_idx = num_range_bins;
param.SEC = SEC;

demo_1(s,param);
function demo_1(s,param)
    % loop related
    fps = param.fps;
    num_range_nfft = param.range_fft_n;
    num_range_bins = param.num_range_bins;
    num_doppler_nfft = param.doppler_fft_n;
    sec = param.SEC;

    win_size = fps;
    step_size = round(fps / 8);

    last_fn = 0;
    framelist = [];
    figure(1)
    set(gcf,'position',[18.6,270.6,1086.4,477.6])
    colormap jet
    while 1
        data = parse_pack(s);
        framelist = [framelist;data];
        
        if data.fno - last_fn > 1
            fprintf('frame no error,last frame no:%d,frame no:%d.\n',last_fn,data.fno);
        elseif mod(data.fno,100) == 0
            fprintf('timestamp:%d,tx:%d,rx:%d,frame no:%d,adc len:%d.\n',data.timestamp,data.tx,data.rx,data.fno,length(data.adc));
        end
        last_fn = data.fno;
        
        if length(framelist) >= win_size
            range_data_tmp = struct_to_cube(framelist,param);
            [ant_cnt,frame_cnt,bin_cnt] = size(range_data_tmp);
            
            %remove dc
            range_data_tmp = range_data_tmp - mean(range_data_tmp,2);

            %time range
            time_range = squeeze(mean(range_data_tmp,1));

            %doppler
            non_coherent = abs(doppler_fft(time_range,num_doppler_nfft));

            %azimuth and elevation
            azimuth_data = range_data_tmp(2:3,:,:);

            elevation_data = zeros(2,frame_cnt,num_range_bins);
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
            
            framelist = framelist(step_size:end);
            pause(0.001)
        end
        
    end
end

function demo_2(s)
    configureCallback(s,"byte",1024,@serial_callback)
    pause
    configureCallback(s,"off")
end

function serial_callback(s,~)
    figure(1)
    data = parse_pack(s);
    plot(abs(data.i + 1i*data.q))
end

function data = parse_pack(s)
    
    flag = [119,105,114,117,115,104,45,118,112,97,115,58];
    packs = [];
    cnt = 1;
    while cnt <=12
        f = read(s,1,'uint8');
        if f == flag(cnt)
            cnt = cnt + 1;
        else
            cnt = 1;
        end
    end
    %sec
    sec = read(s,1,'uint64');
    usec = read(s,1,'uint64');
    % timestamp
    timestamp = sec + cast(usec,'double')  / 1000000;
    % rx
    rx = read(s,1,'uint32');
    % tx
    tx = read(s,1,'uint32');
    % frame no
    fno = read(s,1,'uint32');
    % adc
    adc = read(s,1536,'single');
    
    data.timestamp = timestamp;
    data.rx = rx;
    data.tx = tx;
    data.fno = fno;
    data.adc = adc;
end





