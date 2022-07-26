function range_data = struct_to_cube(framelist,param)
    num_tx = param.num_tx;
    num_rx = param.num_rx;
    num_chirps_per_frame = param.num_chirps_per_frame;
    num_samples_per_chirp = param.num_samples_per_chirp;
    num_range_nfft = param.range_fft_n;
    num_range_bins = param.num_range_bins;

    frame_len = length(framelist);

    range_data = zeros(num_tx*num_rx,frame_len,num_range_bins);
    for i = 1:frame_len
        adc_tmp = framelist(i).adc;
        for j = 1:num_tx*num_rx
            tmp = adc_tmp(num_chirps_per_frame * num_samples_per_chirp * (j - 1) + 1:num_chirps_per_frame * num_samples_per_chirp * j);
            tmp = reshape(tmp,[num_samples_per_chirp,num_chirps_per_frame]);
            tmp = mean(tmp,2); 
            tmp_fft = range_fft(tmp,num_range_nfft);
            range_data(j,i,:) = tmp_fft(1:num_range_bins);
        end
    end
end