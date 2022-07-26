function doppler_data = doppler_fft(data,doppler_fft_n)
    %time-range data
    %data size:[frame_cnt,bin_cnt]
    [frame_cnt,bin_cnt] = size(data);
    data = data .* hann(frame_cnt);
    doppler_data = fftshift(fft(data,doppler_fft_n),1);
    doppler_data = doppler_data / doppler_fft_n;
end