function data_fft = range_fft(data,range_fft_n)
    data = data .* hann(length(data));
    data_fft = fft(data,range_fft_n);
end