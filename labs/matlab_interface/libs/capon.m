function capon_map = capon(data)
    [ant_cnt,frame_cnt,bin_cnt] = size(data);
    angle_one_sided = 90;
    angle_points = angle_one_sided * 2 + 1;
    steering_vector = gen_steering_vec(angle_one_sided,ant_cnt);
    capon_map = zeros(angle_points,bin_cnt);
    for i = 1:bin_cnt
        X = squeeze(data(:,:,i));
        R = X*X';
        R = forward_backward_avg(R);
        R = R / frame_cnt;
        R_INV = inv(R);
        Rxx =  1 ./ diag(abs((steering_vector * R_INV * steering_vector')));
        capon_map(:,i) = Rxx;
    end
    conv_k = ones(5,5) / 25;
    capon_map = conv2(capon_map,conv_k,'same');
    capon_map = imgaussfilt(capon_map);
end