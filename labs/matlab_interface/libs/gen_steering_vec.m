function steering_vector = gen_steering_vec(angle_one_sided,ant_cnt)
    angle_res = 1;
    angle_points = round(2 * angle_one_sided / angle_res + 1);
    steering_vector = zeros(angle_points,ant_cnt);
    for ii = 1:angle_points
        for jj = 1:ant_cnt
            mag = -1 * pi * jj * sin((-angle_one_sided + ii * angle_res) * pi / 180);
            steering_vector(ii, jj) = cos(mag) + 1i * sin(mag);
        end
    end
end