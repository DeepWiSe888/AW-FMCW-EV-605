function Rxx_fb = forward_backward_avg(Rxx)
    [M,~] = size(Rxx);
    J = fliplr(eye(M));
    Rxx_fb = (Rxx + J * conj(Rxx) * J) / 2;
end