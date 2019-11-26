%MEAM517
%Greg Campbell & Spencer Folk
%Rework of Morbidi 2016 Paper

%Function to grab indices - based off of Matt Halm's code from HW5

function [x_i_inds, u_i_inds] = sample_indices(i)
%SAMPLE_INDICES calculates the indices of z such that z(x_i_inds) = x_i and
% z(u_i_inds) = u_i.
%   @param i: sample number; scalar
%   @param nx: dimension of state vector, x; scalar
%   @param nu: dimension of input vector, u; scalar
%
%   @output x_i_inds: indices such that z(x_i_inds) = x_i; 1 by n_x vector
%   @output u_i_inds: indices such that z(u_i_inds) = u_i; 1 by n_u vector
    consts = constants();
    nx = consts.nx;
    nu = consts.nu;

    x_i_inds = ones(1,nx);
    u_i_inds = ones(1,nu);
    
    nt = nx+nu; %total n
    
    %EQNs for indices
    x_i_inds = [nt*i-(nt-1):nt*i-nu];
    u_i_inds = [nt*i-(nu-1):nt*i];

end

