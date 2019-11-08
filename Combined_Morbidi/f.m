%MEAM517
%Greg Campbell & Spencer Folk
%Rework of Morbidi 2016 Paper

% This function expressed the dynamics introduced on page 4
function [xdot] = f(x,alpha)
% Input - x [16 x 1] - [x, xdot, y, ydot, z, zdot, phi, phidot, theta,
%                       thetadot, psi, psidot, omega_1, omega_2, omega_3, 
%                       omega_4]'

% Input - alpha [4 x 1] - [alpha_1, alpha_2, alpha_3]' - Motor
% accelerations

    % Output - xdot [16 x 1] - derivatives of states

    consts = constants();

    % Compute derivatives

    xdot = zeros(size(x));

    xdot(1) = x(2);
    xdot(2) = (consts.k_b/consts.m)*(sin(x(7))*sin(x(11)) + cos(x(7))*cos(x(11))*sin(x(9)))*sum([x(13) , x(14), x(15), x(16)].^2);
    xdot(3) = x(4);
    xdot(4) = (consts.k_b/consts.m)*(cos(x(7))*sin(x(9))*sin(x(11)) - cos(x(11))*sin(x(7)))*sum([x(13) , x(14), x(15), x(16)].^2);
    xdot(5) = x(6);
    xdot(6) = (consts.k_b/consts.m)*(cos(x(9))*cos(x(7)))*sum([x(13) , x(14), x(15), x(16)].^2) - consts.g;
    xdot(7) = x(8);
    xdot(8) = ((consts.I_y - consts.I_z)/consts.I_x)*x(10)*x(12) + (consts.l*consts.k_b/consts.I_x)*(x(14)^2 - x(16)^2) - (consts.J/consts.I_x)*x(10)*(x(13) - x(14) + x(15) - x(16));
    xdot(9) = x(10);
    xdot(10) = ((consts.I_z - consts.I_x)/consts.I_y)*x(8)*x(12) + (consts.l*consts.k_b/consts.I_y)*(x(15)^2 - x(13)^2) + (consts.J/consts.I_y)*x(8)*(x(13) - x(14) + x(15) - x(16));
    xdot(11) = x(12);
    xdot(12) = ((consts.I_x-consts.I_y)/consts.I_z)*x(8)*x(10) + (consts.k_tao/consts.I_z)*(x(13)^2 - x(14)^2 + x(15)^2 - x(16)^2);
    xdot(13) = alpha(1);
    xdot(14) = alpha(2);
    xdot(15) = alpha(3);
    xdot(16) = alpha(4);
end
