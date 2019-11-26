%MEAM517
%Greg Campbell & Spencer Folk
%Rework of Morbidi 2016 Paper

%This function provides all the constants given throughout the paper.

function [constants] = constants()

%Choice Variables
% constants.costtype = 'Simple';
% constants.costtype = 'Trapezoid';
constants.costtype = 'Rectangle';
% constants.costtype = 'NeuralNet';
% constants.solvemode = 'fmincon';
constants.solvemode = 'snsolve';

constants.Final_Position = [3,4,6]; %m
constants.yaw = pi/4;

constants.maxI = 200; %Max number of iterations

% Input Values
% constants.N = 30;
constants.N = 100; %Per page 5 (IV - A Scenario 1)
constants.tf = 20; %s (Per page 5 IV - A Scenario 1)
constants.dt = constants.tf/constants.N; %s
constants.nx = 16;
constants.nu = 4;


    % Constants
constants.K_v = 920; %rpm/V
constants.n_B = 2;
constants.rho = 1.225; %kg/m^3
constants.m_B = 0.0055; %kg
constants.m = 1.3; %kg
constants.T_f = 4e-2; %N*m
constants.r = 0.12; %m
constants.l = 0.175; %m
constants.D_f = 2e-4; %N*m*s/rad
constants.eps = 0.004; %m
constants.I_x = 0.081; %kg*m^2
constants.I_y = 0.081; %kg*m^2
constants.I_z = 0.142; %kg*m^2
constants.R = 0.2; % Ohm
constants.C_T = 0.0048;
constants.J_m = 2.9e-6; %kg*m^2
constants.C_Q = 2.3515e-4;
constants.w_max = 1047.197; %rad/s
constants.r_rot = 0.014; %m
constants.m_rot = 0.025; %kg

%Calculated constants
constants.K_E = 9.5493/constants.K_v; %V*s/rad
constants.K_T = constants.K_E; %per page 3.
constants.J_L = 1/4*constants.n_B*constants.m_B*(constants.r-constants.eps)^2;
constants.A = pi*constants.r^2;
constants.k_b = constants.C_T*constants.rho*constants.A*constants.r^2;
constants.k_tao = constants.C_Q*constants.rho*constants.A*constants.r^3;

constants.J = constants.J_m + constants.J_L;  % total moment of inertia
constants.g = 9.8066;

constants.c1 = constants.R*constants.T_f^2/constants.K_T^2;
constants.c2 = constants.T_f/constants.K_T*(2*constants.R*constants.D_f/constants.K_T+constants.K_E);
constants.c3 = constants.D_f/constants.K_T*(constants.R*constants.D_f/constants.K_T+constants.K_E)+2*constants.R*constants.T_f*constants.k_tao/constants.K_T^2;
constants.c4 = constants.k_tao/constants.K_T*(2*constants.R*constants.D_f/constants.K_T+constants.K_E);
constants.c5 = constants.R*constants.k_tao^2/constants.K_T^2;
constants.c6 = 2*constants.R*constants.J*constants.T_f/constants.K_T^2;
constants.c7 = constants.R*constants.J^2/constants.K_T^2;
constants.c8 = constants.J/constants.K_T*(2*constants.R*constants.D_f/constants.K_T+constants.K_E);
constants.c9 = 2*constants.R*constants.J*constants.k_tao/constants.K_T^2;

constants.wh = 912.109; %rad/s

end