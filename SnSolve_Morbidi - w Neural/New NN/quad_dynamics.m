% This function will take input commands on the motor and numerically
% integrate to get the response based on the dynamic model presented in
% Morbidi et. al. 

% Spencer Folk 2019
clear
clc
close all

% Numerically integrate our quadrotor with the time interval:
t_0 = 0;
t_f = 50;
dt = 0.001;
t = (t_0:dt:t_f);

% and input
alpha = zeros(4, length(t));

% with initial conditions
x = zeros(16, length(t));
x(5,1) = 50;
x(13:16,1) = 912.109;

for i = 2:length(t)
   % Numerically integrate all state vectors using our function for xdot
   x(:,i) = x(:,i-1) + quad_model(x(:,i-1), alpha(:,i-1))*dt;
%    alpha(:,i) = -0.05*(x(5,i) - x(5,1));
end

figure()
subplot(1,2,1)
plot(t,x(5,:),'k-','linewidth',2)
xlabel("Time (s)")
title("Z Height (m)")
ylim([0,100])
grid on

subplot(1,2,2)
plot(t,alpha(1,:),'r-','linewidth',2)
xlabel("Time (s)")
title("Motor 1 Command (rad/s^2)")
grid on

figure()
plot(t,x(13,:),'k-','linewidth',2)
xlabel("Time (s)")
ylabel("Motor 1 speed")
grid on


function xdot = quad_model(x, alpha)
% Input - x [16 x 1] - [x, xdot, y, ydot, z, zdot, phi, phidot, theta,
%                       thetadot, psi, psidot, omega_1, omega_2, omega_3, 
%                       omega_4]'

% Input - alpha [4 x 1] - [alpha_1, alpha_2, alpha_3]' - Motor
% accelerations

% Output - xdot [16 x 1] - derivatives of states

% Constants

K_v = 920; %rpm/V
n_B = 2;
rho = 1.225; %kg/m^3
m_B = 0.0055; %kg
m = 1.3; %kg
T_f = 4e-2; %N*m
r = 0.12; %m
l = 0.175; %m
D_f = 2e-4; %N*m*s/rad
eps = 0.004; %m
I_x = 0.081; %kg*m^2
I_y = 0.081; %kg*m^2
I_z = 0.142; %kg*m^2
R = 0.2; % Ohm
C_T = 0.0048;
J_m = 2.9e-6; %kg*m^2
C_Q = 2.3515e-4;
w_max = 1047.197; %rad/s
r_rot = 0.014; %m
m_rot = 0.025; %kg

%Calculated constants
K_E = 9.5493/K_v; %V*s/rad
K_T = K_E; %per page 3.
J_L = 1/4*n_B*m_B*(r-eps)^2;
A = pi*r^2;
k_b = C_T*rho*A*r^2;
k_tao = C_Q*rho*A*r^3;

J = J_m + J_L;  % total moment of inertia
g = 9.8066;

% Compute derivatives

xdot = zeros(size(x));

xdot(1) = x(2);
xdot(2) = (k_b/m)*(sin(x(7))*sin(x(11)) + cos(x(7))*cos(x(11))*sin(x(9)))*sum([x(13) , x(14), x(15), x(16)].^2);
xdot(3) = x(4);
xdot(4) = (k_b/m)*(cos(x(7))*sin(x(9))*sin(x(11)) - cos(x(11))*sin(x(7)))*sum([x(13) , x(14), x(15), x(16)].^2);
xdot(5) = x(6);
xdot(6) = (k_b/m)*(cos(x(9))*cos(x(7)))*sum([x(13) , x(14), x(15), x(16)].^2) - g;
xdot(7) = x(8);
xdot(8) = ((I_y - I_z)/I_x)*x(10)*x(12) + (l*k_b/I_x)*(x(14)^2 - x(16)^2) - (J/I_x)*x(10)*(x(13) - x(14) + x(15) - x(16));
xdot(9) = x(10);
xdot(10) = ((I_z - I_x)/I_y)*x(8)*x(12) + (l*k_b/I_y)*(x(15)^2 - x(13)^2) + (J/I_y)*x(8)*(x(13) - x(14) + x(15) - x(16));
xdot(11) = x(12);
xdot(12) = ((I_x-I_y)/I_z)*x(8)*x(10) + (k_tao/I_z)*(x(13)^2 - x(14)^2 + x(15)^2 - x(16)^2);
xdot(13) = alpha(1);
xdot(14) = alpha(2);
xdot(15) = alpha(3);
xdot(16) = alpha(4);

end