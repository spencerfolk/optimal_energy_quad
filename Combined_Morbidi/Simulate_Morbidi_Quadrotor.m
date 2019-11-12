%MEAM517
%Greg Campbell & Spencer Folk
%Rework of Morbidi 2016 Paper

%Function to actually Simulate Quadrotor for specified path

function [X,X0,u] = Simulate_Morbidi_Quadrotor

consts = constants();
nx = 16;
nu = 4;

wh = consts.wh;

Final_Position = [4;5;6]; %m

xf = Final_Position(1);
yf = Final_Position(2);
zf = Final_Position(3);

xt0 = [zeros(12,1);wh*ones(4,1)];
xtf = [xf;0;yf;0;zf;zeros(5,1);pi/4;0;wh*ones(4,1)];

N = 100; %Per page 5 (IV - A Scenario 1)
tf = 20; %s (Per page 5 IV - A Scenario 1)
dt = tf/N; %s

[z, Aeq, beq, lb, ub, z0] = find_flight_trajectory(xt0, xtf, N, dt);

u = zeros(nu,0);
X = zeros(nx,0);
X0 = zeros(nx,0);

for i=1:N
   [x_i_inds, u_i_inds] = sample_indices(i, nx, nu);
   
   X(:,i) = z(x_i_inds);
   u(:,i) = z(u_i_inds);
   X0(:,i) = z0(x_i_inds);
end

%Plot x,y,z positions
figure;
plot3(X(1,:),X(3,:),X(5,:));
hold on;
grid on;
plot3(X0(1,:),X0(3,:),X0(5,:));
title('x,y,z position for quadrotor');
xlabel('x');
ylabel('y');
zlabel('z');
axis([-5 5 -4 4 0 12]);

end

