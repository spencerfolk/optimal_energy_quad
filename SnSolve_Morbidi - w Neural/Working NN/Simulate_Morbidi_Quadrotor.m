%MEAM517
%Greg Campbell & Spencer Folk
%Rework of Morbidi 2016 Paper

%Function to actually Simulate Quadrotor for specified path

function [X,X0,u,z] = Simulate_Morbidi_Quadrotor

consts = constants();
nx = consts.nx;
nu = consts.nu;
dt = consts.dt;
N = consts.N;
tf = consts.tf;

wh = consts.wh;

Final_Position = consts.Final_Position;
yaw = consts.yaw;

xf = Final_Position(1);
yf = Final_Position(2);
zf = Final_Position(3);

xt0 = [zeros(12,1);wh*ones(4,1)];
xtf = [xf;0;yf;0;zf;zeros(5,1);yaw;0;wh*ones(4,1)];

[z, Aeq, beq, lb, ub, z0] = find_flight_trajectory(xt0, xtf);

u = zeros(nu,0);
X = zeros(nx,0);
X0 = zeros(nx,0);

for i=1:N
   [x_i_inds, u_i_inds] = sample_indices(i);
   
   X(:,i) = z(x_i_inds);
   u(:,i) = z(u_i_inds);
   X0(:,i) = z0(x_i_inds);
end

%Plot x,y,z positions
close all;
figure;

t = tf/N:tf/N:tf;
plot3(X(1,:),X(3,:),X(5,:),'linewidth',2);
hold on;
grid on;
plot3(X0(1,:),X0(3,:),X0(5,:),'linewidth',0.5);
title('x,y,z position for quadrotor');
xlabel('x');
ylabel('y');
zlabel('z');
legend("Optimal Trajectory","Linear Interpolation")
% axis([-5 5 -4 4 0 12]);


figure()
%Recreate Morbidi Plots
subplot(1,3,1)
plot(t, X(1,:), 'r-')
hold on
plot(t, X(3,:), 'k-')
plot(t, X(5,:), 'b-')
legend("X","Y","Z",'location','northwest')
xlabel("Time (s)")

subplot(1,3,2)
plot(t, X(7,:), 'r-')
hold on
plot(t, X(9,:), 'k-')
plot(t, X(11,:), 'b-')
legend("\phi","\theta","\psi",'location','northwest')
xlabel("Time (s)")

subplot(1,3,3)
plot(t, X(2,:), 'r-')
hold on
plot(t, X(4,:), 'k-')
plot(t, X(6,:), 'b-')
legend("xdot","ydot","zdot",'location','northwest')
xlabel("Time (s)")

%Plot Angular Acceleration Inputs
figure()
plot(t, u(1,:), 'r-')
hold on
plot(t, u(2,:), 'g-')
plot(t, u(3,:), 'b-')
plot(t, u(4,:), 'k-')
xlabel("Time (s)")
ylabel("[rad/s^2]")
title("Motor Inputs (Accelerations)")
legend("\alpha_1","\alpha_2","\alpha_3","\alpha_4")

%Plot Motor Velocities
figure; 
hold on;
plot(X(13,:)); 
plot(X(14,:));
plot(X(15,:))
plot(X(16,:))
legend('m1','m2','m3','m4');
title('Motor velocities');

%Additional Plotting
% t = 20/100:20/100:20;
% 
% plot(t,X(13,:));
% hold on;
% plot(t,X(14,:));
% plot(t,X(15,:));
% plot(t,X(16,:));
% 
% figure
% hold on;
% plot(t,X(1,:));
% plot(t,X(3,:));
% plot(t,X(5,:));

end

