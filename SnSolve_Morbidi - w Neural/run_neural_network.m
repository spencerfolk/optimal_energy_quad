% This script loads in our weights, data, and completes forward propagation
% to make an estimate of power. 

% Spencer Folk 2019
clear
clc
close all

% Load in theta values
load("../weights/optimized_Theta1.txt")
load("../weights/optimized_Theta2.txt")
load("../weights/optimized_Theta3.txt")
load("../weights/optimized_Theta4.txt")

% Generate prediction data set
simSpeed = (0:0.01:20)';
simX = zeros(length(simSpeed), 8);

simX(:,1) = simSpeed;
simX(:,2) = simSpeed.^2;

[simX, sim_mu, sim_sigma] = featureNormalize(simX);

% Now run prediction 

sim_power = nnPredict(optimized_Theta1, optimized_Theta2, optimized_Theta3, optimized_Theta4, simX);

% Undo normalization of parameters

simX = sim_sigma.*simX + sim_mu;  % Undo normalization of speed

sim_speed = simX(:,1);

figure()
plot(sim_speed, sim_power, 'r-', 'linewidth',2)
xlabel("Airspeed (m/s)")
ylabel("Power (W)")
ylim([0,750])
grid on

