function power = nnPredict(Theta1, Theta2, X)
% This function provides an estimate of power given our neural network
% weights (predetermined) and input data

% Spencer Folk 2019

% Input - X [1x8] - [XY Speed, Z Speed, Roll , Pitch , Yaw , 
%                    motor1 speed, motor2 speed, motor3 rpm, motor4 rpm,
%                    Roll Rate, Pitch Rate, Yaw Rate
%                    motor1 acc, motor2 acc, motor3 acc, motor4 acc]

% Input - Theta1, Theta 2 - Weights of the neural network

% Neural network structure:
[m, C] = size(X);
    
% First add ones
X = [ones(m,1) , X];
    
% Move inputs through NN
FirstLayer = activationFunction(X*Theta1',"RELU");
FirstLayer = [ones(m,1) , FirstLayer];  % add null offset in the next layer
    
SecondLayer = activationFunction(FirstLayer*Theta2',"RELU");
power = SecondLayer';

end