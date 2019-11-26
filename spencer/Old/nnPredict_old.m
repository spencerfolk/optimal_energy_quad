function power = nnPredict(Theta1, Theta2, Theta3, Theta4, X)
% This function provides an estimate of power given our neural network
% weights (predetermined) and input data

% Spencer Folk 2019

% Input - X [1x8] - 
% Input - Theta1, ... Theta4 - Weights of the neural network

% Neural network structure:
[m, C] = size(X);
    
% First add ones
X = [ones(m,1) , X];
    
% Move inputs through NN
FirstLayer = activationFunction(X*Theta1',"RELU");
FirstLayer = [one
    s(m,1) , FirstLayer];  % add null offset in the next layer
    
SecondLayer = activationFunction(FirstLayer*Theta2',"RELU");
SecondLayer = [ones(m,1) , SecondLayer]; % add null offset
    
ThirdLayer = activationFunction(SecondLayer*Theta3',"RELU");
ThirdLayer = [ones(m,1) , ThirdLayer];
    
FourthLayer = activationFunction(ThirdLayer*Theta4',"Linear");
power = FourthLayer';

end