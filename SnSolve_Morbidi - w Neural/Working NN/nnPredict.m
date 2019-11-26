function [power] = nnPredict(Theta1,Theta2,Theta3,Theta4,x,sigma,mu)
% This function provides an estimate of power given our neural network
% weights (predetermined) and input data

% Spencer Folk 2019

% Input x_16 = 16x1 state vector

%Constants
consts = constants();

%Create new states 1-8 in 'X0'
X0 = zeros(1,8);

X0(1) = norm([x(2), x(4)]);
X0(2) = norm([x(2), x(4)])^2;
X0(3) = x(6);
xdot2 = (consts.k_b/consts.m)*(sin(x(7))*sin(x(11)) + cos(x(7))*cos(x(11))*sin(x(9)))*sum([x(13) , x(14), x(15), x(16)].^2);
xdot4 = (consts.k_b/consts.m)*(cos(x(7))*sin(x(9))*sin(x(11)) - cos(x(11))*sin(x(7)))*sum([x(13) , x(14), x(15), x(16)].^2);
X0(4) = sqrt(norm([xdot2, xdot4]));
xdot6 = (consts.k_b/consts.m)*(cos(x(9))*cos(x(7)))*sum([x(13) , x(14), x(15), x(16)].^2) - consts.g;
X0(5) = xdot6; 
X0(6) = x(8);
X0(7) = x(10);
X0(8) = x(12);

%Normalize X0 based on sigma and mu:
X_norm = zeros(1,8);
for i=1:length(X_norm)
	if sigma(i) > 0
		X_norm(i) = (X0(i) - mu(i))./sigma(i);
    else
		X_norm(i) = 0;
    end
end

% Neural network structure:
[m, C] = size(X_norm);
    
% First add ones
X = [ones(m,1) , X_norm];
    
% Move inputs through NN
FirstLayer = activationFunction(X*Theta1',"RELU");
FirstLayer = [ones(m,1) , FirstLayer];  % add null offset in the next layer
    
SecondLayer = activationFunction(FirstLayer*Theta2',"RELU");
SecondLayer = [ones(m,1) , SecondLayer]; % add null offset
    
ThirdLayer = activationFunction(SecondLayer*Theta3',"RELU");
ThirdLayer = [ones(m,1) , ThirdLayer];
    
FourthLayer = activationFunction(ThirdLayer*Theta4',"Linear");
power = FourthLayer';

end