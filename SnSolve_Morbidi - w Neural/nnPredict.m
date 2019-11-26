function power = nnPredict(Theta1,Theta2,Theta3,Theta4,x)
% This function provides an estimate of power given our neural network
% weights (predetermined) and input data

% Spencer Folk 2019

% Input x_16 = 16x1 state vector

%Constants
consts = constants();

X = zeros(1,8);

X(1) = norm([x(2), x(4)]);
X(2) = norm([x(2), x(4)])^2;
X(3) = x(6);
xdot2 = (consts.k_b/consts.m)*(sin(x(7))*sin(x(11)) + cos(x(7))*cos(x(11))*sin(x(9)))*sum([x(13) , x(14), x(15), x(16)].^2);
xdot4 = (consts.k_b/consts.m)*(cos(x(7))*sin(x(9))*sin(x(11)) - cos(x(11))*sin(x(7)))*sum([x(13) , x(14), x(15), x(16)].^2);
X(4) = sqrt(norm([xdot2, xdot4]));
xdot6 = (consts.k_b/consts.m)*(cos(x(9))*cos(x(7)))*sum([x(13) , x(14), x(15), x(16)].^2) - consts.g;
X(5) = xdot6; 
X(6) = x(10);
X(7) = x(8);
X(8) = x(12);


% Neural network structure:
[m, C] = size(X);
    
% First add ones
X = [ones(m,1) , X];
    
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