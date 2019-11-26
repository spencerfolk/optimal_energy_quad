function power = nnPredict(Theta1, Theta2, zi,sigma,mu)
% This function provides an estimate of power given our neural network
% weights (predetermined) and input data

% Spencer Folk 2019

% Input - X [1x8] - [XY Speed, Z Speed, Roll , Pitch , Yaw , 
%                    motor1 speed, motor2 speed, motor3 rpm, motor4 rpm,
%                    Roll Rate, Pitch Rate, Yaw Rate
%                    motor1 acc, motor2 acc, motor3 acc, motor4 acc]

% Input - Theta1, Theta 2 - Weights of the neural network
        
        xi = zi(1:16);
        ui = zi(17:20);
        
        X0 = zeros(1,16);
        
        X0(1) = norm([xi(2), xi(4)]);
        X0(2) = xi(6);
        X0(3) = xi(7);
        X0(4) = xi(9);
        X0(5) = xi(11);
        X0(6:9) = xi(13:16);
        X0(10) = xi(8);
        X0(11) = xi(10);
        X0(12) = xi(12);
        X0(13:16) = ui(1:4);
 

%Normalize X0 based on sigma and mu:
X_norm = zeros(1,16);
for i=1:length(X_norm)
	if sigma(i) > 0
		X_norm(i) = (X0(i) - mu(i))./sigma(i);
    else
		X_norm(i) = 0;
    end
end

% Neural network structure:
[m, ~] = size(X_norm);
    
% First add ones
X = [ones(m,1) , X_norm];
    
% Move inputs through NN
FirstLayer = activationFunction(X*Theta1',"RELU");
FirstLayer = [ones(m,1) , FirstLayer];  % add null offset in the next layer
    
SecondLayer = activationFunction(FirstLayer*Theta2',"RELU");
power = SecondLayer';

end