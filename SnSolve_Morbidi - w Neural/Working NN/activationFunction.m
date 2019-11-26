function z = activationFunction(z, fun)
% This function acts as the activation function for each node in a neural
% network. Use this to pass data through layers during forward propagation

% Spencer Folk 2019 

% Input - z - [nx1] - Input data
% Input - fun - "String" - Selected function

% Output - z - Output data

if fun=="RELU" || fun=="relu"
	% RELU
	z = z.*(z>0);
elseif fun=="linear" || fun=="Linear"
	z = z;
elseif fun=="sigmoid" || fun=="Sigmoid"
	z = sigmoid(z);
end

end