function g = sigmoid(z)
% Computes sigmoid function of z, returns numpy array.
% This may or may not be useful for polynomial regression... usually only used 
% for classification problems

% Spencer Folk 2019

g = 1./(1+exp(z*-1));

end