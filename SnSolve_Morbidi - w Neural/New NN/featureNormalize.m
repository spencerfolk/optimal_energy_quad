function [X_norm, mu, sigma] = featureNormalize(X)
% Performs necessary normalization of input data in order to pass through
% the network

% Spencer Folk

% Input - simX [nx8] - Input simulation data
% Output - simX [nx8] - Normalized simulation data
% Output - sim_mu - mean of input data for recovering data
% Output - sim_sigma - stdev of input data for recovering data

[R, C] = size(X);

X_norm = X;
mu = zeros(1,C);
sigma = zeros(1,C);

for i = 1:C
	% For each column... compute mean, std, and normalize the column
	mu(i) = mean(X(:,i));
	sigma(i) = std(X(:,i));
	if sigma(i) > 0
		X_norm(:,i) = (X(:,i) - mu(i))./sigma(i);
    else
		X_norm(:,i) = 0;
    end
end

end