function [X_normal, mu, sigma] = featureNormalize(X)
%   featureNormalize(X) returns a normalized version of X

X_normal = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% First obtain the mu and sigma from X using mean and standard deviation
% formulas
mu = sum(X)/length(X); % could also do mean(x)
% sigma: 2x1
sigma = sqrt(sum((X-mu).^2)/(length(X)-1)); % could also do std(X)

% Normalize X using vectorization
X_normal = (X-mu)./sigma;

end
