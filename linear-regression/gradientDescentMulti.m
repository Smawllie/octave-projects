function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % Set the hypothesis function
    h = (theta'*X')';
    % Using the hypothesis function and the minimized J function, obtain theta
    % the new theta
    % Dimension: theta(nx1); sum((h-y).*X)(1xn)
    theta = theta - alpha*(1/m)*sum((h-y).*X)';
    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);

end % ends for loop
end % ends function
