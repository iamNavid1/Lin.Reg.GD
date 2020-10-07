% ........................................................................

% My template Matlab codes for Linear Regression with multiple variables
% Algorithm: Gradient Descent
% gradientDescent.m
% Navid Salami Pargoo
% 2020

% ........................................................................

% gradientDescent(X, y, theta, alpha, num_iters) performs gradient descent
% to learn theta through updating theta by taking num-iters gradient steps
% with learning rate alpha

function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    h = X * theta;       % x--> m*n+1  theta--> n+1*1   h---> m*1
    error = h - y;
    theta_change = (alpha/m) * X' * error;
    theta = theta - theta_change;

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
