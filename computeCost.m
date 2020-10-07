% ........................................................................

% My template Matlab codes for Linear Regression with multiple variables
% Algorithm: Gradient Descent
% featureNormalize.m
% Navid Salami Pargoo
% 2020

% ........................................................................

% computeCost(X) computes the cost of using theta as the parameter for
% linear regression t fit the data points in X and y

function J = computeCost(X, y, theta)

% Initialize some useful values
m = length(y); % number of training examples

h = X * theta;       % x--> m*n+1  theta--> n+1*1   h---> m*1
error = h - y;
J = 1/(2*m) * error' * error;

end
