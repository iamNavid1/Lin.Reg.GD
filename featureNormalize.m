% ........................................................................

% My template Matlab codes for Linear Regression with multiple variables
% Algorithm: Gradient Descent
% featureNormalize.m
% Navid Salami Pargoo
% 2020

% ........................................................................

% featureNormalize(X) is a preprocessing step which returns a normalized
% version of X where the mean value of each feature is set to 0 and the
% standard deviation is set to 1.

function [X_norm, mu, sigma] = featureNormalize(X)

X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

for i=1:size(X,2);
    mu(i) = mean(X(:,i));
    sigma(i) = std(X(:,i));
    X_norm(:,i) = (X(:,i) - mu(i))./sigma(i);
end

end
