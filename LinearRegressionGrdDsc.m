% ........................................................................

% My template Matlab codes for Linear Regression with multiple variables
% Algorithm: Gradient Descent
% LinearRegressionGrdDsc.m
% Navid Salami Pargoo
% 2020

% ........................................................................

% Clear memory 
clear;

% Load the dataset into variables X and y
data = load ('data2n.txt');
X = data(:, 1:end-1);
y = data(:, end);
m = length(y); % # of training sets
n = size(X,2); % # of features

% Print out first 10 examples from the dataset
disp('First 10 eamples from the dataset are:');
disp([X(1:10,:) y(1:10,:)]);

% ========================= 2D/3D plot of input data ==========================

if n == 1
    figure;
    plot(X, y, 'rx', 'MarkerSize', 10); % Plot the data
    xlabel('???variable???'); % Set the x-axis label
    ylabel('???output???'); % Set the y-axis label
elseif n == 2
    figure;
    scatter3(X(:,1), X(:,2), y);
    xlabel('???variable1???'); % Set the x-axis label
    ylabel('???variable2???'); % Set the y-axis label
    zlabel('???output???'); % Set the z-axis label
else
    disp('The dimension of model is greater than 3 => Not possible to be plotted on screen');
end

% ============================================================================

X_org = X;   % keep record of the original X for further plotting 

% Feature Normalization: Scale features and set them to zero mean
if n~=1
    [X, mu, sigma] = featureNormalize(X);
else
    mu = 0;
    sigma = 1;
end

% Add bias intercept term to X
X = [ones(m,1) X];

% Selecting optimal learning rate
% run gradient descent for 50 iterations at the chosen learning rates
alpha_test = [1 0.3 0.1 0.03 0.01];
num_iters_test = 500;

figure;
for i = 1:length(alpha_test)
    theta = zeros(n+1,1);
    [~, J_history_test] = gradientDescent(X, y, theta, alpha_test(i), num_iters_test);
    plot(1:num_iters_test, J_history_test, 'LineWidth', 2);   
    hold on;
    xlabel('Number of iterations');
    ylabel('Cost J');
    clear theta
end
legend ('alpha=1', 'alpha=0.3', 'alpha=0.1', 'alpha=0.03', 'alpha=0.01');
hold off

% =============================================================================
% ================== PAUSE HERE TO CHOOSE THE OPTIMAL ALPHA ===================
% =============================================================================


% Run gradient descent
% Choose some alpha and # iterations values
alpha = 0.01;       % learning rate
num_iters = 1500;   % # of iterations

% Initialize Theta and Run Gradient Descent 
theta = zeros(n+1,1);
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

% Display gradient descent's result
disp('Theta computed from gradient descent are:');
disp(theta);

% Estimate (predict) the output of a certain set of inputs
x = zeros(1, n);   % Replace this matrix with the parameters of prediction
x = (x - mu)./sigma;
x = [1 x];
predict = x * theta;
disp('Predicted OUTPUT of the set of example x (using gradient descent) is:');
disp(predict);    

% ===================== 2D/3D plot of linear fit / Cost =======================

if n == 1
    % Visualizing J(theta_0, theta_1):
    % Grid over which we will calculate J
    theta0_vals = linspace(-abs(3*floor(theta(1))), 3*(ceil(abs(theta(1)))), 30*ceil(abs(theta(1))));
    theta1_vals = linspace(-abs(3*floor(theta(2))), abs(3*ceil(theta(2))), 30*abs(ceil(theta(2))));
    
    % initialize J_vals to a matrix of 0's
    J_vals = zeros(length(theta0_vals), length(theta1_vals));
    
    % Fill out J_vals
    for i = 1:length(theta0_vals)
        for j = 1:length(theta1_vals)
         t = [theta0_vals(i); theta1_vals(j)];    
         J_vals(i,j) = computeCost(X, y, t);
        end
    end
    J_vals = J_vals';
    
    % Surface plot
    figure;
    surf(theta0_vals, theta1_vals, J_vals)
    xlabel('\theta_0'); 
    ylabel('\theta_1');
    zlabel('Cost');
    colorbar;
    
    % Contour plot
    figure;
    % Plot J_vals as 20 contours spaced logarithmically between 0.01 and 1000
    contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
    xlabel('\theta_0'); 
    ylabel('\theta_1');
    colorbar;
    hold on;
    plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
    legend('Cost function contours', 'Assumed Minimum');
    hold off;
    
    % Visualizing linear fit:
    figure;
    plot(X_org(:,1), y, 'rx', 'MarkerSize', 10); % Plot the data
    hold on;
    plot(X_org(:,1), [ones(m, 1) X_org(:,1)]*theta, '-'); % Plot the data
    legend('Training data', 'Linear regression');
    hold off; 
    xlabel('???variable???'); % Set the x-axis label
    ylabel('???output???'); % Set the y-axis label
    
elseif n == 2
    disp ('The dimension of cost function is greater than 3 => Not possible to be plotted on screen');
    
    % Visualizing linear fit:
    figure;
    scatter3(X_org(:,1), X_org(:,2), y);   
    hold on;
    
    % initialize z axis (outputs) to a matrix of 0's
    z = zeros(m, m);
    
    % Fill out z
    for i = 1:m
        for j = 1:m
            z(i,j) = [theta(1)+((X_org(i,1)-mu(1))/sigma(1))*theta(2)+((X_org(j,2)-mu(2))/sigma(2))*theta(3)];
        end
    end
    z = z';
    
    % Surface plot
    surf(X_org(:,1), X_org(:,2), z);
    legend('Training data', 'Linear regression');   
    hold off; 
    xlabel('???variable1???'); % Set the x-axis label
    ylabel('???variable2???'); % Set the y-axis label
    zlabel('???output???'); % Set the z-axis label
    
else
    disp('The dimensions of model and cost function are both greater than 3 => Not possible to be plotted on screen');
end

% ============================================================================




