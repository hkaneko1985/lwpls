% Demonstration of LWPLS (Locally-Weighted Partial Least Squares)
% Hiromasa Kaneko
clear; close all;

% hyperparameters of LWPLS
component_number = 2;
lambda = 2^-2;

sample_number = 100;
rng('default');
x = 5 * rand( sample_number, 2 );
y = 3 * x(:, 1) .^ 2 + 10 * log(x(:, 2)) + randn(sample_number, 1);
y = y + 0.1 * std(y) * randn(sample_number, 1);
rng('shuffle');
x_train = x(1:70, :);
y_train = y(1:70);
x_test = x(71:end, :);
y_test = y(71:end);

[autoscaled_x_train, mean_of_x_train, std_of_x_train] = zscore( x_train );
[autoscaled_y_train, mean_of_y_train, std_of_y_train] = zscore( y_train );
autoscaled_x_test = (x_test - repmat( mean_of_x_train, size(x_test, 1), 1)) ./ repmat(std_of_x_train, size(x_test, 1), 1);

estimated_y_test = lwpls_prediction(autoscaled_x_train, autoscaled_y_train, autoscaled_x_test, component_number, lambda);
estimated_y_test = estimated_y_test(:, component_number) * std_of_y_train + mean_of_y_train;

figure;
plot( y_test, estimated_y_test, 'b.', 'MarkerSize' , 15 );
hold on;
plot( [-20 90] , [-20 90] , 'k' , 'LineWidth' , 2 );
xlabel('simulated y' , 'FontSize' , 18 , 'FontName', 'Times');
ylabel('estimated y' , 'FontSize' , 18 , 'FontName', 'Times');
axis( [ -20 90 -20 90 ] );
hold off;
set(gcf, 'Color' , 'w' );
set(gca, 'FontSize' ,20 );
axis square;
