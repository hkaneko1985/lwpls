% Demonstration of LWPLS (Locally-Weighted Partial Least Squares) and decision to set hyperparameters using LWPLS
% Hiromasa Kaneko
clear; close all;

% settings 
max_component_number = 2;
candidates_of_lambda = 2 .^ (-9:5);
number_of_fold_in_cv = 5;

sample_number = 100;
rng('default');
rng(100);
x = 5 * rand( sample_number, 2 );
y = 3 * x(:, 1) .^ 2 + 10 * log(x(:, 2)) + randn(sample_number, 1);
y = y + 0.1 * std(y) * randn(sample_number, 1);
x_train = x(1:70, :);
y_train = y(1:70);
x_test = x(71:end, :);
y_test = y(71:end);

[autoscaled_x_train, mean_of_x_train, std_of_x_train] = zscore( x_train );
[autoscaled_y_train, mean_of_y_train, std_of_y_train] = zscore( y_train );
autoscaled_x_test = (x_test - repmat( mean_of_x_train, size(x_test, 1), 1)) ./ repmat(std_of_x_train, size(x_test, 1), 1);

% grid search + cross-validation
r2cvs = zeros(min(rank(autoscaled_x_train), max_component_number), length(candidates_of_lambda));
min_number = fix( size(x_train,1) / number_of_fold_in_cv );
numbers = repmat( 1:number_of_fold_in_cv, 1, min_number );
numbers = [numbers(:)' 1:(x - min_number * number_of_fold_in_cv)];
indexes_for_division_in_cv = numbers(randperm(size(x_train,1)));
rng('shuffle');
for parameter_number = 1 : length(candidates_of_lambda)
    estimated_y_in_cv = zeros( length(y_train), size(r2cvs, 1));
    for fold_number = 1 : number_of_fold_in_cv
        index_of_training_data = find( indexes_for_division_in_cv ~= fold_number );
        index_of_validation_data = find( indexes_for_division_in_cv == fold_number );

        autoscaled_x_validation_in_cv = autoscaled_x_train( index_of_validation_data , : );
        autoscaled_x_train_in_cv = autoscaled_x_train( index_of_training_data , : );
        autoscaled_y_train_in_cv = autoscaled_y_train( index_of_training_data , : );

        estimated_y_validation_in_cv = lwpls_prediction(autoscaled_x_train_in_cv, autoscaled_y_train_in_cv, autoscaled_x_validation_in_cv, size(r2cvs, 1), candidates_of_lambda(parameter_number));
        estimated_y_in_cv(index_of_validation_data, :) = estimated_y_validation_in_cv * std_of_y_train + mean_of_y_train;
    end
    ss = (y_train - mean_of_y_train )' * (y_train - mean_of_y_train);
    press = diag((repmat(y_train, 1, size(estimated_y_in_cv, 2)) - estimated_y_in_cv)' * (repmat(y_train, 1, size(estimated_y_in_cv, 2)) - estimated_y_in_cv));    
    r2cvs(:, parameter_number) = 1 - press ./ ss;
end
[optimal_component, optimal_lambda_number] = find( r2cvs == max(max(r2cvs)) );
optimal_lambda = candidates_of_lambda(optimal_lambda_number);

estimated_y_validation_in_cv = lwpls_prediction(autoscaled_x_train, autoscaled_y_train, autoscaled_x_test, optimal_component, optimal_lambda);
estimated_y_validation_in_cv = estimated_y_validation_in_cv(:, optimal_component) * std_of_y_train + mean_of_y_train;

figure;
plot( y_test, estimated_y_validation_in_cv, 'b.', 'MarkerSize' , 15 );
hold on;
plot( [-20 100] , [-20 100] , 'k' , 'LineWidth' , 2 );
xlabel('simulated y' , 'FontSize' , 18 , 'FontName', 'Times');
ylabel('estimated y' , 'FontSize' , 18 , 'FontName', 'Times');
axis( [ -20 100 -20 100 ] );
hold off;
set(gcf, 'Color' , 'w' );
set(gca, 'FontSize' ,20 );
axis square;
