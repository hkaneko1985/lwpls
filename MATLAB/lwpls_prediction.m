function estimated_y_test = lwpls_prediction(x_train, y_train, x_test, max_component_number, lambda)
%LWPLS_PREDICTION Predict y-values of test samples using LWPLS
%  Hiromasa Kaneko
%
% --- input ---
% x_train : autoscaled m x n matrix,
%   X-variables of training data, m is the number of training sammples and n is the number of X-variables
% y_train : autoscaled m x 1 vector,
%   A Y-variable of training data
% x_test : k x n matrix autoscaled with training data,
%   X-variables of test data, k is the number of test samples
% max_component_number : scalar,
%   number of maximum components
% lambda : scalar,
%   parameter in similarity matrix
%
% --- output ---
% estimated_y_test : k x 1 vector
%   estimated y-values of test data
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


estimated_y_test = zeros(size(x_test, 1), max_component_number); 

for test_sample_number = 1 : size(x_test, 1)
    query_x_test = x_test(test_sample_number, : );
    distance = sqrt(sum((x_train - repmat(query_x_test, size(x_train, 1), 1)) .^2, 2));
    similarity = diag(exp(-distance / std(distance) / lambda));
    
    y_w = y_train' * diag(similarity) / sum(diag(similarity));
    x_w = (x_train' * diag(similarity) / sum(diag(similarity)))'; 
    centered_y = y_train - y_w;
    centered_x = x_train - ones(size(x_train, 1), 1) * x_w;
    centered_query_x_test = query_x_test - x_w;
    estimated_y_test(test_sample_number, :) = estimated_y_test(test_sample_number, :) + y_w;
    for component_number = 1 : max_component_number
        w_a = centered_x' * similarity*centered_y / norm(centered_x' * similarity*centered_y);
        t_a = centered_x * w_a;
        p_a = centered_x' * similarity * t_a / (t_a' * similarity * t_a);
        q_a = centered_y' * similarity * t_a / (t_a' * similarity * t_a);
        t_q_a = centered_query_x_test * w_a;
        estimated_y_test(test_sample_number, component_number:end) = estimated_y_test(test_sample_number, component_number:end) + t_q_a * q_a;
        if component_number ~= max_component_number
            centered_x = centered_x - t_a * p_a';
            centered_y = centered_y - t_a * q_a';
            centered_query_x_test = centered_query_x_test - t_q_a * p_a';
        end
    end
end

end


