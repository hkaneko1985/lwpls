# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of Locally-Weighted Partial Least Squares (LWPLS) and decision to set hyperparameters using LWPLS

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import math

from lwpls import lwpls

# settings
max_component_number = 2
candidates_of_lambda_in_similarity = 2 ** np.arange(-9, 6, dtype=float)
number_of_fold_in_cv = 5

sample_number = 100
np.random.seed(10)
x = 5 * np.random.rand(sample_number, 2)
y = 3 * x[:, 0] ** 2 + 10 * np.log(x[:, 1]) + np.random.randn(sample_number)
y = y + 0.1 * y.std(ddof=1) * np.random.randn(sample_number)
x_train = x[0:70, :]
y_train = y[0:70]
x_test = x[70:, :]
y_test = y[70:]

autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)

# grid search + cross-validation
r2cvs = np.empty(
    (min(np.linalg.matrix_rank(autoscaled_x_train), max_component_number), len(candidates_of_lambda_in_similarity)))
min_number = math.floor(x_train.shape[0] / number_of_fold_in_cv)
mod_numbers = x_train.shape[0] - min_number * number_of_fold_in_cv
index = np.matlib.repmat(np.arange(1, number_of_fold_in_cv + 1, 1), 1, min_number).ravel()
if mod_numbers != 0:
    index = np.r_[index, np.arange(1, mod_numbers + 1, 1)]
indexes_for_division_in_cv = np.random.permutation(index)
np.random.seed()
for parameter_number, lambda_in_similarity in enumerate(candidates_of_lambda_in_similarity):
    estimated_y_in_cv = np.empty((len(y_train), r2cvs.shape[0]))
    for fold_number in np.arange(1, number_of_fold_in_cv + 1, 1):
        autoscaled_x_train_in_cv = autoscaled_x_train[indexes_for_division_in_cv != fold_number, :]
        autoscaled_y_train_in_cv = autoscaled_y_train[indexes_for_division_in_cv != fold_number]
        autoscaled_x_validation_in_cv = autoscaled_x_train[indexes_for_division_in_cv == fold_number, :]

        estimated_y_validation_in_cv = lwpls(autoscaled_x_train_in_cv, autoscaled_y_train_in_cv,
                                             autoscaled_x_validation_in_cv, r2cvs.shape[0], lambda_in_similarity)
        estimated_y_in_cv[indexes_for_division_in_cv == fold_number, :] = estimated_y_validation_in_cv * y_train.std(
            ddof=1) + y_train.mean()

    estimated_y_in_cv[np.isnan(estimated_y_in_cv)] = 99999
    ss = (y_train - y_train.mean()).T.dot(y_train - y_train.mean())
    press = np.diag(
        (np.matlib.repmat(y_train.reshape(len(y_train), 1), 1, estimated_y_in_cv.shape[1]) - estimated_y_in_cv).T.dot(
            np.matlib.repmat(y_train.reshape(len(y_train), 1), 1, estimated_y_in_cv.shape[1]) - estimated_y_in_cv))
    r2cvs[:, parameter_number] = 1 - press / ss

best_candidate_number = np.where(r2cvs == r2cvs.max())

optimal_component_number = best_candidate_number[0][0] + 1
optimal_lambda_in_similarity = candidates_of_lambda_in_similarity[best_candidate_number[1][0]]

estimated_y_test = lwpls(autoscaled_x_train, autoscaled_y_train, autoscaled_x_test, optimal_component_number,
                         optimal_lambda_in_similarity)
estimated_y_test = estimated_y_test[:, optimal_component_number - 1] * y_train.std(ddof=1) + y_train.mean()

# r2p, RMSEp, MAEp
print("r2p: {0}".format(float(1 - sum((y_test - estimated_y_test) ** 2) / sum((y_test - y_test.mean()) ** 2))))
print("RMSEp: {0}".format(float((sum((y_test - estimated_y_test) ** 2) / len(y_test)) ** (1 / 2))))
print("MAEp: {0}".format(float(sum(abs(y_test - estimated_y_test)) / len(y_test))))
# yy-plot
plt.rcParams["font.size"] = 18
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_test, estimated_y_test)
max_y = np.max(np.array([np.array(y_test), estimated_y_test]))
min_y = np.min(np.array([np.array(y_test), estimated_y_test]))
plt.plot([min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y)],
         [min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y)], 'k-')
plt.ylim(min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y))
plt.xlim(min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y))
plt.xlabel("simulated y")
plt.ylabel("estimated y")
plt.show()
