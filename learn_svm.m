function [theta] = learn_svm(X, Y, Cs)
% input: X: [m n], m samples, n features (include biase)
%        Y: labels, 0 negative, 1 positive. If not, please change function generateFolds

bestp = 0;
theta = 0;

% generate K folds----------
K = 5;
folds = generateFolds(X, Y, K);
bestp = return_bestParamter(X, Y, Cs, folds, K);

theta = train_svm(X, Y, bestp);

end