function [labels, val] = predict_svm(theta, X)

[val,labels] = max(X * theta, [], 2);
labels = labels - 1;

end