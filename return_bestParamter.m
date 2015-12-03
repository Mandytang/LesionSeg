function [bestp, accuracy] = return_bestParamter(X, Y, param, folds, K)

bestp = 0;
accuracy = zeros(length(param),1);

for i = 1:length(param)
    fprintf('param = %f, ', param(i));
    subaccuracy = zeros(K,1);
    for j = 1:K
        [trainSet, testSet, labelsTrain, labelsTest] = generateSets(X, Y, folds, j);        
        [theta] = train_svm(trainSet, labelsTrain, param(i));
        p = predict_svm(theta, testSet);
        subaccuracy(j) = mean(double(p==labelsTest));
    end
    accuracy(i) = mean(subaccuracy);
    fprintf('accuracy = %f \n', accuracy(i));
end

[~, indexMax] = max(accuracy);
bestp = param(indexMax);
fprintf('The best lambda = %f \n', bestp);
end