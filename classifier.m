function [y_pred, eval] = classifier(X_train, labels_train, X_test, labels_test, var, numTrees)

switch (var)
    case 'svm'
        model = libsvmtrain(labels_train, X_train);
        [y_pred, accuracy] = libsvmpredict(labels_test, X_test, model);
        [accuracy,precision,recall,f1] = evaluation(y_pred, labels_test, label_p, label_n);
    case 'logistic'
        model = mnrfit(X_train,labels_train+1);
        y_pred = mnrval(model,X_test);
        
    case 'rf'
        model = TreeBagger(numTrees, X_train, labels_train, 'Cost', [0 1; 10 0], 'NumPredictorsToSample', 50);
        [labels, y_pred] = predict(model, X_test);

end