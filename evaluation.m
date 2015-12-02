function [accuracy,precision,recall,f1] = evaluation(y_pred, y_true, label_p, label_n)
% labelp: the label for positive samples
% labeln: the label for negative samples
tp = sum((y_true == y_pred) & (y_true == label_p));
%tn = sum((y_true == y_pred) & (y_true == label_n));
fp = sum((y_true ~= y_pred) & (y_pred == label_p));
fn = sum((y_true ~= y_pred) & (y_pred == label_n));

accuracy =  1 - sum(y_pred ~= y_true) / length(y_true);
fprintf('accuracy %f%%\n', 100 * accuracy);

precision = tp/(tp+fp);
fprintf('precision %f%%\n', 100 * precision);

recall = tp/(tp+fn);
fprintf('recall %f%%\n', 100 * recall);

f1 = 2*precision*recall/(precision+recall);
fprintf('f1 %f%%\n', 100 * f1);
