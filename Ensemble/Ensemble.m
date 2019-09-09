
load('prediction-500-80-50.mat')
load('prediction-dt.mat')
load('prediction_SVM_005.mat')
load('prediction_SVM_001.mat')
load('predictions_LR_005.mat')
load('testing.mat')
load('training.mat')

sum_predictions = predictions_LR_005 + predictions_500_80_50 + predictions_dt;
sum_predictions = sum_predictions + predictions_SVM_001 + predictions_SVM_005;

majority = sum_predictions > 2;

disp(sum(majority, 'all'));

Yt = majority;

submission