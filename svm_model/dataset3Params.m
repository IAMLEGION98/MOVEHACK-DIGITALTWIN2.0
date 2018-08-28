function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns the choice of C and sigma 
%where we select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel

C = 1;
sigma = 0.3;
%predictions = svmPredict(model, Xval);  
%will return the predictions on the cross validation set.
%You can compute the prediction error using mean(double(predictions ~= yval))
%
array = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
for i=1:length(array)
   for j=1:length(array)
     c=array(i);
     s=array(j);
     model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s)); 
     predictions = svmPredict(model, Xval);
     predictions_error(i,j) = mean(double(predictions ~= yval));
     if i==1 && j==1
       m=predictions_error(i,j);
     elseif predictions_error(i,j)<m
       m=predictions_error(i,j);
       C=c;
       sigma=s;
     end 
   end
end






% =========================================================================

end
