%#% Project Finance Oracle - Author Arun Karthikeyan %#%

%This function predicts the value of y for m examples based on the given X %
%The function assumes that the bias hasn't been included %

% Parameters %
% X - the unbiased input data %
% theta - parameters to predict y %
function prediction = LogisticRegressionPrediction(X,theta)

[m n] = size(X);

% Including the bias value %
X = [ones(m,1) X];

% initializing predictions %
prediction = zeros(m,1);

% Makes prediction according the given values of theta assuming the prediction threshold is 0.5 %
prediction = round(sigmoid(X*theta));

end



