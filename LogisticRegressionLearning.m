%#% Project Finance Oracle - Author Arun Karthikeyan %#%

% Makes use of fmincg instead of fminunc - fmincg is more efficient than fminunc when large number of parameters are involved %
% Credits to Andrew Ng. for the implementation of fmincg %
% The function is written with the assumtion that bias hasn't been added to X %
function [optimalTheta dataToPlot] = LogisticRegressionLearning(X, y, lambda, Xtest, ytest)

[m n] = size(X);

% Adding bias value, I guess addition of bias isn't necessary here, since the bias has already been added in the cost function%
X = [ones(m,1) X];

% initializing initial theta values to zero %
% Bias has been included %
%initialTheta = zeros(n+1,1);

% Initializing randome weights to Theta %
einit = (6^(0.5))/((n+2)^(0.5))
initialTheta = (rand(n+1,1)*2*einit)-einit;

% initializing optimalTheta values to zero %
% Bias has been included %
optimalTheta = zeros(n+1,1);

% Options setting for fmincg %
options = optimset('GradObj','on','MaxIter',20000);

display("Training for optimal Theta - 20000 epochs");

% Running fmincg to compute the optimal parameters of theta %
% The Polack- Ribiere flavour of conjugate gradients is used to compute search directions, %
% and a line search using quadratic and cubic polynomial approximations and the %
% Wolfe-Powell stopping criteria is used together with the slope ratio method %
% for guessing initial step sizes. %
[optimalTheta cost iterations dataToPlot] = fmincg(@(t)(LogisticRegressionCost(X,y,t,lambda)),initialTheta,options, Xtest, ytest);

end
