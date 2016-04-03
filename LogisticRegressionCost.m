%#% Project Finance Oracle - Author Arun Karthikeyan %#%

% Parameters %
% X - The biased input data %
% y - Output Labels for the given input data %
% lambda - The regularization parameter %
% Theta - parameter weights to compute the cost, theta is assumed to have the bias value %

function [J grad] = LogisticRegressionCost(X, y, Theta, lambda)
%	display("Cost Function called");
[m n] = size(X);

% Gradient to be returned for unconstrained optimization %
grad = zeros(size(Theta));

hypothesis = sigmoid(X*Theta);

%-- Taking care of the NaN case, not sure if this is the right fix, but seems to work --%
hypothesis(find(hypothesis==1)) = 0.9999999999999999; % To avoid the NaN Case
hypothesis(find(hypothesis==0)) = 1.0000e-318; % To avoid the NaN Case

part1 = y'*log(hypothesis);
part2 = (1-y)'*log(1-hypothesis);

%Cost is computed according to the Logistic Regression Cost Function%
J = (-part1-part2)/m;

% Regularization parameter added to penalize parameter Theta %
% if lambda is passed as 0, regularization is not performed %
regularization = ((Theta'*Theta)-(Theta(1,1)*Theta(1,1)))*(lambda/(2*m));

J = J + regularization;
 
% Gradient is computed as the partial derivative of the cost function with respect to each individual parameter theta %
% Gradient has the same number of dimensions as Theta %
tempgrad = grad + ((X'*(sigmoid(X*Theta)-y))/m);
regularizationGrad = grad + ((lambda*([0;[Theta([2:length(Theta)])]]))/m);
grad = grad + tempgrad + regularizationGrad;

grad = grad(:);
end
