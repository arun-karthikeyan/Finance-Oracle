%#% Project Finance Oracle - Author Arun Karthikeyan %#%
% Assuming X is the input vector (for one example) %

function [J grad] = LogisticRegressionStochasticCost(X,y,Theta,lambda)

n = length(X);
TTX = Theta'*X;
STTX = sigmoid(TTX);
%-- Taking care of the NaN case, not sure if this is the right fix, but seems to work --%
if(STTX==1)
STTX = 0.9999999999999999; % To avoid the NaN Case
elseif(STTX==0)
STTX = 1.0000e-318; % To avoid the NaN Case
end;

J = -(y*log(STTX) + (1-y)*log(1-STTX));
costRegularization = (lambda/2)*((Theta'*Theta)-(Theta(1,1)*Theta(1,1)));
J = J + costRegularization;

grad = (STTX-y) * X;
gradRegularization = lambda*(Theta);
grad = grad + gradRegularization;

end
