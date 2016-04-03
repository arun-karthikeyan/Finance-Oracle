%#% Project Finance Oracle - Author Arun Karthikeyan %#%

% The function computes the cost and gradient of the network D1 %
% Make sure to check the correctness of the implementation with Numerical Gradient Checking %
% The implementation makes use of the sigmoid activation function %

% Parameters - %
% vectorParams - the unrolled version of the original parameters, for compatibility with advanced optimzation techniques (fmincg) %
% ipSize - size of the input layer w/o including bias %
% hlSize - no of activation units in the hiddenlayer %
% X - the input vector X of the original data %
% y - the output y of the original data %

function [J grad] = NeuralNetworkD1Cost(vectorParams, ipSize, hlSize, X, y, lambda)

% Rearranging the parameters in vector params %

Theta1 = reshape(vectorParams(1:hlSize * (ipSize + 1)), hlSize, (ipSize + 1));
Theta2 = reshape(vectorParams((1 + (hlSize * (ipSize + 1))):end), 1, (hlSize + 1));

% m - No of training examples %
m = size(X,1);

% initializing variables for cost and gradient to zero %
J = 0;
gradTheta1 = zeros(size(Theta1));
gradTheta2 = zeros(size(Theta2));

% Feedforward implementation of the network %

finalPrediction = sigmoid(([ones(m,1) (sigmoid([ones(m,1) X] * Theta1'))]*Theta2'));

%-- Taking care of the NaN case, not sure if this is the right fix, but seems to work --%
finalPrediction(find(finalPrediction==1)) = 0.9999999999999999; % To avoid the NaN Case
finalPrediction(find(finalPrediction==0)) = 1.0000e-318; % To avoid the NaN Case

%-- For convenience, will remove later %
newy = y;

% Computing Cost without regularization %
J = (((newy(:))'*log(finalPrediction(:))) + ((1-newy(:))'*log(1-finalPrediction(:))))/(-m);

% Computing regularization part, note that this is a single scalar value %
% the bias parameters have not been included in the computation of the regularized value %
% regularization amounts to zero when lambda is zero %
regularization = (lambda/(2*m))*((Theta1(:,[2 : size(Theta1,2)])(:)'*Theta1(:,[2 : size(Theta1,2)])(:))+(Theta2(:,[2 : size(Theta2,2)])(:)'*Theta2(:,[2 : size(Theta2,2)])(:)));

% The regularized cost value %
J = J + regularization;

% Vectorized implementation of the backpropagation algorithm (efficient) %
% Adding the bias for input X %
newX = [ones(m,1) X];

% Activation Layer 1 is stored as the biased input X %
% The dimensions of this unit should be (m,n+1), where m and n are dimensions of the input vector X %
a1 = newX;

% Input for activation layer 2 %
% The dimensions for this unit should be (m,n) where m = no. of examples, n = no. of units in layer 2 - hlSize %
z2 = (a1 * Theta1');

% Activation Layer 2 with added bias %
% The dimensions of this unit should be (m,n) where m = no. of examples, n = no. of units in layer2 + 1 - hlsize + 1 %
a2 = ( [ones(m,1) sigmoid(z2)] );

% Input for the Output Layer %
% The dimensions of this unit should be (m,n) where m = no. of examples, n = no of class labels (in this case - 1) %
z3 = (a2 * Theta2');

% Output Layer Computation %
% The dimensions of this unit should be (m,n) where m = no. of examples, n = no of class labels (in this case - 1)%
% Note that there is no bias added for the output layer %
% This gives the same result as the final prediction, check consistency and remove the other implementation to improve learning speed %
a3 = sigmoid(z3);

% Computation of error in layer 3 %
% The dimensions of this unit should be (m,n) where m = no. of examples, n = no of class labels (in this case - 1)%
% The dimensions have to be consistent with a3 %
Sdelta3 = a3 - newy;

% Computation of error in layer 2 %
% The dimensions of this unit should be (m,n) where m = no. of examples, n = no. of units in layer2 + 1 - hlsize + 1 | the error for the bias unit is also computed, but isn't used further %
% The dimensions have to be consistent with a2 %
Sdelta2 = (Sdelta3*Theta2 .* [ones(m,1) sigmoidGradient(z2)]);

% Computation of unaveraged-gradient of Theta2 according to the backpropagation algorithm %
% The dimensions are consistent with Theta2 %
Cdelta2 = Sdelta3'*a2;

% Computation of unaveraged-gradient of Theta1 according to the backpropagation algorithm %
% Note that Sdelta2(0) is excluded in this computation i.e., the bias unit error isn't included %
Cdelta1 = Sdelta2(:,[2 : end])'*a1;

% Computation of averaged gradient of Theta1 %
gradTheta1 = Cdelta1/m;

% Computation of averaged gradient of Theta2 %
gradTheta2 = Cdelta2/m;

% Computation of regularization of gradTheta1 %
regTheta1 = [zeros(size(Theta1,1),1) (lambda/m)*Theta1(:,[2:end])];

% Computation of regularization of gradTheta2 %
regTheta2 = [zeros(size(Theta2,1),1) (lambda/m)*Theta2(:,[2:end])];

% Final regularized gradient of Theta1 %
gradTheta1 = gradTheta1 + regTheta1;

% Final regularized gradient of Theta2 %
gradTheta2 = gradTheta2 + regTheta2;

% Unrolling the computed gradients for use with advanced optimization techniques (fmincg) %
grad = [gradTheta1(:) ; gradTheta2(:) ];

end
