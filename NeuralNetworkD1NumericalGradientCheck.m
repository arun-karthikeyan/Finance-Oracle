%#% Project Finance Oracle - Author Arun Karthikeyan %#%

function NeuralNetworkD1NumericalGradientCheck(lambda)

if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

% Contructing a small network to check if backprop algo works properly %
ipSize = 3;
hlSize = 5;
m = 5;

Theta1 = sinInitTheta(hlSize, ipSize+1);
Theta2 = sinInitTheta(1, hlSize+1);

X  = sinInitTheta(m, ipSize);
y  = mod([1:m]', 2)';

% Unroll parameters
vecTheta = [Theta1(:) ; Theta2(:)];

costFunction = costFunction = @(t) NeuralNetworkD1Cost(t, ipSize, hlSize, X, y, lambda);

[cost, grad] = costFunc(vecTheta);
[numGradient] = NeuralNetworkD1NumericalGradient(costFunction,vecTheta);

display([grad numGradient]);
display("Make sure the above displayed values are similar");

% The norm of difference between these two values has to be ~ 1e-9 assuming the movementFactor is 1e-4 %
errorDifference = (norm(numGradient-grad)/norm(numGradient+grad));

display(sprintf("A correct implementation of the backprop algo will have a value of ~<= %g",errorDifference)); 

end
