%#% Project Finance Oracle - Author Arun Karthikeyan %#%

function [optTheta1 optTheta2 dataToPlot] = NeuralNetworkD1Learning(X, y, initTheta1, initTheta2, lambda, Xtest, ytest)

% Adding the Maximum Iterations for Learning using fmincg in the options parameter %
options  = optimset('MaxIter', 5000);
display("Training for 5000 epochs");

hlSize = size(initTheta1,1);
ipSize = size(initTheta1,2)-1; 

initVectorTheta = [ initTheta1(:) ; initTheta2(:) ];

costFunction = @(t) NeuralNetworkD1Cost(t, ipSize, hlSize, X, y, lambda);

[optVectorTheta cost iterations dataToPlot] = fmincg(costFunction, initVectorTheta, options, Xtest, ytest, hlSize, ipSize);

optTheta1 = reshape(optVectorTheta(1:hlSize * (ipSize + 1)), hlSize, (ipSize + 1));
optTheta2 = reshape(optVectorTheta((1 + (hlSize * (ipSize + 1))):end), 1, (hlSize + 1));

end

