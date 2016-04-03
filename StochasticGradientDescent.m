%#% Project Finance Oracle - Author Arun Karthikeyan %#%
%It is assumed that the entire traninig data is passed in here %

function [optimalTheta dataToPlot] = StochasticGradientDescent(X, y, lambda, num_iters, alpha, Xtest, ytest)

[m n] = size(X);

% Adding bias value, I guess addition of bias isn't necessary here, since the bias has already been added in the cost function%
X = [ones(m,1) X];

% Initializing randome weights to Theta %
%einit = (6^(0.5))/((n+2)^(0.5))
%Theta = (rand(n+1,1)*2*einit)-einit;

% Initializing Theta to zero
Theta = zeros(n+1,1);

% initializing optimalTheta values to zero %
% Bias has been included %
optimalTheta = zeros(n+1,1);

% Currently using a fixed value for alpha, later use the formula suggested by Andrew %
costConvergence = 0;
previousCost = 0;
convergedCondition = 10;
currentConvergeCount = 0;
%costAvgIter = 1000;
%dataToPlot = zeros(floor(m/costAvgIter)*num_iters,1);

for j=1:num_iters,
	%display(Theta);
	costConvergence = 0;	
	for i=1:m,
		Xi = X(i,:)';
		yi= y(i);
		[currentCost currentGrad] = LogisticRegressionStochasticCost(Xi, yi, Theta, lambda);
%		display(sprintf("Theta : %0.4f	| CurrentGrad : %0.4f	| alpha = %0.4f\n",Theta,currentGrad,alpha));

		% Updating theta according to current gradient %
		Theta = Theta - (alpha*currentGrad);
		costConvergence = costConvergence + currentCost;

		%if(mod(i,costAvgIter)==0)
		%idx = ((j-1)*floor(m/costAvgIter))+(i/costAvgIter);
		%dataToPlot(idx,1) = costConvergence/costAvgIter;

		%fprintf('Iteration (1x10^3) : %d		| Cost : %0.4f\r',(idx),dataToPlot(idx));
		%costConvergence = 0;
		%end

	end;
	prediction = LogisticRegressionPrediction(Xtest,Theta);
	predAccuracy = mean(prediction==ytest)*100;

	dataToPlot(j,1) = costConvergence/m;
	fprintf('Iteration : %d | Cost : %0.30f | Test Accuracy : %0.4f\r',j,dataToPlot(j,1),predAccuracy);
	
	if exist('OCTAVE_VERSION')
    fflush(stdout);
	end

	if((dataToPlot(j,1)>previousCost) || ((previousCost-dataToPlot(j,1)) < 1e-19 )),
		if(currentConvergeCount < convergedCondition),
		currentConvergeCount = currentConvergeCount + 1;
		else
		display(sprintf("\nConverged at iteration %d, cost %0.30f",j,dataToPlot(j,1)));
		break;
		end;
	else
	currentConvergeCount = 0;	
	end;
	
	previousCost = dataToPlot(j,1);

end;

fprintf('\n');
optimalTheta = optimalTheta + Theta;

end
