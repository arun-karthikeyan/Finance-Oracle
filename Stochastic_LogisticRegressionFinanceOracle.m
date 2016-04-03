%#% Project Finance Oracle - Author Arun Karthikeyan %#%

% Parameters - None %
function Stochastic_LogisticRegressionFinanceOracle()

% Clearing memory %
clear all;

% timeSeriesData - Name of the file containing the time series data, the function assumes the timeSeriesData to be specified in the reverse chronological order %
timeSeriesData = "octavesap500yahoo.txt";

% days - no of days to be taken into account to predict whether the next day is on the rise or on the fall %
days = 1;
display("%------------------------------------------------------------------------------------------------------------------------------%");
display(sprintf("Each feature has information about previous %d days",days));

% predictionColumn - which column in the timeSeriesData has to be predicted for the next day %
predictionColumn = 4;

% Haven't taken care of file Not found case, might behave unexpectedly in such cases %
timeSeriesData = load(timeSeriesData);

% performing feature scaling and mean normalization %
timeSeriesData = normalizeFeatures(timeSeriesData);

%---%
% Trimming the dataset for testing and debugging purposes %
timeSeriesData = timeSeriesData(1:250,:);

% Storing original size of the input time series data %
[morig norig] = size(timeSeriesData);

% Constructing y from time series data %
% For S&P500 data obtained from yahoo finance, prediction column is the "Closing Price" %
y = timeSeriesData((1:(morig-days)),predictionColumn);

%Constructing X from time series data %
% Constructing the first row %
X = (timeSeriesData((2:(days+1)),:))';
X = X(:)';

%Using Dynamic Programming to Construct the rest of the rows %
for i=(days+2):morig,
X = [ X ; [ X((i-days-1),((norig+1):(days*norig))) timeSeriesData(i,:) ] ];
end;

%Reconstructing the values of y based on prediction column of the previous day | increase = 1 ; no increase = 0;
y = y(:) > X(:,predictionColumn);

% m - No of examples %
% n - No of parameters for each example %
[m n] = size(X);


% Splitting test data %
mtest = round(m*0.2);
%idx = randperm(size(X,1));

% Taking first 20% of the shuffled actual data for testing purpose
Xtest = X(1:mtest,:); 
ytest = y(1:mtest,:);
[mtest ntest] = size(Xtest);

% Taking rest 80% of the shuffled actual data for training purpose
Xtrain = X(mtest+1:end,:); 
ytrain = y(mtest+1:end,:);
[mtrain ntrain] = size(Xtrain);

display(sprintf("Mtrain : %d Ntrain : %d\nMtest : %d Ntest : %d",mtrain,ntrain,mtest,ntest));

% Initializing all weights to 0 %
% n+1 because the bias has been included %
% NOTE : THETA INITIALIZATION TAKES PLACE IN "LogisticRegressionLearning.m" %
% The initialization done here is dummy variably initialization %
Theta = zeros(ntrain+1,1);

% Initializing randome weights to Theta %
%einit = (6^(0.5))/((ntrain+2)^(0.5))
%Theta = (rand(ntrain+1,1)*2*einit)-einit;


%initializing lambda to be 0 %
lambda = 0.0005;
% No of times to run through the entire training data Xtrain %
num_iters = 10000;
display(sprintf("Using %d iterations of stochastic gradient descent",num_iters));
bound = 2;
boundDivider = 200;
figure(1);

r = bound-1;
c = bound+2;

for i=1:r,
	lambda = (i/(bound*boundDivider)) - (1/(bound*boundDivider));
	for j=1:c,
		alpha = j/(bound*boundDivider);
		Theta = zeros(ntrain+1,1);
		display("%------------------------------------------------------------------------------------------------------------------------------%");
		prediction = LogisticRegressionPrediction(Xtest,Theta);
		predAccuracy = mean(prediction==ytest)*100;
		display(sprintf("Test Accuracy with Initial Value of Theta - %0.4f | with lambda - %0.4f | with alpha - %0.4f",predAccuracy,lambda,alpha));

		% The call to Stochastic Gradient Descent method runs Stochastic Gradient Descent on the entire training data, i.e, for 'Mtrain' iterations %
		[Theta dataToPlot] = StochasticGradientDescent(Xtrain, ytrain, lambda,num_iters, alpha, Xtest, ytest);
		prediction = LogisticRegressionPrediction(Xtest,Theta);
		predAccuracy = mean(prediction==ytest)*100;
		display(sprintf("Test Accuracy - %0.4f | with lambda - %0.4f | with alpha - %0.4f",predAccuracy,lambda,alpha));

		if(predAccuracy > 50.0),
		Theta
		end;

		% Plotting the convergence function %
		%subplot(r,c,(c*(i-1))+j);
		subplot(2,2,j);
		plot(dataToPlot,'linestyle','-','ro','markersize',5,'linewidth',1,'markerfacecolor','r');
		xlabel('No of Iterations');
		ylabel('Cost');
		title(sprintf("Alpha %0.4f | Lambda %0.4f | Test Accuracy %0.4f",alpha,lambda,predAccuracy));
		hold on;
		%print -dpng 'myplot.png';
		%display("Image saved");
	end;
end;

display("%------------------------------------------------------------------------------------------------------------------------------%");
print -dpng 'LR3_R6.png';
display("Image saved");
close;
end
