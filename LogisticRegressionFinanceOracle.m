%#% Project Finance Oracle - Author Arun Karthikeyan %#%

% Parameters - None %
function LogisticRegressionFinanceOracle()

% Clearing memory %
clear all;

% timeSeriesData - Name of the file containing the time series data, the function assumes the timeSeriesData to be specified in the reverse chronological order %
timeSeriesData = "octave_combined_data_final_senti.txt";

% days - no of days to be taken into account to predict whether the next day is on the rise or on the fall %
days = 1;
display(sprintf("Each feature has information about previous %d days",days));

% predictionColumn - which column in the timeSeriesData has to be predicted for the next day %
predictionColumn = 4;

% Haven't taken care of file Not found case, might behave unexpectedly in such cases %
timeSeriesData = load(timeSeriesData);

% performing feature scaling and mean normalization %
timeSeriesData = normalizeFeatures(timeSeriesData);

%---%
% Trimming the dataset for testing and debugging purposes %
timeSeriesData = timeSeriesData(1:180,:);

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
Xtest = X((1:mtest),:); 
ytest = y((1:mtest),:);
[mtest ntest] = size(Xtest);

% Taking rest 80% of the shuffled actual data for training purpose
Xtrain = X((mtest+1:end),:); 
ytrain = y((mtest+1:end),:);
[mtrain ntrain] = size(Xtrain);



% Initializing all weights to 0 %
% n+1 because the bias has been included %
% NOTE : THETA INITIALIZATION TAKES PLACE IN "LogisticRegressionLearning.m" %
% The initialization done here is dummy variably initialization %
Theta = zeros(ntrain+1,1);

% Initializing randome weights to Theta %
%einit = (6^(0.5))/((ntrain+2)^(0.5))
%Theta = (rand(ntrain+1,1)*2*einit)-einit;


%initializing lambda to be 0 %
lambda = 1;

% Computing the unregularized-cost with the current parameter setting %
% Biased X , y and lambda are passed as parameters to the cost function to compute the initial cost %
%[cost dummy] = LogisticRegressionCost([ones(mtrain,1) Xtrain], ytrain, Theta, lambda);

% Displays the unregularized - cost computed with the initial values of Theta%
%display(sprintf("\nThe unregularized-cost computed with the initial values of theta is %0.4f",cost));

% Computing the regularized - cost with the current parameter setting %
% Manual Initialization of Lambda %
%lambda = 3;
%display(sprintf("The regularization parameter lambda - %d",lambda));

% Biased X , y and lambda are passed as parameters to the cost function to compute the initial cost %
%[cost dummy] = LogisticRegressionCost([ones(mtrain,1) Xtrain], ytrain, Theta, lambda);

% Displays the unregularized - cost computed with the initial values of Theta%
%display(sprintf("The regularized-cost computed with the initial values of theta and lambda = %d is %0.4f",lambda,cost));

% Prediction Accuracy of the training data :P I know this sucks, but it's just to check stuff %
%prediction = LogisticRegressionPrediction(Xtrain,Theta);
%predAccuracy = mean(prediction==ytrain)*100;
%display(sprintf("Prediction Accuracy before training is - %0.4f",predAccuracy));

% looping through different values of lambda %
display("%------------------------------------------------------------------------------------------------------------------------------%");
figure(1);
for i=1:4,
% Training the Logistic Regression Algorithm for finding the optimal value of Theta %
lambda = (i/200) - (1/200);
[Theta dataToPlot] = LogisticRegressionLearning(Xtrain,ytrain,lambda, Xtest, ytest);
prediction = LogisticRegressionPrediction(Xtest,Theta);
predAccuracy = mean(prediction==ytest)*100;
display(sprintf("Test Accuracy - %0.4f | with lambda - %0.4f",predAccuracy,lambda));
display("%------------------------------------------------------------------------------------------------------------------------------%");

	subplot(2,2,i);
	% Averaging Data To Plot by 100 examples %
	
	for d=1:(ceil(size(dataToPlot,1)/100)),
		avgDataToPlot(d,1) = mean(dataToPlot((((d-1)*100)+1):min((d*100),end),1));
	end;

	plot(avgDataToPlot,'linestyle','-');
	xlabel('No of Iterations * 100');
	ylabel('Cost');
	title(sprintf("Lambda %0.4f | Test Accuracy %0.4f",lambda,predAccuracy));
	hold on;
end;
print -dpng 'LogisticRegression_LastMin.png';
close;
sprintf("Image saved");
end
