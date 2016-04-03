%#% Project Finance Oracle - Author Arun Karthikeyan %#%

% Parameters - None %
% The network is designed with 1 Hidden Layer (25 Activation units) %
% With Variable input and output %
% Learning Algorithm used is back propagation algorithm %

function testPredAcc = NeuralNetworkD1FinanceOracle(dataSize)

% Clearing memory %
%clear all;
display(dataSize);
% timeSeriesData - Name of the file containing the time series data, the function assumes the timeSeriesData to be specified in the reverse chronological order %
%timeSeriesData = "octavesap500yahoo.txt";
timeSeriesData = "octave_combined_data_final.txt";

% days - no of days to be taken into account to predict whether the next day is on the rise or on the fall %
days = 1;

% predictionColumn - which column in the timeSeriesData has to be predicted for the next day %
predictionColumn = 5; %Since Adj.Close is Col 5 in octave_combined_data_final.txt

% Haven't taken care of file Not found case, might behave unexpectedly in such cases %
timeSeriesData = load(timeSeriesData);

% performing feature scaling and mean normalization %
timeSeriesData = normalizeFeatures(timeSeriesData);

%---%
% Trimming the dataset to last approx last 8-10 years of data, since I don't think all 65 years of training data is helpful anyway %
% Trimming the dataset for testing and debugging purposes %
timeSeriesData = timeSeriesData(1:dataSize,:);

% Storing original size of the input time series data %
[morig norig] = size(timeSeriesData);

%display(sprintf("Morig : %d, Norig : %d",morig,norig));

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

%display(X(1:2,:));
%display(y(1:2,:));

% Splitting test data 30 %%
percentage = 30;
mtest = round(m*(percentage/100));
%idx = randperm(size(X,1));

% Taking first 20% of the shuffled actual data for testing purpose
%Xtest = X(idx(1:mtest),:); 
%ytest = y(idx(1:mtest),:);
Xtest = X(1:mtest,:);
ytest = y(1:mtest,:);

[mtest ntest] = size(Xtest);

% Taking rest 80% of the shuffled actual data for training purpose
%Xtrain = X(idx(mtest+1:end),:); 
%ytrain = y(idx(mtest+1:end),:);
Xtrain = X((mtest+1):end,:);
ytrain = y((mtest+1):end,:);

[mtrain ntrain] = size(Xtrain);

% Initializing the number of desired hidden units for layer 2 %
hlSize = 30;

% Initializing the input vector size %
ipSize = ntrain;

% Initializing the output class size %
% For two classes 0/1 prediction, opSize = 1 should be sufficient %
opSize = 1;

% Initializing initial values of Theta1 %
% Bias parameter has been included %
%initTheta1 = sinInitTheta(hlSize,ipSize+1);
initTheta1 = randInitTheta(hlSize,ipSize+1);

% Initializing initial values of Theta2 %
% Bias parameter has been included %
%initTheta2 = sinInitTheta(opSize,hlSize+1);
initTheta2 = randInitTheta(opSize,hlSize+1);

display("Initial values of Theta Used");
initTheta1
initTheta2

% Initializing the regularization parameter lambda to zero %
lambda = 0;

% Initial prediction with initial set of parameters on the entire input set %
initPredictions = NeuralNetworkD1Prediction(initTheta1, initTheta2, X);

% Initial prediction with initial set of parameters on the test set %
initTestPreds = NeuralNetworkD1Prediction(initTheta1,initTheta2,Xtest);

% Initial prediction accuracy on the entire input set %
initPredAcc = mean(initPredictions==y)*100;

% Initial prediction accuracy on the test set %
initTestPredAcc = mean(initTestPreds==ytest)*100;

display("%------------------------------------------------------------------------------------------------------------------------------%");
display(sprintf("Each feature has information about previous %d days i.e., the first layer takes as input %d features +1 bias",days,days));
display(sprintf("Mtrain : %d Ntrain : %d\nMtest : %d Ntest : %d",mtrain,ntrain,mtest,ntest));
display("%------------------------------------------------------------------------------------------------------------------------------%");
display(sprintf("No. of hidden units used - %d",hlSize));
display("%------------------------------------------------------------------------------------------------------------------------------%");
display(sprintf("Overall Accuracy before training (random initialization of parameters) - %0.4f",initPredAcc));
display(sprintf("Test Accuracy before training (random initialization of parameters) - %0.4f",initTestPredAcc));
display("%------------------------------------------------------------------------------------------------------------------------------%");
bound = 4;
boundDivider = 200;

figure(1);

%for i=1:bound,

%	lambda = i/(boundDivider) - 1/(boundDivider);
	lambda = 0;
	% Learning of the designed Neural Network Architecture using advanced unconstrained optimization technique (fmincg - Polack Ribiere conjugate gradient technique) %
	[optTheta1 optTheta2 dataToPlot] = NeuralNetworkD1Learning(Xtrain, ytrain, initTheta1, initTheta2, lambda, Xtest, ytest);

	% Predictions on the Testdata %
	testPredictions = NeuralNetworkD1Prediction(optTheta1, optTheta2, Xtest);

	% Computation of prediction accuracy on the Test Data %
	testPredAcc = mean(testPredictions==ytest)*100;

	display(sprintf("Test Accuracy - %0.4f | with lambda - %0.4f",testPredAcc,lambda));
	display("%------------------------------------------------------------------------------------------------------------------------------%");
	
	#if(testPredAcc >= 70.0),
	#	optTheta1
	#	optTheta2
	#	#Alert Sound indicating the 70% Accuracy has been reached
	#	#system("paplay $BEEP2");
	#end;	

	%subplot(2,2,i);
	% Averaging Data To Plot by 100 examples %
	
	%for d=1:(ceil(size(dataToPlot,1)/100)),
		%avgDataToPlot(d,1) = mean(dataToPlot((((d-1)*100)+1):min((d*100),end),1));
	%end;

	%plot(avgDataToPlot,'linestyle','-','linewidth',3);
	%xlabel('No of Iterations * 100');
	%ylabel('Cost');
	%title(sprintf("Lambda %0.4f | Test Accuracy %0.4f",lambda,testPredAcc));
	%hold on;

%end;

	%print -dpng 'NND1_LastMin_R7_2.png';
	%display("Image saved");
	%close;
	#Alert sound to indicate end of code
	#system("paplay $BEEP1");
end
