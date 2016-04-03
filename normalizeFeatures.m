%#% Project Finance Oracle - Author Arun Karthikeyan %#%

function X = normalizeFeatures(X)

% Storing the dimensions of the input parameter X %
[m n] = size(X);

% Calculating the column-wise mean of the input parameter X %
meanVals = mean(X);

%Calculating the column-wise standard deviation of the input parameter X %
sdeviation = std(X);

% Assuming Automatic Broadcasting happens properly %
% Performing mean normalization and feature scaling %
X = X - meanVals;
X = X ./ sdeviation;

end
