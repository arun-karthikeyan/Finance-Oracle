%#% Project Finance Oracle - Author Arun Karthikeyan %#%
function sig = sigmoid(A)

% Code returns an equivalent sigmoid matrix of the input matrix A (i.e) 1/(1+exp(-x)) %
sig = (1.0 + exp(-A)) .\ 1.0;

end
