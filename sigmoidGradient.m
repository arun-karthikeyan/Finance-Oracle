%#% Project Finance Oracle - Author Arun Karthikeyan %#%

% Parameters : %
% z - The input to the sigmoid activation funtion %
% The function returns the derivative of the sigmoid activation function with z as its input --> g'(z) %
function zDash = sigmoidGradient(z)

zDash = sigmoid(z) .* (1 - sigmoid(z));

end
