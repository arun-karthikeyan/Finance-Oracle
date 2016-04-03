%#% Project Finance Oracle - Author Arun Karthikeyan %#%

function numgradient = NeuralNetworkD1NumericalGradientCheck(CostFunction, Theta)

numgradient = zeros(size(Theta));
movement = zeros(size(Theta));
movementFactor = 1e-4;

for i=1:numel(movement),
	movement(i) = movementFactor;
	numgradient(i) = (CostFunction(Theta-movement)+CostFunction(Theta+movement))/(2*movementFactor);
	movement(i) = 0;
end;

end
