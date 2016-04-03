%#% Project Finance Oracle - Author Arun Karthikeyan %#%

% This function initializes theta with non-random distinguishable values from the sin function %
% Can be useful for computing initialTheta for debugging purposes in ANN %
% Parameters - %
% in - no. of inputs to the layers %
% out - no. of outputs from the layer %
function initTheta = sinInitTheta(in, out)

% Setting the initialValues of initTheta to zero %
initTheta = zeros(in,out);

% initializing initTheta with sin vals of index and averaging it over 10 %
initTheta = reshape(sin(1:numel(initTheta)),size(initTheta)) / 10;

end
