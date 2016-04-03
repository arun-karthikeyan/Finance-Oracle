% The function returns random weights for given dimensions based on Andrew Ng's EINIT equation %
function initTheta = randInitTheta(in, out)

einit = (6^(0.5))/((in+out)^(0.5));

initTheta = (rand(in,out)*2*einit)-einit;

end
