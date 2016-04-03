%#% Project Finance Oracle - Author Arun Karthikeyan %#%

function ErrorVsExamples()
j = 1;
m = 50;

for i=1:15,
dataToPlot(j,1) = NeuralNetworkD1FinanceOracle(m);
xAxis(j,1) = m;
m = m+500;
j = j+1;
end;

figure(1);
plot(xAxis,dataToPlot,'linestyle','-','linewidth',2);
xlabel('No of examples');
ylabel('Test Accuracy %');
axis([50 xAxis(end,1)]);
print -dpng 'ErrorVsExamples_2.png';
close;

end
