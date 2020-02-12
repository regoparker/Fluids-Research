% Author: Coleman Moss
% Date:   12 February 2020
% WindFlux Machine Learning

clear all; close all; clc;

% Preprocess Data

% data = chickenpox_dataset;
% data = [data{:}];

data = linspace(0,10,100);
randoms = rand(1, length(data));
data = sin(data) + randoms*0.25;

figure(); plot(data); title('Data');
xlabel('Time'); ylabel('Value');

numTimeStepsTrain = floor(0.9*numel(data));
dataTrain = data(1:numTimeStepsTrain + 1);
dataTest = data(numTimeStepsTrain + 1:end);

mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;

% Training response is dataset shifted by one timestep forward

XTrain = dataTrainStandardized(1:end - 1);
YTrain = dataTrainStandardized(2:end);

% Define LSTM network

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

% Specify Training options

options = trainingOptions('adam', ...
    'MaxEpochs', 250, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 125, ...
    'LearnRateDropFactor', 0.2, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

net = trainNetwork(XTrain, YTrain, layers, options);

dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);

net = predictAndUpdateState(net, XTrain);
[net, YPred] = predictAndUpdateState(net, YTrain(end));



numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net, YPred(:,i)] = predictAndUpdateState(net, YPred(:,i-1), 'ExecutionEnvironment','cpu');
end

YTest = dataTest(2:end);
YPred = sig*YPred + mu;

rmse = sqrt(mean((YPred-YTest).^2));

figure();
plot(dataTrain(1:end-1)); hold on;
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx, [data(numTimeStepsTrain) YPred], '.-');
hold off;
xlabel('Month'); ylabel('Cases'); title('Forecast'); legend('Observed', 'Forecast');


figure(); 
subplot(2, 1, 1); 
plot(YTest); hold on;
plot(YPred, '.-'); hold off; legend('Observed', 'Forecast'); ylabel('Cases');
title('Forecast');

subplot(2, 1, 2); stem(YPred - YTest); xlabel('Month'); ylabel('Error');
title('RMSE');

net = resetState(net);
net = predictAndUpdateState(net,XTrain);

YPred = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end

YPred = sig*YPred + mu;

rmse = sqrt(mean((YPred-YTest).^2));

figure(); subplot(2,1,1);
plot(YTest);
hold on;
plot(YPred,'.-');
hold off;
legend("Observed", "Predicted");
ylabel("Cases");
title("Forecast with Updates");

subplot(2,1,2);
stem(YPred - YTest);
xlabel("Month");
ylabel("Error");
title("RMSE = ");