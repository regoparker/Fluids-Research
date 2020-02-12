% Author: Coleman Moss
% Date:   12 February 2020
% WindFlux Machine Learning

clear all; close all; clc;

%% User setup

steps = 150;
percentageTraining = 0.9;

%% Dataset setup

% Create input dataset with a basic trend
input = linspace(0, 20, steps);

plot(input, '--'); hold on;
title('Input');

% Create output as function of input
output = sin(input*0.5);

figure(); hold on; plot(output, 'o-'); plot(input, '--'); hold off;
grid on; title('Output'); legend('Output', 'Input');

% Data sets completed

%% Preprocess Data

% Partition data
numTimeStepsTrain = floor(percentageTraining*numel(output));
dataTrainX = input(1:numTimeStepsTrain+1);
dataTrainY = output(1:numTimeStepsTrain+1);
dataTestX = input(numTimeStepsTrain+1:end);
dataTestY = output(numTimeStepsTrain+1:end);

% Shift data
muX = mean(dataTrainX); muY = mean(dataTrainY);
sigX = std(dataTrainX); sigY = std(dataTrainY);

trainXStnd = (dataTrainX - muX) / sigX;
trainYStnd = (dataTrainY - muY) / sigY;

figure(); hold on;
plot(trainXStnd); plot(trainYStnd); legend('X', 'Y');

%% Set Up Model

% Define LSTM network
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

%Specify training options
options = trainingOptions('adam',...
    'MaxEpochs', 250,...
    'GradientThreshold', 1,...
    'InitialLearnRate', 0.005,...
    'LearnRateDropPeriod', 125,...
    'LearnRateDropFactor', 0.2,...
    'Verbose', 0,...
    'Plots', 'training-progress');

net = trainNetwork(trainXStnd, trainYStnd, layers, options);

muX = mean(dataTestX); sigX = std(dataTestX);
muY = mean(dataTestY); sigY = std(dataTestY);
testXStnd = (dataTestX - muX) / sigX;
testYStnd = (dataTestY - muY) / sigY;

net = predictAndUpdateState(net, trainYStnd);

%% Make Predictions

%Predict first value using training data
[net, YPred] = predictAndUpdateState(net, trainYStnd(end));

%Predict future values using past predictions
numTimeStepsTest = numel(testXStnd);
for i = 2:numTimeStepsTest
    [net, YPred(:,i)] = predictAndUpdateState(net, YPred(:,i-1));
end
 
% YPred = sigX * YPred + muX;

%Plot results
figure();
plot(trainYStnd(1:end-1)); hold on;
idx = numTimeStepsTrain:(numTimeStepsTrain + numTimeStepsTest);
plot(idx, [trainYStnd(numTimeStepsTrain) YPred], '.-');
hold off;
xlabel('Time'); ylabel('Magnitude'); title('Forecast'); legend('Observed', 'Forecast');

figure();
subplot(2, 1, 1);
plot(testYStnd); hold on;
plot(YPred, '.-'); hold off; legend('Observed', 'Forecast'); ylabel('Magnitude');
title('Forecast');

subplot(2, 1, 2); stem(YPred - testYStnd); xlabel('time'); ylabel('Error');
title('RSME');

%Try forecasting with updates
net = resetState(net);
net = predictAndUpdateState(net, trainXStnd(end));

YPred = [];
numTimeStepsTest = numel(testXStnd);

for i = 1:numTimeStepsTest
    [net, YPred(:,i)] = predictAndUpdateState(net, testXStnd(:,i));
end

figure(); subplot(2, 1, 1);
plot(testYStnd); hold on;
plot(YPred, '.-'); hold off;
legend('Observed','Forecasted');
ylabel('Magnitude');
title('Forecasted with Updates');

subplot(2, 1, 2);
stem(YPred - testYStnd);
xlabel('Time'); ylabel('Error');
title('RSME');


