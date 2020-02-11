%% 
clc
clear all
close all
num = 10000;

set(0,'defaultAxesFontSize',20,'defaultAxesLineWidth',3,'defaultLineMarkerSize',10);
t=0:2:2000;
data= t;


figure('units','normalized','outerposition',[0 0 1 1],'Color',[1,1,1])
plot(t,data)
axis tight;grid on;box on
xlabel('$time$','Interpreter','latex');
ylabel('$Signal$','Interpreter','latex');

%%
train_fraction = 0.02/(num/100000);
numTimeStepsTrain = floor(train_fraction*numel(data));
XTrain = data(1:numTimeStepsTrain);
YTrain = data(2:numTimeStepsTrain+1);
XTest = data(1:end-1);
YTest = data(2:end);

mu = mean(XTrain);
sig = std(XTrain);

XTrain = (XTrain - mu) / sig;
YTrain = (YTrain - mu) / sig;

XTest = (XTest - mu) / sig;
inputSize = 1;
numResponses = 1;
numHiddenUnits = 200;
%% LAYER AND OPTION CHARACTERISTICS

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

maxiterate = 100;
opts = trainingOptions('adam', ...
    'MaxEpochs',maxiterate, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',maxiterate/2, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
%% TRAIN LSTM NETWORK

net = trainNetwork(XTrain,YTrain,layers,opts);

%% UPDATING PREDICTION BASED ON PREVIOUS PREDICTION VALUE

net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,numel(YTrain));

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(1,i)] = predictAndUpdateState(net,YPred(i-1));
end

YPred = sig*YPred + mu;
rmse = sqrt(mean((YPred-YTest).^2))

figure
plot(data(1:numTimeStepsTrain))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Month")
ylabel("Cases")
title("Windspeed")
legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Windspeed")
title("Forecast")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)

%% UPDATIING PREDICTION BASED ON OBSERVED VALUE

net = resetState(net);
net = predictAndUpdateState(net,XTrain);

YPred = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [net,YPred(1,i)] = predictAndUpdateState(net,XTest(i));
end

YPred = sig*YPred + mu;
rmse = sqrt(mean((YPred-YTest).^2))

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Windspeed")
title("Forecast with Updates")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)
