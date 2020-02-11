clc
clear all
close all


% wind speed SCADA data
t = [1:5000];
data = sind(t) + rand(1,length(t))/2;

for i=1:numel(data)
    if isnan(data(i))==1
        data(i) = 0;
    end
end

figure()
plot(data)
xlabel("Time (every 10min)")
ylabel("Wind Speed (m/s)")

Signal = data;
%% POD of signal
colonna=2;         % 
Fsamp=200;        % Frequency in Hz
en=100;            % 
DeltaFreq=1;         % for convolution
Nfoto=1;          
Promt=0;% Yes/NO? (1/0)      input('for promtmode option Yes/NO? (1/0)');
%
Name='Signal';
nt = 2000; % value t
dt=1/Fsamp; % step through time 0.005 sec at a time = 1/Fsamp
fs=1/dt; % sampling frequency
t=0:dt:dt*(nt-1);
freq=(0:nt/2-1)*fs/nt;

Thr=ceil(Fsamp/DeltaFreq);

SigMean=nanmean(Signal);
% % Calculate the fluctuation of signal
Signal=Signal-SigMean;
% 
%   
x=Signal;
figure()
plot(Signal)
hold on
plot(x)

W=30;
nsigma=5;
% eliminates signal values more than 5 std deviations from mean
[y,W,xmedian,xsigma] = hampel(x,W,nsigma);
Signal=y;

f1=Fsamp/length(Signal)*(0:length(Signal)/2);
fsignal=fft(Signal)/length(Signal);
figure
plot(f1(2:end),2*abs(fsignal(2:floor(length(Signal)/2)+1)),'r');
grid on
xlabel('f(Hz)')
ylabel('A^2')

%%
Fcut=Fsamp-0.0001*Fsamp;
Filter=input('Do you want to filter (Yes/NO)? (1/0) ');
close all
if Filter==1
    d1 = designfilt('lowpassiir','FilterOrder',8.0,'HalfPowerFrequency',Fcut/Fsamp/2.0,'DesignMethod','butter');
    % cut out high frequencies and reduce noise in signal with Butterworth
    % second order section filter method to make low pass filter
    Signal2 = filtfilt(d1,double(Signal));
    
%     figure
%     plot(Signal)
%     hold on
%     plot(Signal2,'r')
%     box on; grid on
%     set(gcf,'color',[1 1 1])
     fsignal2=fft(Signal2)/length(Signal2);
    
    figure
  loglog(f1(2:end),abs(fsignal(2:floor(length(Signal)/2)+1)),'b');
    hold on;
  loglog(f1(2:end),abs(fsignal2(2:floor(length(Signal)/2)+1)),'r');

    box on; grid on
    set(gcf,'color',[1 1 1])
  xlabel('Frequency')
ylabel('Amplitude')
legend('Original','Low-pass')

figure()

plot(Signal)
hold on
plot(Signal2)
xlabel('Time')
ylabel('Velocity (m/s)')
legend('Original','Low-pass')
end
%% saving variable
  Signalold=Signal;
  
%%
N=100; % half of sampling frequency b/c Fsamp is twice the highest frequency measurable
% DeltaFreq2=input('Enter the Delta Freq x separate ways= ');

% divide data into chunks that are DeltaFreq2 long for each snapshot
DeltaFreq2=N*unique(diff(f1));  % 
% N_period = No. of samples in each snapshot
% number of periods to separate chunks of data
Nperiod=floor(ceil(Fsamp/DeltaFreq2(1))/2)*2;   % 
Nperiod=2500;

% N_snap  % N_M=No. of snapshot
Nmisure=floor(length(Signal)/Nperiod);    % No. of Snapshot=Nmisure   % N samples / period (minimum N = Fsamp / fmin)
X=reshape(Signal(1:Nperiod*Nmisure),Nperiod,Nmisure);
disp(['Minimum number of Snapshots available : ' num2str(Nmisure)]) 
disp(['Maximum number of Snapshots available : ' num2str(length(Signal)-Nperiod+1)]) %Nmeasure*(Nperiod-1)-1)
Nmisure2=input('Number of Snapshots (minimum threshold) required= ');

% in case too small of snapshot # is selected, the minimum value necessary
% will be used
if Nmisure2 <= Nmisure 
 M = Nmisure;
 X1=reshape(Signal(1:Nperiod*Nmisure),Nperiod,Nmisure);
else
% Snapshot Extraction - window translation
 % meaning of kappa = counting how many iterations in loop
    kappa = 0;
  M = Nmisure;
  while M < Nmisure2
  kappa = kappa + 1;
    M = M*2-1;
  end
  X1=[];
  for i=1:M 
      if (i-1)*ceil(Nperiod/2^kappa)+Nperiod <= length(Signal)
          Mok = i;
          X1(:,i) = Signal(ceil((i-1)*Nperiod/2^kappa+1):ceil((i-1)*Nperiod/2^kappa)+Nperiod);
      end   
  end
    M = Mok;
    disp(['Number of Snapshots extracted : ' num2str(M)])   
end


% POD

% difference between Classical and Snapshot messages - depends on which
% correlation matrix will be smaller
if M > Nperiod
    %clc
    disp('POD Classical (Nmisure > Nperiodo)')
    [phi, lam, Xmean, nbasis]=POD_original(X1,en);
else
    %clc
    disp('POD Snapshot (Nperiodo > Nmisure)')
    %calculate the total energy and correlation matrix
    [phi, lam, Xmean, nbasis]=POD_snapshot(X1,en);

end

%%
% Energy of the modes

figure
plot([1:nbasis],lam(1:nbasis)/sum(lam(1:nbasis))*100,'o')
xlabel('Number of POD eigenvalues')
ylabel('Energy captured')
box on; grid on
set(gcf,'color',[1 1 1])
box on; grid on
% saveas(gcf,'Fig1.fig')
% saveas(gcf,'Fig1.png')
%close


%clc
Modi=input(' How many mode to visualize? = ');
% plot the mode

figure
for i=1:Modi
    subplot(Modi,1,i)
    plot(phi(:,i))
    hold all
    box on; grid on
    ylabel(strcat('Mode_',num2str(i)))
end
xlabel('Samples')
set(gcf,'color',[1 1 1])
box on; grid on
% saveas(gcf,'Fig2.fig')
% saveas(gcf,'Fig2.png')

figure
for i=1:2:Modi
    subplot(Modi,1,i)
    plot(phi(:,i))
    hold on
    plot(phi(:,i+1))
    hold all
  
    ylabel(strcat('Mode_',num2str(i)))
end
xlabel('Samples')

% saveas(gcf,'Fig3.fig')
% saveas(gcf,'Fig3.png')

%%
% PROJECTION OF MEASURES ON THE WAYS
figure()
for i=1:nbasis
    for j=1:Nmisure
        A(j,i)=(X(:,j))'*phi(:,i);
    end
    
    plot(A(:,i))
    hold on
end
% figure()
% B=permute(A,[2 1]);
% plot(B(:))
xlabel('Sample')
ylabel('a_j')
%% Cummulative mode choice
nbasis=input('Select the total number of modes you want to consider: ')
Xric=zeros(Nperiod,Nmisure);
for i=1:Nmisure
   for j=1:nbasis
       Xric(:,i)=Xric(:,i)+A(i,j)*phi(:,j);
   end
end
%%
% fft
figure()
for i=1:1
 Fou_a=abs(fft(Xric(:)))/length(Xric(:));
ff=(0:Fsamp/floor(length(Xric(:))):Fsamp/2);
 plot(ff(1:end),2*Fou_a(1:length(ff)))
 hold on
end
xlabel('f (Hz)')
ylabel('Amplitude')
% saveas(gcf,strcat('Fig_4_modes',num2str(modes),'.fig'),'fig')
% saveas(gcf,strcat('Fig_4_modes',num2str(modes),'.emf'),'emf')
      
%% Single mode choice
modes=[ 1 2 3 4 5 6 7 8 9 10 11 12]; % chose a pair of mode
Xric_single=zeros(Nperiod,Nmisure);
for i=1:Nmisure
   for ll=1: length(modes)
       Xric_single(:,i)=Xric_single(:,i)+A(i,modes(ll))*phi(:,modes(ll));
   end
end

figure()
for i=1:1
 Fou_a=abs(fft(Xric_single(:)))/length(Xric_single(:));
ff=(0:Fsamp/floor(length(Xric_single(:))):Fsamp/2);
 plot(ff(1:end),2*Fou_a(1:length(ff)))
 hold on
end
ylabel('Amplitude')
xlabel('f(Hz)')
% saveas(gcf,strcat('Fig_5_freq_modes',num2str(modes),'.fig'),'fig')
% saveas(gcf,strcat('Fig_5_freq_modes',num2str(modes),'.emf'),'emf')
      
figure()
subplot(2,1,1)
plot(0:1/Fsamp:1/Fsamp*(length(Signal)-1),Signal)
hold on
plot(0:1/Fsamp:1/Fsamp*(length(Xric_single(:))-1),Xric_single(:))
xlabel('Time')
ylabel('Amplitude')
subplot(2,1,2)
plot(0:1/Fsamp:1/Fsamp*(length(Signal)-1),Signal'-Xric_single(:))
xlabel('Time')
ylabel('Difference of Amplitude')
%     saveas(gcf,strcat('Fig_6_modes',num2str(modes),'.fig'),'fig')
%      saveas(gcf,strcat('Fig_6_modes',num2str(modes),'.emf'),'emf')

%% Network training
%train_fraction = 0.8/(num/100000);
numTimeStepsTrain = floor(0.7*numel(phi(:,1)));
XTest=0:1:(numel(phi(:,1))-numTimeStepsTrain-2);
YTest=0:1:(numel(phi(:,1))-numTimeStepsTrain-2);
XTest = transpose(XTest);
YTest = transpose(YTest);

% using 70% of signal to train network, will predict next 30% of signal
for i=1:Modi
    numTimeStepsTrain(1,i) = floor(0.7*numel(phi(:,i)));

    XTrain(1:numTimeStepsTrain,i) = phi(1:numTimeStepsTrain,i);
    YTrain(1:numTimeStepsTrain,i) = phi(2:numTimeStepsTrain+1,i);
    XTest(:,:) = phi(numTimeStepsTrain+1:end-1,i);
    YTest(:,:) = phi(numTimeStepsTrain+2:end,i);
    XTrain = transpose(XTrain);
    YTrain = transpose(YTrain);
    XTest = transpose(XTest);
    YTest = transpose(YTest);

    mu(i) = nanmean(XTrain(i));
    sig(i) = nanstd(XTrain(i));

    %XTrain(i) = (XTrain(i) - mu(i)) / sig(i);
    %YTrain(i) = (YTrain(i) - mu(i)) / sig(i);

    %XTest(i) = (XTest(i) - mu(i)) / sig(i);

    inputSize = 1;
    numResponses = 1;
    numHiddenUnits = 200;
end

t=0:.1:10;
Sig=sin(2*pi*t)+rand(size(t));
figure('units','normalized','outerposition',[0 0 1 1],'Color',[1,1,1])
plot(t,Sig)
axis tight;box on; grid on

numTimeStepsTrain= floor(0.7*numel(t));
XTrain(1:numTimeStepsTrain)=t(1:numTimeStepsTrain);
YTrain(1:numTimeStepsTrain) = Sig(1:numTimeStepsTrain);
inputSize = 1;
numResponses = 1;
numHiddenUnits = 10;


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
    'Verbose',1, ...
    'Plots','training-progress');
%% TRAIN LSTM NETWORK

net = trainNetwork(XTrain(1,:),YTrain(1,:),layers,opts);

%% UPDATING PREDICTION BASED ON PREVIOUS PREDICTION VALUE
% for i=1:Modi
i=1;
net = predictAndUpdateState(net,XTrain(i,:));
[net,YPred] = predictAndUpdateState(net,YTrain(i,numTimeStepsTrain));

numTimeStepsTest = numel(XTest);
%net = resetState(net);


for j = 2:numTimeStepsTest
%    [net12,YPred12(1,i)] = predictAndUpdateState(net,XTest(i));
    [net,YPred(i,j)] = predictAndUpdateState(net,YPred(j-1));
%     rmse = sqrt(mean((YPred(1,i)-YPred12(1,i)).^2));

    
%     if rmse >= 0.2
%         [net,YPred(1,i)] = predictAndUpdateState(net,XTest(i));
%     end
        
      % use updated value to fix prediction
      if mod(j,100) == 0
          [net,YPred(i,j)] = predictAndUpdateState(net,XTest(j));
      end
end
%YPred = sig*YPred + mu;
%rmse = sqrt(mean((YPred-YTest).^2))

figure
% subplot(3,1,1)
plot(phi(1:numTimeStepsTrain,1))
hold on
idx = numTimeStepsTrain+1:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,YPred(i,:),'.-')
hold off
xlabel("Month")
ylabel("Windspeed")
title("Windspeed Prediction")
legend(["Observed" "Forecast"])

% figure
% subplot(2,1,1)
% plot(YTest)
% hold on
% plot(YPred,'.-')
% hold off
% legend(["Observed" "Forecast"])
% ylabel("Windspeed")
% title("Forecast")
% 
% subplot(2,1,2)
% stem(YPred - YTest)
% xlabel("Month")
% ylabel("Error")
% title("RMSE")

% end
%% STOP %%