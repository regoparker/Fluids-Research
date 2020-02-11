%% Data
clear all 
close all



t=0:1:10000;
scadadata = sind(t).*cosd(t) + rand(1,length(t))/2;

%% Date Inputs
%format: 'dd-mmm-yyyy HH:MM:SS' e.g '01-Jan-2016 00:05:00'
starttime = '23-Mar-2017 00:05:00';
endtime = '30-Mar-2017 00:05:00';

%% Decision Tree for Individual Turbines (Regression)c

for k = 1:size(scadadata,3);
    
    % Predictor Data: WS and TI
    Turbine_Data_2016 = scadadata(1:5000);
    % Response Data: Power (kW)
    Turbine_Power_Actual_2016 = scadadata(1:5000);

    % Test Data:
    Turbine_Data_2017 = scadadata(5001:10000);
    Turbine_Power_Actual_2017 = scadadata(5001:10000);

    %Tree Bagging
    
    rng(1);
    b = TreeBagger(200,Turbine_Data_2016,Turbine_Power_Actual_2016,'Method','regression','OOBPred',...
        'On','OOBPredictorImportance','On','MinLeafSize',5);


    % OOB Feature Importance
    figure;
    bar(b.OOBPermutedVarDeltaError);
    xlabel('Feature Number');
    ylabel('Out of Bag Feature Importance');


    % Processing Predicted Power
    Turbine_Power_Predicted(:,1,k) = predict(b,Turbine_Data_2017);
    Turbine_Power_Predicted(isnan(Turbine_Power_Actual_2017),:,k) = NaN;
    
    % Finding Total Power Error
    Total_Turbine_Power_Actual = nansum(Turbine_Power_Actual_2017);
    Total_Turbine_Power_Predicted = nansum(Turbine_Power_Predicted(:,1,k));
    Total_Power_Error(k,1) = ((Total_Turbine_Power_Predicted - Total_Turbine_Power_Actual)...
        /Total_Turbine_Power_Predicted)*100;

    
   % Predicted and Actual Power Curve 
    fig1 = figure;
    subplot(1,2,1);
    plot(Turbine_Data_2017(:,1),Turbine_Power_Predicted(:,1,k),'.b');
    xlabel('WS');
    ylabel('Predicted Power')
    subplot(1,2,2);
    plot(Turbine_Data_2017(:,1),Turbine_Power_Actual_2017,'.r');
    xlabel('WS');
    ylabel('Actual Power');
    xlim([0 30]);
    if k <= 53;
        ylim([0 1600]);
    else
        ylim([0 1100]);
    end
    set(gcf,'color','white');
    savefig(strcat('C:\Users\lya140030\Desktop\CC_Analysis\DATA_RESULTS\MachineLearing\P_curves\PredvsAct_P_curve_T_',num2str(k),'.fig')); close;
    
    % Predicted vs Actual Corellation Plot
    fig2 = figure; 
    plot_lin_fit(Turbine_Power_Actual_2017,Turbine_Power_Predicted(:,1,k));
    xlabel('Actual Power');
    ylabel('Predicted Power');
    if k <= 53;
        xlim([0 1600]);
        ylim([0 1600]);
    else
        xlim([0 1100]);
        ylim([0 1100]);
    end
    set(gcf,'color','white');
    savefig(strcat('C:\Users\lya140030\Desktop\CC_Analysis\DATA_RESULTS\MachineLearing\Correlations\PredvsAct_T_',num2str(k),'.fig')); close;

    % Predicted and Actual vs Time
    fig3 = figure;
    plot(mettime_form(find(scadadata(:,1,k)==datenum(starttime)):find(scadadata(:,1,k)==datenum(endtime)),1),...
        Turbine_Power_Actual_2017(find(scadadata(:,1,k)==datenum(starttime))-...
        size(Turbine_Power_Predicted,1):find(scadadata(:,1,k)==datenum(endtime))-size(Turbine_Power_Predicted,1),1),'LineWidth',1.3);
    hold on;
    plot(mettime_form(find(scadadata(:,1,k)==datenum(starttime)):find(scadadata(:,1,k)==datenum(endtime)),1),...
        Turbine_Power_Predicted(find(scadadata(:,1,k)==datenum(starttime))-...
        size(Turbine_Power_Predicted,1):find(scadadata(:,1,k)==datenum(endtime))-size(Turbine_Power_Predicted,1),1,k),'LineWidth',1.3);
    
    xlabel('Time');
    ylabel('Turbine Power');
    legend('Actual','Predicted');
    set(gcf,'color','white');
    savefig(strcat('C:\Users\lya140030\Desktop\CC_Analysis\DATA_RESULTS\MachineLearing\Time_plots\PredActvsTime_T_',num2str(k),'.fig')); close;
    display(k);
end



save('C:\Users\lya140030\Desktop\CC_Analysis\DATA_RESULTS\Turbine_Power_Predicted.mat','Turbine_Power_Predicted','-v7.3');
save('C:\Users\lya140030\Desktop\CC_Analysis\DATA_RESULTS\Total_Power_Error.mat','Total_Power_Error','-v7.3');


% Farm Calculations based on Individual Result
Individual_Turbine_Power = nansum(Turbine_Power_Predicted(:,1,:));
TPP = nansum(Individual_Turbine_Power(:,:,:));

TAP = nansum(scadadata(50001:end,2,:));
TAP = nansum(TAP(:,:,:));
TFE = ((TPP - TAP)/TAP)*100;

%% Decision Tree for Farm
% 
% % Response: total farm power at each time step
% for i = 1:size(scadadata,1)
%     P_farm_tot(i,1) = nansum(scadadata(i,2,:));
% end
% P_farm_2016 = P_farm_tot(1:50000,1);
% P_farm_2017 = P_farm_tot(51001:end,1);
% % Predictor: WS and TI from met FS for 2016
% farm_2016 = metdata_s(1:50000,[2 4 3]);
% %Prediction: 2017
% farm_2017 = metdata_s(51001:end,[2 4 3]);
% 
% 
% 
% rng(1);
% 
% c = TreeBagger(100,farm_2016,P_farm_2016,'Method','regression','OOBPred',...
%     'On','OOBPredictorImportance','On','MinLeafSize',5);
% 
% 
% 
% figure;
% plot(oobError(c));
% xlabel('Number of Grown Trees'); 
% ylabel('Out of Bag Mean Squared Error');
% 
% figure;
% bar(c.OOBPermutedVarDeltaError);
% xlabel('Feature Number');
% ylabel('Out of Bag Feature Importance');
% 
% %Predicted Farm Power using Model
% 
% P_mean_farm = predict(c,farm_2017);
% 
% % Total Predicted and Observed Power and Error
% 
% 
% P_mean_farm(isnan(P_farm_2017)) = NaN;
% P_act_farm = nansum(P_farm_2017);
% P_ML_farm = nansum(P_mean_farm);
% P_err_farm = ((P_ML_farm - P_act_farm)/P_ML_farm)*100;
% 
% 
% % Correlate Actual and predicted Power
% 
% for i = 1:size(P_farm_2017,1)
%     if P_farm_2017(i,1) < 0 || P_mean_farm(i,1) < 0
%         P_farm_2017(i,1) = NaN;
%         P_mean_farm(i,1) = NaN;
%     end
% end
% figure;
% plot_lin_fit(P_farm_2017(P_farm_2017(:,1) >= 0,1),P_mean_farm(P_mean_farm(:,1) >= 0,1));
% xlabel('Actual Power');
% ylabel('Predicted Power');
% 
% 
% % Power vs Time comparing Observed vs Predicted
% 
% figure;
% plot(P_farm_2017(27196:28196,1));
% hold on;
% plot(P_mean_farm(27196:28196,1),'LineWidth',2);
% title('Summer')
% 
% figure;
% plot(P_farm_2017(1:1000,1));
% hold on;
% plot(P_mean_farm(1:1000,1),'LineWidth',2);
% title('Winter');