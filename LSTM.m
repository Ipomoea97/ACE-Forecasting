%% Environment initialization
warning off
close all
clear
clc

%% Data import
max_R2 = 0;
t = (1:1:29)';
data = readtable('Cor_Beijing.xlsx','VariableNamingRule','preserve');
a1 = table2array(data(:,6));
a2 = table2array(data(:,7));
a3 = table2array(data(:,8));
a4 = table2array(data(:,9));
a5 = table2array(data(:,10));
a6 = table2array(data(:,11));
a7 = table2array(data(:,12));
a8 = table2array(data(:,13));
a9 = table2array(data(:,14));
a10 = table2array(data(:,15));
a11 = table2array(data(:,16));
a12 = table2array(data(:,17));
a13 = table2array(data(:,18));
a14 = table2array(data(:,19));
a15 = table2array(data(:,20));
a16 = table2array(data(:,21));
a17 = table2array(data(:,22));
a18 = table2array(data(:,23));
a19 = table2array(data(:,24));
y = table2array(data(:,1));

for numberoftimes = 1:10000
    disp('**************************');
    disp(['Number of training times:', num2str(numberoftimes)]);

%% Divide training set and test set
data = [y,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19];
numTimeStepsTrain = floor(0.9*numel(y));
dataTrain = data(1:numTimeStepsTrain+1,:);
dataTest = data(numTimeStepsTrain+1:end,:);

%% Data normalization
mu = mean(dataTrain);
sig = std(dataTrain);
dataTrainStandardized = (dataTrain - ones(length(dataTrain(:,1)),1)*mu) ./ (ones(length(dataTrain(:,1)),1)*sig); 
XTrain = dataTrainStandardized(:,2:20)'; 
YTrain = dataTrainStandardized(:,1)'; 

%% Create LSTM neural network
layers = [
    sequenceInputLayer(19,"Name","input") 
    lstmLayer(20,"Name","lstm1")
    dropoutLayer(0.2,"Name","drop1")
    fullyConnectedLayer(1,"Name","fc")
    regressionLayer("Name","regressionoutput")];
options = trainingOptions('adam', ...
    'MaxEpochs',10000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',250, ...
    'MiniBatchSize',6, ...
    'LearnRateDropFactor',0.5, ...
    'Verbose',0, ...
    'Plots','none'); 
%% Neural Network training
net = trainNetwork(XTrain,YTrain,layers,options); 
dataTestStandardized = (dataTest - ones(length(dataTest(:,1)),1)*mu) ./ (ones(length(dataTest(:,1)),1)*sig);
XTest = dataTestStandardized(:,2:20)'; 
net = predictAndUpdateState(net,XTrain);
numTimeStepsTest = numel(XTest(1,:));
YPred = [];
for i = 1:numTimeStepsTest
    [net,YPred(i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','gpu');
end 
YPred = sig(1)*YPred + mu(1); 
R2 = 1 - norm(data(25:end,1) -  YPred)^2 / norm((data(25:end,1) -  mean((data(25:end,1)))^2));
disp(['The R2 of the test set data is:', num2str(R2)]);

if(R2 > max_R2)
max_R2 = R2;
end
disp(['The current maximum R2 of the test set data is:', num2str(max_R2)]);

%% Figure
idx = (numTimeStepsTrain+1):(numTimeStepsTrain+numTimeStepsTest);
figure
set(gcf,'position',[1,1,1500,1000])
set(gca,'position',[0.1,0.1,0.8,4])
subplot(2,1,1)
plot(data(1:end,1),'b-') 
hold on
plot(idx, YPred,'r-','LineWidth',1.5)
hold off
xlabel("t")
ylabel("y")
subplot(2,1,2)
plot(data(1:end,1),'b-') 
hold on
plot(idx, YPred,'r-','LineWidth',1.5)
hold off
xlabel("t")
ylabel("y")
xlim([25,29])
title("Forecast")
legend(["Observed" "Forecast"])

end