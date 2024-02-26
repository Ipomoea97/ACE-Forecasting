%% Environment initialization
warning off
close all
clear
clc

%% TPEBO algorithm parameter initialization

opt.Delays = 1:5;
opt.dataPreprocessMode  = 'Data Standardization';
opt.learningMethod      = 'LSTM';
opt.trPercentage        = 0.85;
opt.maxEpochs     = 500;
opt.miniBatchSize = 16;             %The GPU can perform better with a batch size that is a power of 2.
opt.executionEnvironment = 'gpu';   %'cpu' 'gpu' 'auto'
opt.LR                   = 'adam';  %'sgdm' 'rmsprop' 'adam'
opt.trainingProgress     = 'none';
opt.isUseBiLSTMLayer  = true;
opt.isUseDropoutLayer = true;
opt.DropoutValue      = 0.5;
opt.isUseOptimizer           = true;
opt.MaxOptimizationTime      = 14*60*60;
opt.MaxItrationNumber        = 50;
opt.isDispOptimizationLog    = true;
opt.isSaveOptimizedValue     = false;
opt.isSaveBestOptimizedValue = true;
opt.optimVars = [
    optimizableVariable('NumOfLayer',[1 5],'Type','integer')
    optimizableVariable('NumOfUnits',[15 500],'Type','integer')
    optimizableVariable('isUseBiLSTMLayer',[1 2],'Type','integer')
    optimizableVariable('InitialLearnRate',[1e-2 1],'Transform','log')
    optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')];

%% Data import
data.CompleteData = readtable('BayersOpt_LSTM_testdata.xlsx','Sheet',1);
data.seriesdataHeder = data.CompleteData.Properties.VariableNames(1,:);
data.seriesdata = table2array(data.CompleteData(:,:));
disp('Data import success!');
data.isDataRead = true;
figure('Name','InputData','NumberTitle','off');
plot(data.seriesdata,'--','Color',[0 0 180]./255,'linewidth',2,'Markersize',4,'MarkerFaceColor',[0 0 180]./255);
grid minor;
title({['Mean = ' num2str(mean(data.seriesdata)) ', STD = ' num2str(std(data.seriesdata)) ];});
if strcmpi(opt.dataPreprocessMode,'None')
    data.x = data.seriesdata;
elseif strcmpi(opt.dataPreprocessMode,'Data Normalization')
    
    for i=1:size(data.seriesdata,2)
        data.x(:,i) = (data.seriesdata(:,i) -min(data.seriesdata(:,i)))./ (max(data.seriesdata(:,i))-min(data.seriesdata(:,i)));
    end
    figure('Name','NormilizedInputData','NumberTitle','off');
    plot(data.x,'--','Color',[255 0 0]./255,'linewidth',2,'Markersize',4,'MarkerFaceColor',[255 0 0]./255);
    grid minor;
    title({['Mean = ' num2str(mean(data.x)) ', STD = ' num2str(std(data.x)) ];});
elseif strcmpi(opt.dataPreprocessMode,'Data Standardization')
    for i=1:size(data.seriesdata,2)
        x.mu(1,i)   = mean(data.seriesdata(:,i),'omitnan');
        x.sig(1,i)  = std (data.seriesdata(:,i),'omitnan');
        data.x(:,i) = (data.seriesdata(:,i) - x.mu(1,i))./ x.sig(1,i);
    end
    
    figure('Name','NormilizedInputData','NumberTitle','off');
    plot(data.x,'--','Color',[255 0 0]./255,'linewidth',2,'Markersize',4,'MarkerFaceColor',[255 0 0]./255); grid minor;
    title({['Mean = ' num2str(mean(data.x)) ', STD = ' num2str(std(data.x)) ];});
end
%% Data preparation
Delays = opt.Delays;
x = data.x';
T = size(x,2);
MaxDelay = max(Delays);
Range = MaxDelay+1:T;
X= [];
for d = Delays
    X=[X; x(:,Range-d)];
end
Y = x(:,Range);
data.X  = X;
data.Y  = Y;

% Data division
data.XTr   = [];
data.YTr   = [];
data.XTs   = [];
data.YTs   = [];

numTrSample = round(opt.trPercentage*size(data.X,2));
data.XTr   = data.X(:,1:numTrSample);
data.YTr   = data.Y(:,1:numTrSample);
data.XTs   = data.X(:,numTrSample+1:end);
data.YTs   = data.Y(:,numTrSample+1:end);
disp(['Time Series data divided to ' num2str(opt.trPercentage*100) '% Train data and ' num2str((1-opt.trPercentage)*100) '% Test data']);

% Data normalization
for i=1:size(data.XTr,2)
    XTr{i,1} = data.XTr(:,i);
    YTr(i,1) = data.YTr(:,i);
end

for i=1:size(data.XTs,2)
    XTs{i,1} =  data.XTs(:,i);
    YTs(i,1) =  data.YTs(:,i);
end
data.XTr   = XTr;
data.YTr   = YTr;
data.XTs   = XTs;
data.YTs   = YTs;
data.XVl   = XTs;
data.YVl   = YTs;

%% Find the best LSTM parameters based on TPEBO algorithm
if opt.isDispOptimizationLog
    isLog = 2;
else
    isLog = 0;
end
if opt.isUseOptimizer
    opt.ObjFcn  = ObjFcn(opt,data);
    BayesObject = bayesopt(opt.ObjFcn,opt.optimVars, ...
        'MaxTime',opt.MaxOptimizationTime, ...
        'IsObjectiveDeterministic',false, ...
        'MaxObjectiveEvaluations',opt.MaxItrationNumber,...
        'Verbose',isLog,...
        'UseParallel',false);
end

%% Data evaluation
[opt,data] = EvaluationData(opt,data);

%% Local sub-function
function ObjFcnn = ObjFcn(opt,data)
ObjFcnn = @CostFunction;

    function [valError,cons,fileName] = CostFunction(optVars)
        inputSize    = size(data.X,1);
        numResponses = 1;
        dropoutVal   = 0.5;
        
        if optVars.isUseBiLSTMLayer == 2
            optVars.isUseBiLSTMLayer = 0;
        end
        % if dropout layer is true
        if opt.isUseDropoutLayer
            if optVars.NumOfLayer ==1
                if optVars.isUseBiLSTMLayer
                    opt.layers = [ ...
                        sequenceInputLayer(inputSize)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','last')
                        dropoutLayer(dropoutVal)
                        fullyConnectedLayer(numResponses)
                        regressionLayer];
                else
                    opt.layers = [ ...
                        sequenceInputLayer(inputSize)
                        lstmLayer(optVars.NumOfUnits,'OutputMode','last')
                        dropoutLayer(dropoutVal)
                        fullyConnectedLayer(numResponses)
                        regressionLayer];
                end
            elseif optVars.NumOfLayer==2
                if optVars.isUseBiLSTMLayer
                    opt.layers = [ ...
                        sequenceInputLayer(inputSize)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        dropoutLayer(dropoutVal)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','last')
                        dropoutLayer(dropoutVal)
                        fullyConnectedLayer(numResponses)
                        regressionLayer];
                else
                    opt.layers = [ ...
                        sequenceInputLayer(inputSize)
                        lstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        dropoutLayer(dropoutVal)
                        lstmLayer(optVars.NumOfUnits,'OutputMode','last')
                        dropoutLayer(dropoutVal)
                        fullyConnectedLayer(numResponses)
                        regressionLayer];
                end
            elseif optVars.NumOfLayer ==3
                if optVars.isUseBiLSTMLayer
                    opt.layers = [ ...
                        sequenceInputLayer(inputSize)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        dropoutLayer(dropoutVal)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        dropoutLayer(dropoutVal)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','last')
                        dropoutLayer(dropoutVal)
                        fullyConnectedLayer(numResponses)
                        regressionLayer];
                else
                    opt.layers = [ ...
                        sequenceInputLayer(inputSize)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        dropoutLayer(dropoutVal)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        dropoutLayer(dropoutVal)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','last')
                        dropoutLayer(dropoutVal)
                        fullyConnectedLayer(numResponses)
                        regressionLayer];
                end
            elseif optVars.NumOfLayer==4
                if optVars.isUseBiLSTMLayer
                    opt.layers = [ ...
                        sequenceInputLayer(inputSize)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        dropoutLayer(dropoutVal)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        dropoutLayer(dropoutVal)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        dropoutLayer(dropoutVal)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','last')
                        dropoutLayer(dropoutVal)
                        fullyConnectedLayer(numResponses)
                        regressionLayer];
                else
                    opt.layers = [ ...
                        sequenceInputLayer(inputSize)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        dropoutLayer(dropoutVal)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        dropoutLayer(dropoutVal)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        dropoutLayer(dropoutVal)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','last')
                        dropoutLayer(dropoutVal)
                        fullyConnectedLayer(numResponses)
                        regressionLayer];
                end
            end
        else % if dropout layer is false
            if optVars.NumOfLayer ==1
                if optVars.isUseBiLSTMLayer
                    opt.layers = [ ...
                        sequenceInputLayer(inputSize)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','last')
                        fullyConnectedLayer(numResponses)
                        regressionLayer];
                else
                    opt.layers = [ ...
                        sequenceInputLayer(inputSize)
                        lstmLayer(optVars.NumOfUnits,'OutputMode','last')
                        fullyConnectedLayer(numResponses)
                        regressionLayer];
                end
            elseif optVars.NumOfLayer ==2
                if optVars.isUseBiLSTMLayer
                    opt.layers = [ ...
                        sequenceInputLayer(inputSize)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','last')
                        fullyConnectedLayer(numResponses)
                        regressionLayer];
                else
                    opt.layers = [ ...
                        sequenceInputLayer(inputSize)
                        lstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        lstmLayer(optVars.NumOfUnits,'OutputMode','last')
                        fullyConnectedLayer(numResponses)
                        regressionLayer];
                end
            elseif optVars.NumOfLayer ==3
                if optVars.isUseBiLSTMLayer
                    opt.layers = [ ...
                        sequenceInputLayer(inputSize)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','last')
                        fullyConnectedLayer(numResponses)
                        regressionLayer];
                else
                    opt.layers = [ ...
                        sequenceInputLayer(inputSize)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','last')
                        fullyConnectedLayer(numResponses)
                        regressionLayer];
                end
            elseif optVars.NumOfLayer ==4
                if optVars.isUseBiLSTMLayer
                    opt.layers = [ ...
                        sequenceInputLayer(inputSize)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','last')
                        fullyConnectedLayer(numResponses)
                        regressionLayer];
                else
                    opt.layers = [ ...
                        sequenceInputLayer(inputSize)
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                        bilstmLayer(optVars.NumOfUnits,'OutputMode','last')
                        fullyConnectedLayer(numResponses)
                        regressionLayer];
                end
            end
        end
        miniBatchSize    = opt.miniBatchSize;
        maxEpochs        = opt.maxEpochs;
        trainingProgress = opt.trainingProgress;
        executionEnvironment = opt.executionEnvironment;
        validationFrequency  = floor(numel(data.XTr)/miniBatchSize);
        opt.opts = trainingOptions(opt.LR, ...
            'MaxEpochs',maxEpochs, ...
            'GradientThreshold',1, ...
            'InitialLearnRate',optVars.InitialLearnRate, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',125, ...
            'LearnRateDropFactor',0.2, ...
            'L2Regularization',optVars.L2Regularization, ...
            'Verbose',0, ...
            'MiniBatchSize',miniBatchSize,...
            'ExecutionEnvironment',executionEnvironment,...
            'ValidationData',{data.XVl,data.YVl}, ...
            'ValidationFrequency',validationFrequency,....
            'Plots',trainingProgress);
        disp('LSTM architect successfully created.');
        
        % Model training
        try
            data.BiLSTM.Net = trainNetwork(data.XTr,data.YTr,opt.layers,opt.opts);
            disp('Model training success!');
            data.IsNetTrainSuccess =true;
        catch me
            disp('Model training fail!');
            data.IsNetTrainSuccess = false;
            return;
        end
        
        predict(data.BiLSTM.Net,data.XVl,'MiniBatchSize',opt.miniBatchSize);
        valError = mse(predict(data.BiLSTM.Net,data.XVl,'MiniBatchSize',opt.miniBatchSize)-data.YVl);
        
        Net  = data.BiLSTM.Net;
        Opts = opt.opts;
        
        fieldName = ['ValidationError' strrep(num2str(valError),'.','_')];
        if ismember('OptimizedParams',evalin('base','who'))
            OptimizedParams =  evalin('base', 'OptimizedParams');
            OptimizedParams.(fieldName).Net  = Net;
            OptimizedParams.(fieldName).Opts = Opts;
            assignin('base','OptimizedParams',OptimizedParams);
        else
            OptimizedParams.(fieldName).Net  = Net;
            OptimizedParams.(fieldName).Opts = Opts;
            assignin('base','OptimizedParams',OptimizedParams);
        end
        
        fileName = num2str(valError) + ".mat";
        if opt.isSaveOptimizedValue
            save(fileName,'Net','valError','Opts')
        end
        cons = [];
        
    end

end



% Data evaluation sub-function
function [opt,data] = EvaluationData(opt,data)
if opt.isUseOptimizer
    OptimizedParams =  evalin('base', 'OptimizedParams');
    [valBest,indxBest] = sort(str2double(extractAfter(strrep(fieldnames(OptimizedParams),'_','.'),'Error')));
    data.BiLSTM.Net = OptimizedParams.(['ValidationError' strrep(num2str(valBest(1)),'.','_')]).Net;
elseif ~opt.isUseOptimizer
    [chosenfile,chosendirectory] = uigetfile({'*.mat'},...
        'Select Net File','BestNet.mat');
    if chosenfile==0
        error('Choose to save or set optimization: true');
    end
    filePath = [chosendirectory chosenfile];
    Net = load(filePath);
    data.BiLSTM.Net = Net.Net;
end

data.BiLSTM.TrainOutputs = deNorm(data.seriesdata,predict(data.BiLSTM.Net,data.XTr,'MiniBatchSize',opt.miniBatchSize),opt.dataPreprocessMode);
data.BiLSTM.TrainTargets = deNorm(data.seriesdata,data.YTr,opt.dataPreprocessMode);
data.BiLSTM.TestOutputs  = deNorm(data.seriesdata,predict(data.BiLSTM.Net,data.XTs,'MiniBatchSize',opt.miniBatchSize),opt.dataPreprocessMode);
data.BiLSTM.TestTargets  = deNorm(data.seriesdata,data.YTs,opt.dataPreprocessMode);
data.BiLSTM.AllDataTargets = [data.BiLSTM.TrainTargets data.BiLSTM.TestTargets];
data.BiLSTM.AllDataOutputs = [data.BiLSTM.TrainOutputs data.BiLSTM.TestOutputs];

data = PlotResults(data,'Tr',...
    data.BiLSTM.TrainOutputs, ...
    data.BiLSTM.TrainTargets);
data = plotReg(data,'Tr',data.BiLSTM.TrainTargets,data.BiLSTM.TrainOutputs);

data = PlotResults(data,'Ts',....
    data.BiLSTM.TestOutputs, ...
    data.BiLSTM.TestTargets);
data = plotReg(data,'Ts',data.BiLSTM.TestTargets,data.BiLSTM.TestOutputs);

data = PlotResults(data,'All',...
    data.BiLSTM.AllDataOutputs, ...
    data.BiLSTM.AllDataTargets);
data = plotReg(data,'All',data.BiLSTM.AllDataTargets,data.BiLSTM.AllDataOutputs);

disp('Performance evaluation');

end
function vars = deNorm(data,stdData,deNormMode)
if iscell(stdData(1,1))
    for i=1:size(stdData,1)
        tmp(i,:) = stdData{i,1}';
    end
    stdData = tmp;
end
if strcmpi(deNormMode,'Data Normalization')
    for i=1:size(data,2)
        vars(:,i) = (stdData(:,i).*(max(data(:,i))-min(data(:,i)))) + min(data(:,i));
    end
    vars = vars';
    
elseif strcmpi(deNormMode,'Data Standardization')
    for i=1:size(data,2)
        x.mu(1,i)   = mean(data(:,i),'omitnan');
        x.sig(1,i)  = std (data(:,i),'omitnan');
        vars(:,i) = ((stdData(:,i).* x.sig(1,i))+ x.mu(1,i));
    end
    vars = vars';
    
else
    vars = stdData';
    return;
end
end

% Data visualization sub-function
function data = PlotResults(data,firstTitle,Outputs,Targets)
Errors = Targets - Outputs;
MSE   = mean(Errors.^2);
RMSE  = sqrt(MSE);
NRMSE = RMSE/mean(Targets);
ErrorMean = mean(Errors);
ErrorStd  = std(Errors);
rankCorre = RankCorre(Targets,Outputs);

if strcmpi(firstTitle,'tr')
    Disp1Name = 'OutputGraphEvaluation_TrainData';
    Disp2Name = 'ErrorEvaluation_TrainData';
    Disp3Name = 'ErrorHistogram_TrainData';
elseif strcmpi(firstTitle,'ts')
    Disp1Name = 'OutputGraphEvaluation_TestData';
    Disp2Name = 'ErrorEvaluation_TestData';
    Disp3Name = 'ErrorHistogram_TestData';
elseif strcmpi(firstTitle,'all')
    Disp1Name = 'OutputGraphEvaluation_ALLData';
    Disp2Name = 'ErrorEvaluation_ALLData';
    Disp3Name = 'ErrorHistogram_AllData';
end

figure('Name',Disp1Name,'NumberTitle','off');
plot(1:length(Targets),Targets,'b--','Color',[0 0 255]./255,'linewidth',2,'Markersize',4,'MarkerFaceColor',[0 0 255]./255);
hold on
plot(1:length(Outputs),Outputs,'k-','Color',[0 0 0]./255,'linewidth',2,'Markersize',5,'MarkerFaceColor',[0 0 0]./255);
grid minor
legend('Targets','Outputs','Location','best') ;
title(['Rank Correlation = ' num2str(rankCorre)]);

figure('Name',Disp2Name,'NumberTitle','off');
plot(Errors,'--','Color',[180 60 0]./255,'linewidth',2,'Markersize',4,'MarkerFaceColor',[180 60 0]./255);grid minor
title({['MSE = ' num2str(MSE) ', RMSE = ' num2str(RMSE)...
    ' NRMSE = ' num2str(NRMSE)] ;});
xlabel('Error Per Sample');

figure('Name',Disp3Name,'NumberTitle','off');
histogram(Errors);grid minor

title(['Error Mean = ' num2str(ErrorMean) ', Error StD = ' num2str(ErrorStd)]);
xlabel('Error Histogram');

if strcmpi(firstTitle,'tr')
    data.Err.MSETr = MSE;
    data.Err.STDTr = ErrorStd;
    data.Err.NRMSETr     = NRMSE;
    data.Err.rankCorreTr = rankCorre;
elseif strcmpi(firstTitle,'ts')
    data.Err.MSETs = MSE;
    data.Err.STDTs = ErrorStd;
    data.Err.NRMSETs     = NRMSE;
    data.Err.rankCorreTs = rankCorre;
elseif strcmpi(firstTitle,'all')
    data.Err.MSEAll = MSE;
    data.Err.STDAll = ErrorStd;
    data.Err.NRMSEAll     = NRMSE;
    data.Err.rankCorreAll = rankCorre;
end
end

% Correlation between predictive and actual values
function [r]=RankCorre(x,y)
x=x';
y=y';

% Data length
N = length(x);

% Data sequence
R = crank(x)';
for i=1:size(y,2)
    % Data sequence
    S = crank(y(:,i))';
    % Correlation calculation
    r(i) = 1-6*sum((R-S).^2)/N/(N^2-1);
end
end
function r=crank(x)
u = unique(x);
[~,z1] = sort(x);
[~,z2] = sort(z1);
r = (1:length(x))';
r=r(z2);
for i=1:length(u)
    s=find(u(i)==x);
    r(s,1) = mean(r(s));
end
end
% plot the regression line of output and real value
function data = plotReg(data,Title,Targets,Outputs)

if strcmpi(Title,'tr')
    DispName = 'RegressionGraphEvaluation_TrainData';
elseif strcmpi(Title,'ts')
    DispName = 'RegressionGraphEvaluation_TestData';
elseif strcmpi(Title,'all')
    DispName = 'RegressionGraphEvaluation_ALLData';
end
figure('Name',DispName,'NumberTitle','off');
x = Targets';
y = Outputs';
format long
b1 = x\y;
yCalc1 = b1*x;
scatter(x,y,'MarkerEdgeColor',[0 0.4470 0.7410],'LineWidth',1);
hold('on');
plot(x,yCalc1,'--','Color',[180 60 0]./255,'linewidth',2,'Markersize',4,'MarkerFaceColor',[180 60 0]./255);
xlabel('Prediction');
ylabel('Target');
grid minor
X = [ones(length(x),1) x];
b = X\y;
yCalc2 = X*b;
plot(x,yCalc2,'b-o','Color',[255 0 255]./255,'linewidth',1,'Markersize',4,'MarkerFaceColor',[255 0 255]./255)
legend('Data','Fit','Y=T','Location','best');
Rsq2 = 1 -  sum((y - yCalc1).^2)/sum((y - mean(y)).^2);

if strcmpi(Title,'tr')
    data.Err.RSqur_Tr = Rsq2;
    title(['Train Data, R^2 = ' num2str(Rsq2)]);
elseif strcmpi(Title,'ts')
    data.Err.RSqur_Ts = Rsq2;
    title(['Test Data, R^2 = ' num2str(Rsq2)]);
elseif strcmpi(Title,'all')
    data.Err.RSqur_All = Rsq2;
    title(['All Data, R^2 = ' num2str(Rsq2)]);
end

end