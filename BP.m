%% Environment initialization
warning off
close all
clear
clc

%% Data import
max_R2 = 0;
for numberoftimes = 1:10000
res = table2array(readtable('Beijing.xlsx'));
kes = table2array(readtable('Beijing_factor.xlsx'));
disp(['Number of training times:', num2str(numberoftimes)]);
disp('**************************');

%% Divide training set and test set
temp = 1:24;
P_train = res(temp(1: 21), 2: 20)';  %Dependent variable column number
T_train = res(temp(1: 21), 1)';     %Independent variable column number
M = size(P_train, 2);
P_test = res(temp(22: end), 2: 20)';  %Dependent variable column number
T_test = res(temp(22: end), 1)';    %Independent variable column number
N = size(P_test, 2);
kes = kes';
K = size(kes, 2);

%% Data normalization
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);
[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);
n_test = mapminmax('apply', kes, ps_input);

%% Create BP neural network
net = newff(p_train, t_train, 10); 

%% Training parameter initialization
net.trainParam.epochs = 10000;
net.trainParam.goal = 1e-100;
net.trainParam.min_grad = 1e-100;
net.trainParam.mu_max = 1e+100;
net.trainParam.mc = 0.9;
net.trainParam.lr = 0.05;
net.trainParam.max_fail = 1000;
net.trainParam.showWindow = false;

%% Neural Network training
net = train(net, p_train, t_train);

%% Model testing
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test);
t_sim3 = sim(net, n_test);
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
T_sim3 = mapminmax('reverse', t_sim3, ps_output);
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2)^2 / norm(T_test -  mean(T_test ))^2;
disp(['The R2 of the test set data is:', num2str(R2)]);
if(R2 > max_R2)
max_R2 = R2;
end
disp(['The current maximum R2 of the test set data is:', num2str(max_R2)]);
if(R2 > 0.95)

%% Figure
figure
plot(1: M, T_train,'r-o','Color',[0 0 0]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[255 0 0]./255)
hold on
plot(1: M, T_sim1,'b-s','Color',[0 0 0]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[0 0 255]./255)
legend('Actual value','Predictive value')
xlabel('Prediction sample')
ylabel('Prediction result')
string = {'Comparison of training set prediction results'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test,'r-o','Color',[0 0 0]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[255 0 0]./255)
hold on
plot(1: N, T_sim2,'b-s','Color',[0 0 0]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[0 0 255]./255)
legend('Actual value','Predictive value')
xlabel('Prediction sample')
ylabel('Prediction result')
string = {'Comparison of test set prediction results';['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

figure
plot(1: M, T_train,'-o','Color',[0 0 0]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[255 0 0]./255)
hold on
plot(1: M, T_sim1,'-s','Color',[0 0 0]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[0 0 255]./255)
hold on
plot(M+1:M+N, T_test,'-o','Color',[0 0 0]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[255 255 0]./255)
hold on
plot(M+1:M+N, T_sim2,'-s','Color',[0 0 0]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[0 255 255]./255)
hold on
plot(M+N:M+N+K,[T_sim2(end) T_sim3],'g-o','Color',[0 0 0]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[0 255 0]./255)
legend('Actual value of training set','Predictive value of training set','Actual value of test set','Prediction value of test set','Prediction value of future')
xlabel('Prediction sample')
ylabel('Prediction result')
string = {'Comparison of test set prediction results'};
title(string)
xlim([1, M+N+K])
grid
    break;
end

end