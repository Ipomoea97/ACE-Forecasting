%% Environment initialization
warning off
close all
clear
clc
rng('default')
%% Data import

data = readtable('Tuotest.xlsx','VariableNamingRule','preserve');
data = table2array(data);
input =data(:,1:6)';
input = mapminmax(input,0,1);
output=data(:,7)';
output = mapminmax(output,0,1);
nwhole=size(input,2);
train_ratio=0.8;
ntrain=round(nwhole*train_ratio);
ntest =nwhole-ntrain;

input_train =input(:,1:ntrain);
output_train=output(:,1:ntrain);

input_test =input(:, ntrain+1:ntrain+ntest);
output_test=output(:,ntrain+1:ntrain+ntest);

%% MIV algorithm 
p=input;
t=output;
p=p';
[m,n]=size(p);
yy_temp=p;

% p_increase is a matrix that increases by 10% p_decrease is a matrix that decreases by 10%

for i=1:n
    p=yy_temp;
    pX=p(:,i);
    pa=pX*1.1;
    p(:,i)=pa;
    aa=['p_increase'  int2str(i) '=p;'];
    eval(aa);
end

for i=1:n
    p=yy_temp;
    pX=p(:,i);
    pa=pX*0.9;
    p(:,i)=pa;
    aa=['p_decrease' int2str(i) '=p;'];
    eval(aa);
end
%% Feature importance neural network
nntwarn off;
p=yy_temp;
p=p';
net=newff(p,t,12,{'tansig','purelin'},'trainlm');
net=init(net);
net.trainParam.goal = 1e-300;
net.trainParam.show=500;
net.trainParam.min_grad = 1e-300;
net.trainParam.mu_max = 1e+300;
net.trainParam.lr=0.05;
net.trainParam.mc=0.9;
net.trainParam.epochs=1000;
net.trainParam.max_fail = 1000;
net.divideFcn = 'dividerand';
net.performFcn = 'mse';
net=train(net,p,t);

%% Variable importance calculation

for i=1:n
    eval(['p_increase',num2str(i),'=transpose(p_increase',num2str(i),');'])
end
for i=1:n
    eval(['p_decrease',num2str(i),'=transpose(p_decrease',num2str(i),');'])
end
% result_in is the output after increasing by 10% result_de is the output after decreasing by 10%
for i=1:n
    eval(['result_in',num2str(i),'=sim(net,','p_increase',num2str(i),');'])
end
for i=1:n
    eval(['result_de',num2str(i),'=sim(net,','p_decrease',num2str(i),');'])
end
for i=1:n
    eval(['result_in',num2str(i),'=transpose(result_in',num2str(i),');'])
end
for i=1:n
    eval(['result_de',num2str(i),'=transpose(result_de',num2str(i),');'])
end
% MIV value
% MIV is considered one of the best indicators for evaluating variable correlation in neural networks
% Its sign represents the direction of correlation, and its absolute value represents the relative importance of the influence
for i=1:n
    IV= ['result_in',num2str(i), '-result_de',num2str(i)];
    eval(['MIV_',num2str(i) ,'=mean(',IV,')*(1e7)']) 
    eval(['MIVX=', 'MIV_',num2str(i),';']);
    MIV(i,:)=MIVX;
end
[MB,iranked] = sort(MIV,'descend');

%% Data visualization analysis
figure()
barh(MIV(iranked),'g');
xlabel('Variable Importance','FontSize',12,'Interpreter','latex');
ylabel('Variable Rank','FontSize',12,'Interpreter','latex');
hold on
barh(MIV(iranked(1:5)),'y');
hold on
barh(MIV(iranked(1:3)),'r');
grid on 
xt = get(gca,'XTick');    
xt_spacing=unique(diff(xt));
xt_spacing=xt_spacing(1);    
yt = get(gca,'YTick');    
for ii=1:length(MIV)
    text(...
        max([0 MIV(iranked(ii))+0.02*max(MIV)]),ii,...
        ['Col ' num2str(iranked(ii))],'Interpreter','latex','FontSize',12);
end
set(gca,'FontSize',12)
set(gca,'YTick',yt);
set(gca,'TickDir','out');
set(gca, 'ydir', 'reverse' )
set(gca,'LineWidth',2);
drawnow