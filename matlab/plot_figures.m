
%% Figure 2
clc
clear all
close all
store_def = '../save_mat/store_';
store_end = '_60_1_BRK-A_0.png.mat';
vals = {'0_0','0_1'};
for i=1:2
    figure(i)
    filename = [store_def vals{i} store_end];
    loaded = load(filename);
    y = loaded.y;
    pred = loaded.pred ; 
    pred_up = loaded.pred_up ;
    plot(y,'r','linewidth',3)
    hold on
    plot(pred,'k--','linewidth',3)
    plot(pred_up,'b-.','linewidth',3)
    set(gca, 'FontSize', 18,'FontWeight','normal','LineWidth',2)
    xlabel('Days','Interpreter',"latex",'FontSize',21)
    ylabel('Normalized Stock Price','Interpreter',"latex",'FontSize',21)
    legend({'Ground Truth (BRK-A)','Past: Ground Truth','Past: Updated Truth'},'Interpreter',"latex",'FontSize',18, 'location', 'best')
end

figure(3)
plot(100.*(pred - pred_up)./pred,'linewidth',3)
set(gca, 'FontSize', 18,'FontWeight','normal','LineWidth',2)
xlabel('Days','Interpreter',"latex",'FontSize',21)
ylabel('Relative Difference ($\%$)','Interpreter',"latex",'FontSize',21)
% legend({'Ground Truth (BRK-A)','Past: Ground Truth','Past: Updated Truth'},'Interpreter',"latex",'FontSize',18, 'location', 'best')

% bot_def = 'bot_0_1_60_1_MSFT_0.png.mat';

%% Figure 3
clc
clear all
close all
store_def = '../save_mat/EDstore_';
store_end = '_60_1_BRK-A_0.png.mat';
vals = {'0_1'};
for i=1:1
    figure(i)
    filename = [store_def vals{i} store_end];
    loaded = load(filename);
    y = loaded.y;
    pred = loaded.pred ; 
    pred_up = loaded.pred_up ;
    plot(y,'r','linewidth',3)
    hold on
    plot(pred,'k--','linewidth',3)
    plot(pred_up,'b-.','linewidth',3)
    set(gca, 'FontSize', 18,'FontWeight','normal','LineWidth',2)
    xlabel('Days','Interpreter',"latex",'FontSize',21)
    ylabel('Normalized Stock Price','Interpreter',"latex",'FontSize',21)
    legend({'Ground Truth (BRK-A)','Past: Ground Truth','Past: Updated Truth'},'Interpreter',"latex",'FontSize',18, 'location', 'best')
end

%% Figure 4
clc
clear all
close all
store_def = '../save_mat/MSstore_1_0';
store_end = '_60_1_XOM_0.png.mat';
vals = {'0_0','0_1'};
for i=1:1
    figure(i)
    filename = [store_def store_end];
    loaded = load(filename);
    y = loaded.y;
    pred = loaded.pred ; 
    pred_up = loaded.pred_up ;
    plot(y,'r','linewidth',3)
    hold on
    plot(pred,'k--','linewidth',3)
    plot(pred_up,'b-.','linewidth',3)
    set(gca, 'FontSize', 18,'FontWeight','normal','LineWidth',2)
    xlabel('Days','Interpreter',"latex",'FontSize',21)
    ylabel('Normalized Stock Price','Interpreter',"latex",'FontSize',21)
    legend({'Ground Truth (BRK-A)','Past: Ground Truth','Past: Updated Truth'},'Interpreter',"latex",'FontSize',18, 'location', 'best')
end
ccl = load('../save_mat/cross_corr.mat');
cc = ccl.cc;
rmse = ccl.rms;
names = {'SHEL','RIO','NEE','BRK-A','CCI'};
inds = {'energy','materials','utilities','financials','estate'};
figure(2)
imagesc(cc)
colorbar
set(gca, 'FontSize', 18,'FontWeight','normal',...
    'LineWidth',2,'xtick',[1:5],'xticklabel',names,...
    'ytick',[1:5],'yticklabel',names)

figure(3)
imagesc(rmse)
h = colorbar
t=get(h,'Limits');
T=linspace(t(1),t(2),5)
set(h,'Ticks',T)
TL=arrayfun(@(x) sprintf('%.3f',x),T,'un',0)
set(h,'TickLabels',TL)
set(gca, 'FontSize', 18,'FontWeight','normal',...
    'LineWidth',2,'xtick',[1:5],'xticklabel',inds,...
    'ytick',[1:5],'yticklabel',names)


%% Figure 5
clc
clear all
close all
store_def = '../save_mat/store_0_1_60_20_BRK-A_';
store_end = '.png.mat';
vals = {'0','1'};
for i=1:2
    figure(i)
    filename = [store_def vals{i} store_end];
    loaded = load(filename);
    y = loaded.y;
    pred = loaded.pred ; 
    pred_up = loaded.pred_up ;
    plot(y,'r','linewidth',3)
    hold on
    plot(pred,'k--','linewidth',3)
    plot(pred_up,'b-.','linewidth',3)
    set(gca, 'FontSize', 18,'FontWeight','normal','LineWidth',2)
    xlabel('Days','Interpreter',"latex",'FontSize',21)
    ylabel('Normalized Stock Price','Interpreter',"latex",'FontSize',21)
    legend({'Ground Truth (BRK-A)','Past: Ground Truth','Past: Updated Truth'},'Interpreter',"latex",'FontSize',18, 'location', 'best')
end
con = load('../save_mat/plot_conv.mat');
conv = con.conv;
fits = 1 + (1/200)*[0:length(conv)-1];
figure(3)
plot((conv-1)*12/20+1,'r--*','linewidth',3)
hold on
plot(fits,'b:','linewidth',3)
set(gca, 'FontSize', 18,'FontWeight','normal','LineWidth',2)
xlabel('Days from Target day','Interpreter',"latex",'FontSize',21)
ylabel('Relative Stock Prediction','Interpreter',"latex",'FontSize',21)
legend({'','$w=1+\frac{1}{200}t$'},'Interpreter',"latex",'FontSize',24, 'location', 'best')
legend boxoff

%% Figure 6
clc
clear all
close all
bot_def = '../save_mat/bot_0_0_60_1_';
bot_end = '_0.png.mat';
store_def = '../save_mat/store_0_0_60_1_';
store_end = '_0.png.mat';
vals = {'BRK-A','GOOG','MSFT'};
for i=1:3
    figure(i)
    filename = [store_def vals{i} store_end];
    filename_2 = [bot_def vals{i} bot_end];
    loaded = load(filename);
    boted = load(filename_2);
    y = loaded.y;
    ideal = boted.ideal ; 
    pred = boted.pred ; 
    pred_up = boted.pred_up ;
    plot(100*y/y(1),'r','linewidth',3)
    hold on
    plot(ideal,'g:','linewidth',3)
    plot(pred,'k--','linewidth',3)
    plot(pred_up,'b-.','linewidth',3)
    set(gca, 'FontSize', 18,'FontWeight','normal','LineWidth',2)
    xlabel('Days','Interpreter',"latex",'FontSize',21)
    ylabel('Percentage Growth','Interpreter',"latex",'FontSize',21)
    legend({['Stock Value (' vals{i} ')'],'Bot: Ground Truth','Bot: Past $\rightarrow$ Updated Truth','Bot: Past $\rightarrow$ Updated Truth'},'Interpreter',"latex",'FontSize',18, 'location', 'best')
end

%% Figure 7
clc
clear all
close all
store_def = '../save_mat/GTMSstore_';
store_end = '_60_1_Ensemble_0.png.mat';
vals = {'0_1'};
for i=1:1
    figure(i)
    filename = [store_def vals{i} store_end];
    loaded = load(filename);
    y = loaded.y;
    pred = loaded.pred ; 
    pred_up = loaded.pred_up ;
    plot(y(:,1),'r','linewidth',3)
    hold on
    plot(pred(:,1),'k--','linewidth',3)
    plot(pred_up(:,1),'b-.','linewidth',3)
    set(gca, 'FontSize', 18,'FontWeight','normal','LineWidth',2)
    xlabel('Days','Interpreter',"latex",'FontSize',21)
    ylabel('Normalized Stock Price','Interpreter',"latex",'FontSize',21)
    legend({'Ground Truth (Ensemble)','Past: Ground Truth','Past: Updated Truth'},'Interpreter',"latex",'FontSize',18, 'location', 'best')
end

