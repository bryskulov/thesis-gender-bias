%===============================Master Thesis===================================
% Master Thesis 2022
% Author: Bakbergen Ryskulov

clear all;
close all;
clc;

%% Part 1: Import of data and defining the variables

data = readtable('hist_time_series');

lfp = data.LFP;
career = data.career;
family = data.family;
maths = data.maths;
arts = data.arts;
science = data.science;
intelligence = data.intelligence;
apperance = data.appearance;
strength = data.strength;
weakness = data.weakness;
professions = data.professions;
time = 1948:1:2000;


%% Part 2: Plotting the series

figure(1)
subplot(2,2,1)
plot(time,career)
axis tight
xlabel('Time')
title('Career')
set(gca,'FontSize',8)

subplot(2,2,2)
plot(time,family)
axis tight
xlabel('Time')
title('Family')
set(gca,'FontSize',8)

subplot(2,2,3)
plot(time,maths)
axis tight
xlabel('Time')
title('Maths')
set(gca,'FontSize',8)

subplot(2,2,4)
plot(time,science)
axis tight
xlabel('Time')
title('Science')
set(gca,'FontSize',8)

print(gcf,'fig1.png','-dpng','-r500');


figure(2)
subplot(2,1,1)
plot(time,strength)
axis tight
xlabel('Time')
title('Strength')
set(gca,'FontSize',8)

subplot(2,1,2)
plot(time,weakness)
axis tight
xlabel('Time')
title('Weakness')
set(gca,'FontSize',8)

print(gcf,'fig2.png','-dpng','-r500');



%% Part 3: Data transformation and Estimation

% Take log of the variables
x = strength;
h = log(lfp);
w = weakness;

% Take first differences of y and h to make the processes stationary
dx = diff(x)*100;
dh = diff(h)*100;
dw = diff(w)*100


% Plot x and h to check for stationarity
figure(3)
subplot(2,1,1)
yyaxis left
plot(time(2:end),dx)
yyaxis right
plot(time(2:end),dh);

axis tight
xlabel('Time')
title('Strength vs LFP')
legend('Strength','LFP')
set(gca,'FontSize',8)

subplot(2,1,2)
yyaxis left
plot(time(2:end),dw)
yyaxis right
plot(time(2:end),dh);
axis tight
xlabel('Time')
title('Weakness vs LFP')
legend('Weakness','LFP')
set(gca,'FontSize',8)

print(gcf,'fig3.png','-dpng','-r500');


% Define parameters of the VAR model
data = [dh, dx]; % Dataset
p = 5; % number of lags
constant = 1; % include constant
n = 2; % number of variables

% Compute VAR(4) for dx and dh
% dep, ind: vectors of dependent and independent vectors of the VAR model
[dep, ind, beta_hat, companion] = VAR(data, p, constant);



%% Part 4

% It is useful to check whether the process in VAR is stable or not.
% Stability implies that the process is stationary, so our VAR model admits
% the Wold representation. In VAR case, we can check stability by computing
% the eigenvalues of the companion matrix and check whether they are
% smaller than 1 in absolute value.

eigA = eig(companion);
if abs(eigA) < 1
    disp('The process is stable')
else, disp('The process is not stable')
end

% Define parameters for bootstrap
horizon = 15;
up = 84;
lb = 16;
t = [1:horizon]';

% Define parameters for bootstrap
horizon = 20;
up = 84;
lb = 16;
t = [1:horizon]';

% Compute IRFs and confidence intervals with bootstrap
[irf, cirf, low_irf, upp_irf, low_irf_chol, upp_irf_chol]=bootstrap(data, n, p, constant, horizon, up, lb);

% Plot the IRFs
figure(4)
subplot(2,2,1)      
plot(t,squeeze(irf(1, 1, :)),t,squeeze(low_irf(1, 1, :)), '--', t,squeeze(upp_irf(1, 1, :)), '--'); 
axis tight
xlabel('horizons')
title(' shock 1 -> LFP')
set(gca,'FontSize',8)

subplot(2,2,2)       
plot(t,squeeze(irf(1, 2, :)), t,squeeze(low_irf(1, 2, :)), '--', t,squeeze(upp_irf(1, 2, :)), '--'); 
axis tight
xlabel('horizons')
title('shock 2 -> LFP')
set(gca,'FontSize',8)

subplot(2,2,3)      
plot(t,squeeze(irf(2, 1, :)), t,squeeze(low_irf(2, 1, :)), '--', t,squeeze(upp_irf(2, 1, :)), '--'); 
axis tight
xlabel('horizons')
title('shock 1 -> Language')
set(gca,'FontSize',8)

subplot(2,2,4)       
plot(t,squeeze(irf(2, 2, :)), t,squeeze(low_irf(2, 2, :)), '--', t,squeeze(upp_irf(2, 2, :)), '--'); 
axis tight
xlabel('horizons')
title('shock 2 -> Language')
set(gca,'FontSize',8)

print(gcf,'fig4.png','-dpng','-r500');


%% Part 5 identify a new shock using the restriction that the shock has no effect contemporaneously on BIAS growth


% We need to use the cholesky decomposition
% First, compute the estimated errors
errors = dep - ind * beta_hat;

% Compute the variance-covariance matrix of the sample errors
omega_hat = (errors'*errors)./size(errors,1)

% Cholesky decomposition
B0 = chol(omega_hat)'

% Finally, identify the new shock
u = [inv(B0)*errors']';

% Recall bootstrap.m
[irf, cirf, low_irf, upp_irf, low_irf_chol, upp_irf_chol]= bootstrap(data, n, p, constant, horizon, up, lb);


% Plot
figure(5)
subplot(2,2,1)      
plot(t,squeeze(cirf(1, 1, :)),t,squeeze(low_irf_chol(1, 1, :)), '--', t,squeeze(upp_irf_chol(1, 1, :)), '--'); 
xlabel('horizons')
title('Economic shock -> LFP')
set(gca,'FontSize',8)
subplot(2,2,2)       
plot(t,squeeze(cirf(1, 2, :)), t,squeeze(low_irf_chol(1, 2, :)), '--', t,squeeze(upp_irf_chol(1, 2, :)), '--'); 
xlabel('horizons')
title('shock2 -> LFP')
set(gca,'FontSize',8)
subplot(2,2,3)      
plot(t,squeeze(cirf(2, 1, :)), t,squeeze(low_irf_chol(2, 1, :)), '--', t,squeeze(upp_irf_chol(2, 1, :)), '--'); 
xlabel('horizons')
title('Economic shock -> Language')
set(gca,'FontSize',8)
subplot(2,2,4)       
plot(t,squeeze(cirf(2, 2, :)), t,squeeze(low_irf_chol(2, 2, :)), '--', t,squeeze(upp_irf_chol(2, 2, :)), '--'); 
xlabel('horizons')
title('shock2 -> Language')
set(gca,'FontSize',8)

print(gcf,'fig5.png','-dpng','-r500');

