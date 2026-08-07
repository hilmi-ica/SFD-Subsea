% PFExp.m for LPPI
% by Muhammad Hilmi, ITK NTNU

clear all; close all; clc;

%% Load Data

data = readtable('CV1Data.xlsx','VariableNamingRule','preserve');
dT_all = data.('dT (Month)'); dy_all = data.delta;

valid = dT_all > 0;
dT_all = dT_all(valid); dy_all = dy_all(valid);
T = length(dy_all); t_cum = cumsum(dT_all);

%% AIC & BIC - MLE

k_par = 2;

% Wiener
par_w = mle(dy_all,Distribution="Normal"); 
mu_w = par_w(1); sig_w = par_w(2);
logL_w = sum(log(normpdf(dy_all,mu_w,sig_w)));

% Gamma (shifted distribution)
shiftgampdf = @(x,a,b,xi) max(realmin,gampdf(x-xi,a,b));
par_g = mle(dy_all,'pdf',shiftgampdf,'start',[0.1,0,0.2]);
alpha_g = par_g(1); beta_g = par_g(2);
logL_g = sum(log(shiftgampdf(dy_all,alpha_g,beta_g,par_g(3))));

% Inverse Gaussian (shifted distribution)
shiftigpdf = @(x,a,b,xi) max(realmin,pdf('InverseGaussian',x-xi,a,b));
par_ig = mle(dy_all,'pdf',shiftigpdf,'start',[0.1,0,0.2]);
mu_ig = par_ig(1); lam_ig = par_ig(2);
logL_ig = sum(log(shiftigpdf(dy_all, mu_ig, lam_ig,par_ig(3))));

% Compound Poisson (shifted distribution)
shiftcppdf = @(x,a,b,xi) max(realmin,cpoisson_pdf(x-xi,a,b));
par_cp = mle(dy_all,'pdf',shiftcppdf,'start',[0.1,0,0.2]);
lam_j = par_cp(1); mu_j = par_cp(2);
logL_cp = sum(log(shiftcppdf(dy_all, lam_j, mu_j,par_cp(3))));

% AIC and BIC
logLs = [logL_w; logL_g; logL_ig; logL_cp];
AIC = 2*k_par - 2*logLs;
BIC = k_par*log(T) - 2*logLs;
[~, AIC_model] = min(AIC);
[~, BIC_model] = min(BIC);

model_names = {'Wiener','Gamma','Inv. Gaussian','C. Poisson'};
mle_params = [mu_w, sig_w; alpha_g, beta_g; mu_ig, lam_ig; lam_j, mu_j];

%% IMM-SMC Filter Settings

N_MAX = 400; N_MIN = 20; N_MODELS = 4;
sigma_obs = 0.1; sigma_eff = sqrt(2) * sigma_obs;

LW_H_MAX = 0.25; LW_H_MIN = 0.05; LW_DECAY = 0.97;
KLD_EPSILON = 0.05; KLD_DELTA = 0.99; KLD_BINS = 12;
SWITCH_TAU = 0.55; SWITCH_K = 5;

% Prior bounds initizalization
PARAM_PRIOR = [ ...
   -0.05, 0.01, 0.01, 0.50; ...  
    5.00, 15.00, 0.01, 1.00; ...  
    0.50, 5.00,  5.00, 20.00; ... 
    5.00, 20.00, 0.01, 1.0];

rng(100);

%% Initialize Particles

N_init = floor(N_MAX / N_MODELS);
particles = cell(N_MODELS,1);
weights = cell(N_MODELS,1);
model_prob = ones(N_MODELS,1) / N_MODELS;

for i = 1:N_MODELS
    p1 = PARAM_PRIOR(i,1)+(PARAM_PRIOR(i,2)-PARAM_PRIOR(i,1))*rand(1,N_init);
    p2 = PARAM_PRIOR(i,3)+(PARAM_PRIOR(i,4)-PARAM_PRIOR(i,3))*rand(1,N_init);
    particles{i} = [zeros(1,N_init); p1; p2];
    weights{i} = ones(1,N_init)/N_init;
end

%% Storage

dx_est = zeros(T,1); 
p1_traj = zeros(T,N_MODELS); p2_traj = zeros(T,N_MODELS);
model_hist = zeros(T,N_MODELS); selected_model = zeros(T,1);
particle_count = zeros(T,N_MODELS); consec_count = zeros(N_MODELS,1);
committed_model = 0; lw_h = LW_H_MAX;

%% Main Filter Loop

for t = 1:T
    dt_t = dT_all(t);

    lw_h = max(LW_H_MIN, lw_h * LW_DECAY);
    lw_a = sqrt(1 - lw_h^2);

    % Liu-West parameter perturbation
    for i = 1:N_MODELS
        Ni  = size(particles{i},2); w = weights{i};
        th1 = particles{i}(2,:); th2 = particles{i}(3,:);

        mu1 = sum(w.*th1); V1 = max(sum(w.*(th1-mu1).^2), 1e-8);
        mu2 = sum(w.*th2); V2 = max(sum(w.*(th2-mu2).^2), 1e-8);

        th1_n = lw_a*th1 + (1-lw_a)*mu1 + lw_h*sqrt(V1)*randn(1,Ni);
        th2_n = lw_a*th2 + (1-lw_a)*mu2 + lw_h*sqrt(V2)*randn(1,Ni);
        th1_n = max(min(th1_n, PARAM_PRIOR(i,2)), PARAM_PRIOR(i,1));
        th2_n = max(th2_n, PARAM_PRIOR(i,3)*0.1);

        particles{i}(1,:) = sample_increment(i, th1_n, th2_n, dt_t, Ni);
        particles{i}(2,:) = th1_n;
        particles{i}(3,:) = th2_n;
    end

    % Weight update
    raw_evidence = zeros(N_MODELS,1);
    for i = 1:N_MODELS
        Ni = size(particles{i},2); dx_p = particles{i}(1,:);
        lhood = normpdf(dy_all(t), dx_p, sigma_eff);
        w_sum = sum(lhood);
        if w_sum < 1e-300
            weights{i} = ones(1,Ni)/Ni;
            raw_evidence(i) = 1e-300;
        else
            weights{i} = lhood / w_sum;
            raw_evidence(i) = w_sum / Ni;
        end
    end

    % Model probability update
    model_prob = raw_evidence .* model_prob;
    model_prob = model_prob / sum(model_prob);
    model_prob = max(model_prob, 1e-6);
    model_prob = model_prob / sum(model_prob);
    model_hist(t,:) = model_prob';

    % KLD adaptive resampling
    N_rem = N_MAX - N_MIN*N_MODELS;
    N_alloc = N_MIN + round(model_prob * N_rem);
    diff_n = sum(N_alloc) - N_MAX;
    if diff_n ~= 0
        [~,adj] = max(N_alloc - N_MIN);
        N_alloc(adj) = N_alloc(adj) - diff_n;
    end
    N_alloc = max(N_alloc, N_MIN);

    z_kld = norminv(KLD_DELTA);
    for i = 1:N_MODELS
        dx_s = particles{i}(1,:);
        x_lo = min(dx_s)-1e-6;  x_hi = max(dx_s)+1e-6;
        edges = linspace(x_lo, x_hi, KLD_BINS+1);
        k = max(2, sum(histcounts(dx_s, edges) > 0)); d = 9*(k-1);
        N_kld = ceil((k-1)/(2*KLD_EPSILON)*(1-2/d+z_kld*sqrt(2/d))^3);
        N_target = max(N_MIN, min(N_alloc(i), N_kld));
        idx = systematic_resample(weights{i}, N_target);
        particles{i} = particles{i}(:,idx);
        weights{i} = ones(1,N_target)/N_target;
    end

    particle_count(t,:) = cellfun(@(p) size(p,2), particles)';

    % Point estimates
    dx_model = zeros(N_MODELS,1);
    for i = 1:N_MODELS
        w = weights{i};
        dx_model(i) = sum(w .* particles{i}(1,:));
        p1_traj(t,i) = sum(w .* particles{i}(2,:));
        p2_traj(t,i) = sum(w .* particles{i}(3,:));
    end
    dx_est(t) = model_prob' * dx_model;

    % Two-tier robust model selection
    K_eff = min(SWITCH_K, t);
    [prob_max, best_i] = max(model_prob);
    if prob_max >= SWITCH_TAU
        committed_model = best_i;
        consec_count(:) = 0;
    else
        consec_count(best_i) = consec_count(best_i) + 1;
        consec_count(setdiff(1:N_MODELS,best_i)) = 0;
        if consec_count(best_i) >= K_eff
            committed_model = best_i;
        end
    end
    if committed_model == 0
        selected_model(t) = best_i;
    else
        selected_model(t) = committed_model;
    end
end

final_model = selected_model(T);

%% Visualization

cols = lines(N_MODELS);
tvec = t_cum;

p_norm_traj = sqrt(p1_traj(:,final_model).^2 + p2_traj(:,final_model).^2);
dx_err = dy_all - dx_est;

figure('Position',[60 60 700 500]);

% Increment error + parameter norm
ax1 = subplot(2,1,1); yyaxis left; 
ax1.YAxis(1).Color = 'k'; ax1.YAxis(2).Color = 'k';
plot(tvec, dx_err, 'LineWidth', 5, 'Color', '#36454F');
ylabel('Increment Error', 'FontName', 'serif', 'FontSize', 14);
yyaxis right; plot(tvec, p_norm_traj, 'LineWidth', 5, 'Color', '#2E7D32');
ylabel('L2-Norm Parameter','FontName','serif','FontSize',14);
legend('\delta-\Deltax', '||\theta||', 'FontName', 'serif', ...
    'FontSize', 14, 'Location', 'southeast', 'Orientation', 'horizontal');
set(gca, 'FontName', 'serif', 'FontSize', 14);
grid on; xlim tight;

% Model probability
ax2 = subplot(2,1,2);
dtThreshold = 0.9;
dyThreshold = 0.1;

ax2 = subplot(2,1,2);
for i = 1:N_MODELS
    [tPlot,pPlot] = adaptiveDensify(...
        tvec, model_hist(:,i), dtThreshold, dyThreshold);
    plot(tPlot, pPlot, '.', 'Color', cols(i,:), 'MarkerSize', 20,...
        'DisplayName',model_names{i}); hold on;
end
yline(SWITCH_TAU,'k--','LineWidth',5,'DisplayName','\tau');
ylabel('Model Probability', 'FontName', 'serif', 'FontSize', 14);
xlabel('Time (Months)', 'FontName', 'serif', 'FontSize', 14);
legend('FontName', 'serif', 'FontSize', 14, 'Location', ...
    'southeast', 'Orientation', 'horizontal');
set(gca, 'FontName', 'serif', 'FontSize', 14);
grid on; xlim tight; hold off;

% Console summary
fprintf('AIC best model: %s\n', model_names{AIC_model});
fprintf('BIC best model: %s\n', model_names{BIC_model});
fprintf('\n=== MLE parameter estimates ===\n');
for i = 1:N_MODELS
    fprintf('  %-14s  p1 =%8.4f  p2 =%8.4f  AIC =%7.2f  BIC =%7.2f\n', ...
            model_names{i}, mle_params(i,1), mle_params(i,2), AIC(i), BIC(i));
end

fprintf('\nIMM-SMC final model: %s\n', model_names{final_model});
fprintf('\n=== IMM-SMC final parameter estimates ===\n');
for i = 1:N_MODELS
    fprintf('  %-14s  p1 =%8.4f  p2 =%8.4f\n', ...
            model_names{i}, p1_traj(T,i), p2_traj(T,i));
end

%% Local Functions

function p = igpdf(x, mu, lambda)
    x=max(x,1e-12); mu=max(mu,1e-12); lambda=max(lambda,1e-12);
    p = sqrt(lambda./(2*pi*x.^3)) .* exp(-lambda.*(x-mu).^2./(2*mu.^2.*x));
    p = max(p,0);
end

function p = cpoisson_pdf(x, lam, mu_j)

    % Initialize output array
    p = zeros(size(x));
    
    % Handle the zero-probability case
    zero_idx = (x == 0);
    p(zero_idx) = poisspdf(0, lam); 
    
    % Calculate continuous density
    pos_idx = (x > 0);
    if ~any(pos_idx)
        return;
    end
    
    x_pos = x(pos_idx);
    
    % Truncation based on lambda
    max_lam = max(lam(:));
    K_max = max(50, ceil(max_lam * 3 + 20));
    
    % Sum over possible number of jumps
    for kk = 1:K_max
        pk = poisspdf(kk, lam);
        g_pdf = gampdf(x_pos, kk, mu_j);
        
        % Accumulate conditional density
        if isscalar(lam)
            p(pos_idx) = p(pos_idx) + pk * g_pdf;
        else
            lam_pos = lam(pos_idx);
            pk_pos = poisspdf(kk, lam_pos);
            p(pos_idx) = p(pos_idx) + pk_pos .* g_pdf;
        end
        
        % Early exit if negligible additional weight
        if max(pk) < 1e-15
            break; 
        end
    end
end

function dx = sample_increment(model_id, th1, th2, dt, N)
    switch model_id
        case 1  % Wiener
            dx = th1*dt + th2*sqrt(dt).*randn(1,N);
        case 2  % Gamma
            dx = zeros(1,N);
            for j=1:N
                dx(j)=gamrnd(max(th1(j)*dt,1e-6),1/max(th2(j),1e-6));
            end
        case 3  % Inverse Gaussian
            dx = zeros(1,N);
            for j=1:N
                dx(j)=random('InverseGaussian',max(th1(j)*dt,1e-9), ...
                    max(th2(j)*dt^2,1e-9));
            end
        case 4  % Compound Poisson
            dx = zeros(1,N);
            for j=1:N
                n=poissrnd(max(th1(j)*dt,1e-9));
                if n>0, dx(j)=sum(exprnd(max(th2(j),1e-9),1,n)); end
            end
    end
end

function idx = systematic_resample(weights, N_target)
    w=weights/sum(weights); cdf=cumsum(w); N=length(w);
    idx=zeros(1,N_target); u0=rand/N_target;
    u=u0+(0:N_target-1)/N_target; j=1;
    for i=1:N_target
        while j<N && cdf(j)<u(i), j=j+1; end
        idx(i)=j;
    end
end

function [tDense, yDense] = adaptiveDensify(t, y, dtThresh, dyThresh)
tDense = t(1); yDense = y(1);
    for k = 2:length(t)
        dt = t(k) - t(k-1);
        dy = abs(y(k) - y(k-1));
    
        % Only densify if necessary
        if (dt > dtThresh) || (dy > dyThresh)
    
            % Number of inserted points
            n = ceil(max(dt/median(diff(t)),dy/0.05));
            ti = linspace(t(k-1), t(k), n+2);
            yi = interp1(t([k-1 k]), y([k-1 k]), ti, 'pchip');
            tDense = [tDense; ti(2:end)']; yDense = [yDense; yi(2:end)'];
        else
            tDense = [tDense; t(k)]; yDense = [yDense; y(k)];
        end
    end
end
