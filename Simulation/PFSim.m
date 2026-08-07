% PFSim.m for LPPI
% by Muhammad Hilmi, ITK NTNU

clear all; close all;
clc; rng(50);

%% Initialization

% User settings 
T = 150; dt = 1.0;
x0 = 0.0; sigma_obs = 0.1;

GT_MODEL = 4; % 1=Wiener 2=Gamma 3=InvGauss 4=CPoisson
GT_PARAMS.wiener = [0.15, 0.20];
GT_PARAMS.gamma = [1.50, 8.00];
GT_PARAMS.invgauss = [0.15, 4.00];
GT_PARAMS.cpoisson = [0.30, 0.50];

switch GT_MODEL
    case 1, gt_p=GT_PARAMS.wiener;
    case 2, gt_p=GT_PARAMS.gamma;
    case 3, gt_p=GT_PARAMS.invgauss;
    case 4, gt_p=GT_PARAMS.cpoisson;
end

N_MAX = 400; N_MIN = 20;
N_MODELS = 4;

% Liu-West perturbation
LW_H_MAX = 0.25;
LW_H_MIN = 0.05;
LW_DECAY = 0.97;

% KLD resampling
KLD_EPSILON = 0.05;
KLD_DELTA = 0.99;
KLD_BINS = 12;

% Robust model switching
SWITCH_TAU = 0.55;
SWITCH_K = 5;

% Prior bounds
PARAM_PRIOR = [ ...
    0.01, 0.60,  0.01, 0.60; ...
    0.20, 4.00,  1.00, 12.00; ...
    0.01, 0.60,  0.20, 10.00; ...
    0.05, 1.50,  0.05, 2.00];
DPRIOR = [0.59, 0.59; 3.80, ...
    11.00; 0.59, 9.80; 1.45, 1.95];

% Ground-truth simulation
[dx_true, x_true, y_obs, dy_obs] = ...
    simulate_ground_truth(GT_MODEL, GT_PARAMS, T, dt, x0, sigma_obs);

% Effective likelihood propagation
sigma_eff = sqrt(2) * sigma_obs;

% Initialize particles
N_init = floor(N_MAX / N_MODELS);

particles = cell(N_MODELS,1); weights = cell(N_MODELS,1);
model_prob = ones(N_MODELS,1) / N_MODELS;

for i = 1:N_MODELS
    p1 = PARAM_PRIOR(i,1) + (PARAM_PRIOR(i,2) - ...
        PARAM_PRIOR(i,1)) * rand(1,N_init);
    p2 = PARAM_PRIOR(i,3) + (PARAM_PRIOR(i,4) - ...
        PARAM_PRIOR(i,3)) * rand(1,N_init);
    particles{i} = [zeros(1,N_init); p1; p2];
    weights{i} = ones(1,N_init)/N_init;
end

% Variable storage
dx_est = zeros(T,1); x_est = zeros(T,1); x_est_cum = x0;
p1_est = zeros(T,1); p2_est = zeros(T,1);
model_hist = zeros(T,N_MODELS); selected_model = zeros(T,1);
particle_count = zeros(T,N_MODELS); consec_count = zeros(N_MODELS,1);
lw_h_sched = zeros(T,1); committed_model = 0;

%% Main Loop

lw_h = LW_H_MAX;

for t = 1:T
    % Liu-West scheduling
    lw_h = max(LW_H_MIN, lw_h * LW_DECAY);
    lw_a = sqrt(1 - lw_h^2); lw_h_sched(t) = lw_h;

    % Parameter perturbation (Liu-West, 2001)
    for i = 1:N_MODELS
        Ni = size(particles{i},2); w = weights{i};
        th1 = particles{i}(2,:); th2 = particles{i}(3,:);

        mu1 = sum(w.*th1); V1 = max(sum(w.*(th1-mu1).^2), 1e-8);
        mu2 = sum(w.*th2); V2 = max(sum(w.*(th2-mu2).^2), 1e-8);

        th1_new = lw_a*th1 + (1-lw_a)*mu1 + lw_h*sqrt(V1)*randn(1,Ni);
        th2_new = lw_a*th2 + (1-lw_a)*mu2 + lw_h*sqrt(V2)*randn(1,Ni);
        th1_new = max(th1_new, PARAM_PRIOR(i,1)*0.05);
        th2_new = max(th2_new, PARAM_PRIOR(i,3)*0.05);

        dx_new = sample_increment(i, th1_new, th2_new, dt, Ni);

        particles{i}(1,:) = dx_new;
        particles{i}(2,:) = th1_new;
        particles{i}(3,:) = th2_new;
    end

    % Weight update
    raw_evidence = zeros(N_MODELS,1);

    for i = 1:N_MODELS
        Ni = size(particles{i},2);
        dx_p = particles{i}(1,:);
        th1 = particles{i}(2,:);
        th2 = particles{i}(3,:);

        % Likelihood
        lhood = normpdf(dy_obs(t), dx_p, sigma_eff);

        w_sum = sum(lhood);
        if w_sum < 1e-300
            weights{i}      = ones(1,Ni)/Ni;
            raw_evidence(i) = 1e-300;
        else
            weights{i}      = lhood / w_sum;
            raw_evidence(i) = w_sum / Ni;
        end
    end

    % Model probability
    model_prob = raw_evidence .* model_prob;
    model_prob = model_prob / sum(model_prob);
    model_prob = max(model_prob, 1e-6);
    model_prob = model_prob / sum(model_prob);
    model_hist(t,:) = model_prob';

    % IMM and KLD resapmling (Fox, 2003)
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

    % Mixing estimates
    dx_model = zeros(N_MODELS,1);
    p1_model = zeros(N_MODELS,1);
    p2_model = zeros(N_MODELS,1);

    for i = 1:N_MODELS
        w = weights{i};
        dx_model(i) = sum(w .* particles{i}(1,:));
        p1_model(i) = sum(w .* particles{i}(2,:));
        p2_model(i) = sum(w .* particles{i}(3,:));
    end

    dx_est(t) = model_prob' * dx_model;
    x_est_cum = x_est_cum + dx_est(t);
    x_est(t) = x_est_cum;

    % Model selection
    K_eff = min(SWITCH_K, t);
    [prob_max, best_i] = max(model_prob);

    if prob_max >= SWITCH_TAU
        committed_model = best_i;
        consec_count(:) = 0;
    else
        consec_count(best_i) = consec_count(best_i) + 1;
        consec_count(setdiff(1:N_MODELS, best_i)) = 0;
        if consec_count(best_i) >= K_eff
            committed_model = best_i;
        end
    end
    
    if committed_model == 0
        selected_model(t) = best_i;
    else
        selected_model(t) = committed_model;
    end

    % Parameter error
    p1_est(t) = p1_model(GT_MODEL); p2_est(t) = p2_model(GT_MODEL);
    np_est(t) = norm((gt_p-[p1_est(t),p2_est(t)])./DPRIOR(GT_MODEL,:));
end

% Performance metrics
rn_est = sqrt(mean(np_est.^2));
correct_flags = (selected_model == GT_MODEL);
CIR = mean(correct_flags(1:end));

%% Visualization

model_names = {'Wiener','Gamma','Inv. Gaussian','C. Poisson'};
gt_names = model_names{GT_MODEL};
tvec = (1:T)'; cols = lines(N_MODELS);

figure('Position', [100 300 1500 600]);
h = tiledlayout(2, 6, 'TileSpacing', 'compact', 'Padding', 'compact');

% Degradation increments
nexttile(1, [1, 3]);
plot(tvec, dx_true, 'k-', 'LineWidth', 5); hold on;
plot(tvec, dy_obs, 'kx', 'MarkerSize', 10);
plot(tvec, dx_est, 'LineWidth', 5, 'Color', '#0072BD');
xlabel('Time Step'); ylabel('\Deltax');
title('(a) Degradation Increment');
legend('True Increment', 'Observed Increment', 'Estimated Increment');  
grid on; set(gca, 'FontName', 'serif', 'FontSize', 14); hold off

% Cumulative degradation
nexttile(4, [1, 3]);
plot(tvec, x_true, 'k-', 'LineWidth', 5); hold on;
plot(tvec, y_obs, 'kx', 'MarkerSize', 10);
plot(tvec, x_est, 'LineWidth', 5, 'Color', '#0072BD');
xlabel('Time Step'); ylabel('x'); title('(b) Cumulative Degradation');
legend('True Cumulation', 'Observed Cumulation', 'Estimated Cumulation'); 
grid on; set(gca, 'FontName', 'serif', 'FontSize', 14); hold off

% Model selection
nexttile(7, [1, 2]);
stairs(tvec, selected_model, 'LineWidth', 5, 'Color', '#7E2F8E');
yline(GT_MODEL, 'k--', 'LineWidth', 5); xlabel('Time Step');
title('(c) Model Selection'); yticks(1:N_MODELS); yticklabels(model_names);
ylim([0.5 N_MODELS+0.5]); legend('Selected Model', 'True Model');
grid on; set(gca, 'FontName', 'serif', 'FontSize', 14); hold off

% Parameter estimates
nexttile(9, [1, 2]);
plot(tvec, p1_est, 'LineWidth', 5, 'Color', '#2E7D32');  hold on;
yline(gt_p(1), 'k--', 'LineWidth', 5);
xlabel('Time Step'); ylabel('\theta_1');
title('(d) Degradation Parameter 1');  
legend('Estimated Parameter', 'True Parameter');  
grid on; set(gca, 'FontName', 'serif', 'FontSize', 14); hold off

nexttile(11, [1, 2]);
plot(tvec, p2_est, 'LineWidth', 5, 'Color', '#C62828');  hold on;
yline(gt_p(2), 'k--', 'LineWidth', 5);
xlabel('Time Step'); ylabel('\theta_2');
title('(e) Degradation Parameter 2');  
legend('Estimated Parameter', 'True Parameter');  
grid on; set(gca, 'FontName', 'serif', 'FontSize', 14); hold off

% Console results
fprintf('\n=== Results (t=%d) ===\n',T);
fprintf('Norm RMSE : %.4f \n',rn_est);
fprintf('CIR Score : %.4f \n',CIR);

%% Functions

function [dx_true, x_true, y_obs, dy_obs] = ...
        simulate_ground_truth(model_id, params, T, dt, x0, sig_obs)

    dx_true = zeros(T,1);
    switch model_id
        case 1  % Wiener
            mu=params.wiener(1); sig=params.wiener(2);
            dx_true = mu*dt + sig*sqrt(dt)*randn(T,1);
        case 2  % Gamma
            a=params.gamma(1); b=params.gamma(2);
            for t=1:T, dx_true(t)=gamrnd(a*dt,1/b); end
        case 3  % Inverse Gaussian
            mu_ig=params.invgauss(1); lam=params.invgauss(2);
            for t=1:T, dx_true(t)=sample_ig(mu_ig*dt,lam*dt^2); end
        case 4  % Compound Poisson
            lj=params.cpoisson(1); mj=params.cpoisson(2);
            for t=1:T
                n=poissrnd(lj*dt);
                if n>0, dx_true(t)=sum(exprnd(mj,1,n)); end
            end
    end

    x_true = x0 + cumsum(dx_true);

    % Sensor measurement
    v_t    = sig_obs * randn(T,1);
    y_obs  = x_true + v_t;

    % Observed difference
    y_prev = [x0; y_obs(1:end-1)];
    dy_obs = y_obs - y_prev;
end

function dx = sample_increment(model_id, th1, th2, dt, N)
    switch model_id
        case 1
            dx = th1*dt + th2*sqrt(dt).*randn(1,N);
        case 2
            dx = zeros(1,N);
            for j=1:N
                dx(j)=gamrnd(max(th1(j)*dt,1e-6),1/max(th2(j),1e-6));
            end
        case 3
            dx = zeros(1,N);
            for j=1:N
                dx(j)=sample_ig(max(th1(j)*dt,1e-9),max(th2(j)*dt^2,1e-9));
            end
        case 4
            dx = zeros(1,N);
            for j=1:N
                n=poissrnd(max(th1(j)*dt,1e-9));
                if n>0, dx(j)=sum(exprnd(max(th2(j),1e-9),1,n)); end
            end
    end
end

function mu = expected_increment(model_id, th1, th2, dt)
    switch model_id
        case 1, mu = th1 * dt;
        case 2, mu = (th1 ./ th2) * dt;
        case 3, mu = th1 * dt;
        case 4, mu = th1 .* th2 * dt;
    end
end

function x = sample_ig(mu, lambda)
    y = randn^2; c = mu/(2*lambda);
    x1 = mu + c*mu*y - c*sqrt(max(4*mu*lambda*y+(mu*y)^2,0));
    if rand <= mu/(mu+x1),  x=x1; else,  x=mu^2/x1; end
    x = max(x,1e-12);
end

function idx = systematic_resample(weights, N_target)
    w = weights/sum(weights);
    cdf = cumsum(w); N = length(w);
    idx = zeros(1,N_target); u0 = rand/N_target;
    u = u0 + (0:N_target-1)/N_target;
    j   = 1;
    for i = 1:N_target
        while j<N && cdf(j)<u(i), j=j+1; end
        idx(i) = j;
    end
end
