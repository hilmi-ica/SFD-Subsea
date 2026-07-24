% PFHypMC.m for LPPI
% by Muhammad Hilmi, ITK NTNU

clear all; close all; clc;

%% Sweep Settings

N_MAX_LIST = [100, 200, 300, 400, 500];
LW_H_MAX_LIST = [0.10, 0.17, 0.25, 0.32, 0.40];
N_MODELS = 4; N_MC_RUNS = 20;

% Fixed settings
T = 150; dt = 1.0;
x0 = 0.0; sigma_obs = 0.1;
N_MIN = 20;

GT_PARAMS.wiener = [0.15, 0.20];
GT_PARAMS.gamma = [1.50, 8.00];
GT_PARAMS.invgauss = [0.15, 4.00];
GT_PARAMS.cpoisson = [0.30, 0.50];

LW_H_MAX = 0.25; LW_H_MIN = 0.05; LW_DECAY = 0.97;
KLD_EPSILON = 0.05; KLD_DELTA = 0.99; KLD_BINS = 12;
SWITCH_TAU = 0.55; SWITCH_K = 5;

PARAM_PRIOR = [ ...
    0.01, 0.60,  0.01, 0.60; ...
    0.20, 4.00,  1.00, 12.00; ...
    0.01, 0.60,  0.20, 10.00; ...
    0.05, 1.50,  0.05,  2.00];
DPRIOR = [0.59, 0.59; 3.80, ...
    11.00; 0.59, 9.80; 1.45, 1.95];

n_nmax = numel(N_MAX_LIST);
n_lw_h_max = numel(LW_H_MAX_LIST);

% Results
RES_RMSE = zeros(n_lw_h_max, n_nmax, N_MODELS);
RES_CIR = zeros(n_lw_h_max, n_nmax, N_MODELS);

%% Sweep Loop

total_sims = n_nmax * n_lw_h_max * N_MODELS; sim_count  = 0;
fprintf('Monte Carlo sweep: %d cells x %d runs = %d simulations\n', ...
        n_nmax*n_lw_h_max*N_MODELS, N_MC_RUNS, total_sims*N_MC_RUNS);

for gm = 1:N_MODELS
    switch gm
        case 1, gt_p = GT_PARAMS.wiener;
        case 2, gt_p = GT_PARAMS.gamma;
        case 3, gt_p = GT_PARAMS.invgauss;
        case 4, gt_p = GT_PARAMS.cpoisson;
    end

    rng(gm);

    % Ground truth
    [dx_true, x_true, y_obs, dy_obs] = ...
        simulate_ground_truth(gm, GT_PARAMS, T, dt, x0, sigma_obs);

    sigma_eff = sqrt(2) * sigma_obs;

    for lw = 1:n_lw_h_max
        LW_H_MAX = LW_H_MAX_LIST(lw);

        for ni = 1:n_nmax
            N_MAX = N_MAX_LIST(ni);

            if N_MAX < N_MIN * N_MODELS
                RES_RMSE(lw,ni,gm) = NaN;
                RES_CIR(lw,ni,gm)  = NaN;
                sim_count = sim_count + 1;
                continue
            end

            rmse_mc = zeros(N_MC_RUNS,1);
            cir_mc  = zeros(N_MC_RUNS,1);

            for mc = 1:N_MC_RUNS

                % Initialise particles
                N_init = floor(N_MAX / N_MODELS);
                particles = cell(N_MODELS,1);
                weights = cell(N_MODELS,1);
                model_prob = ones(N_MODELS,1) / N_MODELS;

                for i = 1:N_MODELS
                    p1 = PARAM_PRIOR(i,1) + ...
                         (PARAM_PRIOR(i,2)-PARAM_PRIOR(i,1))*rand(1,N_init);
                    p2 = PARAM_PRIOR(i,3) + ...
                         (PARAM_PRIOR(i,4)-PARAM_PRIOR(i,3))*rand(1,N_init);
                    particles{i} = [zeros(1,N_init); p1; p2];
                    weights{i} = ones(1,N_init)/N_init;
                end

                selected_model = zeros(T,1);
                consec_count = zeros(N_MODELS,1); 
                committed_model = 0;
                np_est = zeros(T,1);
                lw_h = LW_H_MAX;

                % Main filter loop
                for t = 1:T
                    lw_h = max(LW_H_MIN, lw_h * LW_DECAY);
                    lw_a = sqrt(1 - lw_h^2);

                    % Liu-West parameter perturbation
                    for i = 1:N_MODELS
                        Ni = size(particles{i},2); w = weights{i};
                        th1 = particles{i}(2,:); th2 = particles{i}(3,:);

                        mu1 = sum(w.*th1); V1 = max(sum(w.*(th1-mu1).^2),1e-8);
                        mu2 = sum(w.*th2); V2 = max(sum(w.*(th2-mu2).^2),1e-8);

                        th1_n = lw_a*th1+(1-lw_a)*mu1+lw_h*sqrt(V1)*randn(1,Ni);
                        th2_n = lw_a*th2+(1-lw_a)*mu2+lw_h*sqrt(V2)*randn(1,Ni);
                        th1_n = max(th1_n, PARAM_PRIOR(i,1)*0.05);
                        th2_n = max(th2_n, PARAM_PRIOR(i,3)*0.05);

                        particles{i}(1,:) = sample_increment(i,th1_n,th2_n,dt,Ni);
                        particles{i}(2,:) = th1_n;
                        particles{i}(3,:) = th2_n;
                    end

                    % Weight update (sliding window)
                    raw_evidence = zeros(N_MODELS,1);
                    for i = 1:N_MODELS
                        Ni = size(particles{i},2);
                        dx_p = particles{i}(1,:);
                        th1 = particles{i}(2,:);
                        th2 = particles{i}(3,:);

                        lhood = normpdf(dy_obs(t), dx_p, sigma_eff);

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

                    % Budget allocation + KLD resampling
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
                        x_lo = min(dx_s)-1e-6; x_hi = max(dx_s)+1e-6;
                        edges = linspace(x_lo, x_hi, KLD_BINS+1);
                        k = max(2, sum(histcounts(dx_s,edges)>0));
                        d = 9*(k-1);
                        N_kld = ceil((k-1)/(2*KLD_EPSILON)* ...
                                (1-2/d+z_kld*sqrt(2/d))^3);
                        N_target = max(N_MIN, min(N_alloc(i), N_kld));
                        idx = systematic_resample(weights{i}, N_target);
                        particles{i} = particles{i}(:,idx);
                        weights{i} = ones(1,N_target)/N_target;
                    end

                    % Point estimates
                    p1_model = zeros(N_MODELS,1);
                    p2_model = zeros(N_MODELS,1);
                    for i = 1:N_MODELS
                        w = weights{i};
                        p1_model(i) = sum(w .* particles{i}(2,:));
                        p2_model(i) = sum(w .* particles{i}(3,:));
                    end

                    % Model selection (two-tier)
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

                    % Parameter norm error
                    np_est(t) = norm((gt_p-[p1_model(gm),p2_model(gm)]) ...
                        ./DPRIOR(gm,:));
                end 

                % RMSE Metrics for an MC run
                rmse_mc(mc) = sqrt(mean(np_est.^2));

                % CIR with(/out) warm-up
                % t_start = min(SWITCH_K + 1, T);
                t_start = min(1, T);
                correct_flags = (selected_model(t_start:end) == gm);
                cir_mc(mc) = mean(correct_flags);
            end

            RES_RMSE(lw,ni,gm) = mean(rmse_mc);
            RES_CIR(lw,ni,gm) = mean(cir_mc);

            sim_count = sim_count + 1;
            fprintf('[%3d/%3d] Model %d | N_MAX=%3d | LW_H_MAX=%.2f | RMSE=%.4f | CIR=%.3f\n', ...
                    sim_count, total_sims, gm, N_MAX, LW_H_MAX, ...
                    RES_RMSE(lw,ni,gm), RES_CIR(lw,ni,gm));
        end 
    end
end 

fprintf('Sweep complete.\n');

%% Heatmap Visualization

model_names = {'Wiener','Gamma','Inv. Gaussian','C. Poisson'};
nmax_labels = arrayfun(@(x) num2str(x), N_MAX_LIST, 'UniformOutput', false);
lw_h_max_labels = arrayfun(@(x) num2str(x), LW_H_MAX_LIST, 'UniformOutput', false);

cmap = flipud(gray);

figure('Position',[60 60 1600 420]);

for gm = 1:N_MODELS
    ax = subplot(1, N_MODELS, gm);

    rmse_mat = RES_RMSE(:,:,gm);
    cir_mat = RES_CIR(:,:,gm);

    % Display the colour image
    imagesc(rmse_mat, [0 1]);
    colormap(ax, cmap);
    axis xy; axis square;

    % Axis labels
    xticks(1:n_nmax); xticklabels(nmax_labels);
    yticks(1:n_lw_h_max); yticklabels(lw_h_max_labels);
    xlabel('No. of Particles','FontName','serif','FontSize',14);
    ylabel('Kernel Smoothing','FontName','serif','FontSize',14);
    title(model_names{gm}, 'FontName','serif','FontSize',14,'FontWeight','bold');
    set(ax,'FontName','serif','FontSize',14,'TickDir','out');

    % Annotate CIR percentage
    for lw = 1:n_lw_h_max
        for ni = 1:n_nmax
            cir_val = cir_mat(lw,ni) * 100;
            bg_level = rmse_mat(lw,ni);

            % Text colour relative to background
            if bg_level > 0.5
                txt_col = [1 1 1];
            else
                txt_col = [0 0 0];
            end

            text(ni, lw, sprintf('%.0f%%', cir_val), ...
                 'HorizontalAlignment','center', ...
                 'VerticalAlignment','middle', ...
                 'Color', txt_col, ...
                 'FontName','serif','FontSize',11,'FontWeight','bold');
        end
    end
end

% Shared colourbar
cb = colorbar('Position',[0.93 0.15 0.012 0.70]);
cb.Label.String = ['L2-Norm NRMSE (CIR in %)'];
cb.Label.FontName = 'serif';
cb.Label.FontSize = 14;
% Annotate actual RMSE range on colourbar
cb.Ticks = [0 0.5 1];

%% Local Functions

function [dx_true, x_true, y_obs, dy_obs] = ...
        simulate_ground_truth(model_id, params, T, dt, x0, sig_obs)
    dx_true = zeros(T,1);
    switch model_id
        case 1
            mu=params.wiener(1); sig=params.wiener(2);
            dx_true = mu*dt + sig*sqrt(dt)*randn(T,1);
        case 2
            a=params.gamma(1); b=params.gamma(2);
            for t=1:T, dx_true(t)=gamrnd(a*dt,1/b); end
        case 3
            mu_ig=params.invgauss(1); lam=params.invgauss(2);
            for t=1:T, dx_true(t)=sample_ig(mu_ig*dt,lam*dt^2); end
        case 4
            lj=params.cpoisson(1); mj=params.cpoisson(2);
            for t=1:T
                n=poissrnd(lj*dt);
                if n>0, dx_true(t)=sum(exprnd(mj,1,n)); end
            end
    end
    x_true = x0 + cumsum(dx_true);
    v_t    = sig_obs * randn(T,1);
    y_obs  = x_true + v_t;
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
    if rand <= mu/(mu+x1), x=x1; else, x=mu^2/x1; end
    x = max(x,1e-12);
end

function idx = systematic_resample(weights, N_target)
    w   = weights/sum(weights);
    cdf = cumsum(w); N = length(w);
    idx = zeros(1,N_target); u0 = rand/N_target;
    u   = u0 + (0:N_target-1)/N_target;
    j   = 1;
    for i = 1:N_target
        while j<N && cdf(j)<u(i), j=j+1; end
        idx(i) = j;
    end
end