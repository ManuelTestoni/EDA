%% ========================================================================
%  MAIN_ANALYSIS_MOSTI.m - Classification Analysis of Mosti (Grape Must) Dataset
%  Methods: PCA (exploratory), SIMCA (class modeling), PLS-DA (discriminant)
%  Dataset: mosti.mat (98 samples, 6 anthocyanin HPLC variables, 5 grape varieties)
%  ========================================================================
%  This script is self-contained and does NOT require PLS Toolbox.
%  It uses only standard MATLAB + Statistics Toolbox + the utility
%  functions provided in the SIMCA folder.
%
%  Grape varieties:
%    1 = Ancellotta (A)
%    2 = Montepulciano (M)
%    3 = Lambrusco Pugliese (LP)
%    4 = Sangiovese (S)
%    5 = Nero d'Avola (N)
%
%  Variables (anthocyanin HPLC areas %):
%    DPD%, CYD%, PTD%, PND%, MVD%, R lib/lrg
%
%  Two vintages: 2000 and 2001 (combined for classification).
%
%  To run: simply execute this script in MATLAB.
%  All plots are saved automatically in the 'plots2/' folder.
%  ========================================================================
clear; close all; clc;
fprintf('============================================================\n');
fprintf('  MOSTI (GRAPE MUST) CLASSIFICATION ANALYSIS\n');
fprintf('  SIMCA + PLS-DA | %s\n', datestr(now));
fprintf('============================================================\n\n');

%% ---- 0. SETUP -----------------------------------------------------------
basePath = fileparts(mfilename('fullpath'));
if isempty(basePath); basePath = pwd; end
addpath(fullfile(basePath, 'SIMCA'));
addpath(fullfile(basePath, 'PLS_DA'));

plotDir = fullfile(basePath, 'plots2');
if ~exist(plotDir, 'dir'); mkdir(plotDir); end

rng(42); % reproducibility

% Force graphics system initialization (prevents blank first figures on macOS)
hWarmup = figure('Position', [0 0 100 100]);
plot(0,0); drawnow; pause(0.2);
close(hWarmup);
set(0, 'DefaultFigurePaperPositionMode', 'auto');

% Class color palette (5 grape varieties)
classColors = [0.20 0.40 0.80;   % A  - blue
               0.85 0.20 0.20;   % M  - red
               0.15 0.70 0.30;   % LP - green
               0.80 0.20 0.80;   % S  - magenta
               0.95 0.60 0.10];  % N  - orange

%% ---- 1. DATA LOADING ----------------------------------------------------
fprintf('[1] Loading data...\n');
dataFile = fullfile(basePath, 'mosti.mat');
loadedData = load(dataFile);

% Extract variables from .mat file
X_full = loadedData.mosti;              % 98 x 6 data matrix

% Class assignment vector
classid_v = loadedData.classid_v;
y_full = classid_v(:);                  % ensure column vector

% Sample names
if ischar(loadedData.nameobj_mosti)
    nameobj_mosti = cellstr(loadedData.nameobj_mosti);
else
    nameobj_mosti = loadedData.nameobj_mosti;
end
nameobj_mosti = strtrim(nameobj_mosti);

% Variable names
if ischar(loadedData.namevar_mosti)
    namevar_mosti = cellstr(loadedData.namevar_mosti);
else
    namevar_mosti = loadedData.namevar_mosti;
end
namevar_mosti = strtrim(namevar_mosti);

% Use names from file; provide fallback
if length(namevar_mosti) == size(X_full, 2)
    featNames = namevar_mosti(:)';
else
    featNames = {'DPD%','CYD%','PTD%','PND%','MVD%','R lib/lrg'};
end

% Class definitions
classNames     = {'A','M','LP','S','N'};
classFullNames = {'Ancellotta','Montepulciano','Lambrusco Pugliese','Sangiovese','Nero d''Avola'};

[N_total, M_vars] = size(X_full);
nClasses = length(classNames);

% Extract vintage info from sample names
annata = zeros(N_total, 1);
for i = 1:N_total
    nome = nameobj_mosti{i};
    if contains(nome, '_00') || endsWith(nome, '00')
        annata(i) = 2000;
    elseif contains(nome, '_01') || endsWith(nome, '01')
        annata(i) = 2001;
    end
end

fprintf('   Samples: %d | Variables: %d | Classes: %d\n', N_total, M_vars, nClasses);
for ic = 1:nClasses
    n00 = sum(y_full == ic & annata == 2000);
    n01 = sum(y_full == ic & annata == 2001);
    fprintf('   Class %d (%s): %d samples  [2000: %d | 2001: %d]\n', ...
        ic, classFullNames{ic}, sum(y_full==ic), n00, n01);
end
fprintf('\n');

%% ---- 1b. PREPROCESSING ANALYSIS ----------------------------------------
fprintf('[1b] Preprocessing Analysis...\n');

% Preprocessing strategy: AUTOSCALING (mean-centering + unit variance scaling)
% Motivation: the 6 anthocyanin variables share the same unit (area %)
% but have very different ranges (e.g. MVD% >> CYD%). Without scaling,
% high-variance variables would dominate PCA and classification models.
% Autoscaling gives equal weight to all variables.

preprocessing_method = 'Autoscaling (Mean-Centering + Unit Variance)';

% Compute raw statistics
raw_means = mean(X_full);
raw_stds  = std(X_full);
raw_mins  = min(X_full);
raw_maxs  = max(X_full);
raw_ranges = raw_maxs - raw_mins;

fprintf('   Preprocessing: %s\n', preprocessing_method);
fprintf('   Raw Variable Statistics:\n');
fprintf('   %-12s  Mean     Std      Min      Max      Range\n', 'Variable');
for iv = 1:M_vars
    fprintf('   %-12s  %7.2f  %7.2f  %7.2f  %7.2f  %7.2f\n', ...
        featNames{iv}, raw_means(iv), raw_stds(iv), raw_mins(iv), raw_maxs(iv), raw_ranges(iv));
end

% Plot: Raw vs Autoscaled comparison
X_auto_preview = (X_full - raw_means) ./ raw_stds;
fig_prepr = figure('Position',[100 100 1200 500]);
subplot(1,2,1);
boxplot(X_full);
set(gca, 'XTickLabel', featNames); xtickangle(30);
ylab = ylabel('Raw Value (area %)'); title('Raw Data');
grid on;
subplot(1,2,2);
boxplot(X_auto_preview);
set(gca, 'XTickLabel', featNames); xtickangle(30);
ylab = ylabel('Autoscaled Value'); title('After Autoscaling');
grid on;
try sgtitle(sprintf('Preprocessing: %s', preprocessing_method), 'FontSize', 14); ...
catch; annotation('textbox',[0.2 0.96 0.6 0.04],'String',sprintf('Preprocessing: %s', preprocessing_method),'FontSize',14,'FontWeight','bold','HorizontalAlignment','center','EdgeColor','none'); end
drawnow; pause(0.1);
saveas(fig_prepr, fullfile(plotDir, '01b_preprocessing_comparison.png'));
close(fig_prepr);

fprintf('   Preprocessing plot saved.\n\n');

%% ---- 2. EXPLORATORY DATA ANALYSIS (EDA) --------------------------------
fprintf('[2] Exploratory Data Analysis...\n');

% ---- 2.1 Boxplot of raw variables ----
fig1 = figure('Position',[100 100 900 500]);
boxplot(X_full);
set(gca, 'XTickLabel', featNames);
ylabel('Area (%)'); title('Distribution of Anthocyanin Variables');
xtickangle(30);
grid on;
drawnow; pause(0.1);
saveas(fig1, fullfile(plotDir, '01_raw_data_boxplot.png'));
close(fig1);

% ---- 2.2 Boxplot per class ----
fig2 = figure('Position',[100 100 1200 600]);
for iv = 1:M_vars
    subplot(2, 3, iv);
    data_var = [];
    groups_var = [];
    for ic = 1:nClasses
        vals = X_full(y_full==ic, iv);
        data_var = [data_var; vals];
        groups_var = [groups_var; ic*ones(length(vals),1)];
    end
    boxplot(data_var, groups_var);
    set(gca, 'XTickLabel', classNames);
    title(featNames{iv}, 'FontSize', 10);
    ylabel('Area (%)');
    grid on;
end
try sgtitle('Anthocyanin Distribution by Grape Variety', 'FontSize', 14); catch; annotation('textbox',[0.2 0.96 0.6 0.04],'String','Anthocyanin Distribution by Grape Variety','FontSize',14,'FontWeight','bold','HorizontalAlignment','center','EdgeColor','none'); end
drawnow; pause(0.1);
saveas(fig2, fullfile(plotDir, '02_boxplot_by_class.png'));
close(fig2);

% ---- 2.2b Boxplot per vintage (to assess vintage effect) ----
fig2b = figure('Position',[100 100 1200 600]);
for iv = 1:M_vars
    subplot(2, 3, iv);
    data_var = [];
    groups_var = [];
    for ia = [2000, 2001]
        vals = X_full(annata==ia, iv);
        data_var = [data_var; vals];
        groups_var = [groups_var; ia*ones(length(vals),1)];
    end
    boxplot(data_var, groups_var);
    title(featNames{iv}, 'FontSize', 10);
    ylabel('Area (%)');
    grid on;
end
try sgtitle('Anthocyanin Distribution by Vintage (2000 vs 2001)', 'FontSize', 14); catch; annotation('textbox',[0.15 0.96 0.7 0.04],'String','Anthocyanin Distribution by Vintage (2000 vs 2001)','FontSize',14,'FontWeight','bold','HorizontalAlignment','center','EdgeColor','none'); end
drawnow; pause(0.1);
saveas(fig2b, fullfile(plotDir, '02b_boxplot_by_vintage.png'));
close(fig2b);

% ---- 2.3 Correlation matrix ----
fig3 = figure('Position',[100 100 650 550]);
corrMat = corrcoef(X_full);
imagesc(corrMat);
colorbar; colormap(jet);
set(gca, 'XTick', 1:M_vars, 'XTickLabel', featNames, 'YTick', 1:M_vars, 'YTickLabel', featNames);
xtickangle(30);
title('Correlation Matrix of Anthocyanin Variables');
% Add text annotations
for ii = 1:M_vars
    for jj = 1:M_vars
        text(jj, ii, sprintf('%.2f', corrMat(ii,jj)), 'HorizontalAlignment','center','FontSize',7);
    end
end
drawnow; pause(0.1);
saveas(fig3, fullfile(plotDir, '03_correlation_matrix.png'));
close(fig3);

% ---- 2.4 PCA on full dataset (autoscaled) ----
fprintf('   Running PCA...\n');
mx = mean(X_full);
sx = std(X_full);
X_auto = (X_full - mx) ./ sx; % autoscaling

[U_pca, S_pca, V_pca] = svd(X_auto, 'econ');
nPC_max = min(N_total, M_vars);
eigenvalues = diag(S_pca).^2 / (N_total - 1);
explainedVar = 100 * eigenvalues / sum(eigenvalues);
cumVar = cumsum(explainedVar);
scores_pca = U_pca * S_pca;
loadings_pca = V_pca;

fprintf('   Explained variance: ');
for ipc = 1:min(5, nPC_max)
    fprintf('PC%d=%.1f%% ', ipc, explainedVar(ipc));
end
fprintf('\n');

% ---- 2.5 Scree plot ----
fig4 = figure('Position',[100 100 800 400]);
subplot(1,2,1);
bar(explainedVar(1:nPC_max), 'FaceColor', [0.3 0.5 0.8]);
xlabel('Principal Component'); ylabel('Explained Variance (%)');
title('Scree Plot'); grid on;
subplot(1,2,2);
plot(1:nPC_max, cumVar(1:nPC_max), '-o', 'LineWidth', 2, 'Color', [0.3 0.5 0.8]);
hold on;
xl95 = xlim; plot(xl95, [95 95], '--r', 'LineWidth', 1.5); text(xl95(2), 95, '95%', 'Color', 'r', 'FontSize', 9);
xlabel('Number of PCs'); ylabel('Cumulative Variance (%)');
title('Cumulative Explained Variance'); grid on;
drawnow; pause(0.1);
saveas(fig4, fullfile(plotDir, '04_pca_scree_plot.png'));
close(fig4);

% ---- 2.6 Score plots (colored by variety) ----
fig5 = figure('Position',[100 100 1100 500]);
subplot(1,2,1);
hold on;
for ic = 1:nClasses
    idx = y_full == ic;
    scatter(scores_pca(idx,1), scores_pca(idx,2), 40, classColors(ic,:), 'filled', 'MarkerEdgeColor','k','LineWidth',0.3);
end
xlabel(sprintf('PC1 (%.1f%%)', explainedVar(1)));
ylabel(sprintf('PC2 (%.1f%%)', explainedVar(2)));
title('PCA Score Plot: PC1 vs PC2 (by Variety)');
legend(classFullNames, 'Location','best'); grid on;

subplot(1,2,2);
hold on;
for ic = 1:nClasses
    idx = y_full == ic;
    scatter(scores_pca(idx,1), scores_pca(idx,3), 40, classColors(ic,:), 'filled', 'MarkerEdgeColor','k','LineWidth',0.3);
end
xlabel(sprintf('PC1 (%.1f%%)', explainedVar(1)));
ylabel(sprintf('PC3 (%.1f%%)', explainedVar(3)));
title('PCA Score Plot: PC1 vs PC3 (by Variety)');
legend(classFullNames, 'Location','best'); grid on;
drawnow; pause(0.1);
saveas(fig5, fullfile(plotDir, '05_pca_scores_variety.png'));
close(fig5);

% ---- 2.6b Score plots colored by vintage ----
fig5v = figure('Position',[100 100 1100 500]);
vintageColors = [0.2 0.6 0.2; 0.8 0.3 0.1]; % 2000=green, 2001=brown
vintageLabels = {'2000','2001'};
subplot(1,2,1);
hold on;
for iv = 1:2
    yr = [2000, 2001];
    idx = annata == yr(iv);
    scatter(scores_pca(idx,1), scores_pca(idx,2), 40, vintageColors(iv,:), 'filled', 'MarkerEdgeColor','k','LineWidth',0.3);
end
xlabel(sprintf('PC1 (%.1f%%)', explainedVar(1)));
ylabel(sprintf('PC2 (%.1f%%)', explainedVar(2)));
title('PCA Score Plot: PC1 vs PC2 (by Vintage)');
legend(vintageLabels, 'Location','best'); grid on;

subplot(1,2,2);
hold on;
for iv = 1:2
    yr = [2000, 2001];
    idx = annata == yr(iv);
    scatter(scores_pca(idx,1), scores_pca(idx,3), 40, vintageColors(iv,:), 'filled', 'MarkerEdgeColor','k','LineWidth',0.3);
end
xlabel(sprintf('PC1 (%.1f%%)', explainedVar(1)));
ylabel(sprintf('PC3 (%.1f%%)', explainedVar(3)));
title('PCA Score Plot: PC1 vs PC3 (by Vintage)');
legend(vintageLabels, 'Location','best'); grid on;
drawnow; pause(0.1);
saveas(fig5v, fullfile(plotDir, '05b_pca_scores_vintage.png'));
close(fig5v);

% ---- 2.7 3D Score plot ----
fig5b = figure('Position',[100 100 700 600]);
hold on;
for ic = 1:nClasses
    idx = y_full == ic;
    scatter3(scores_pca(idx,1), scores_pca(idx,2), scores_pca(idx,3), 40, classColors(ic,:), 'filled', 'MarkerEdgeColor','k','LineWidth',0.3);
end
xlabel(sprintf('PC1 (%.1f%%)', explainedVar(1)));
ylabel(sprintf('PC2 (%.1f%%)', explainedVar(2)));
zlabel(sprintf('PC3 (%.1f%%)', explainedVar(3)));
title('PCA 3D Score Plot (by Variety)');
legend(classFullNames, 'Location','best');
grid on; view(30,25);
drawnow; pause(0.1);
saveas(fig5b, fullfile(plotDir, '06_pca_scores_3d.png'));
close(fig5b);

% ---- 2.8 Loading plot ----
fig6 = figure('Position',[100 100 900 400]);
subplot(1,2,1);
nLoadPC = min(3, nPC_max);
bar(loadings_pca(:,1:nLoadPC));
set(gca, 'XTick', 1:M_vars, 'XTickLabel', featNames);
xtickangle(30); ylabel('Loading Value');
title(sprintf('PCA Loadings (PC1-PC%d)', nLoadPC));
pcLabels = cell(1, nLoadPC);
for ipc = 1:nLoadPC; pcLabels{ipc} = sprintf('PC%d', ipc); end
legend(pcLabels, 'Location','best'); grid on;

subplot(1,2,2);
hold on;
for iv = 1:M_vars
    plot([0 loadings_pca(iv,1)], [0 loadings_pca(iv,2)], '-', 'LineWidth', 1.5);
    text(loadings_pca(iv,1)*1.08, loadings_pca(iv,2)*1.08, featNames{iv}, 'FontSize', 9);
end
xlabel(sprintf('PC1 Loading (%.1f%%)', explainedVar(1)));
ylabel(sprintf('PC2 Loading (%.1f%%)', explainedVar(2)));
title('Loading Plot (PC1 vs PC2)');
grid on; axis equal;
th = 0:0.01:2*pi; plot(cos(th), sin(th), '--', 'Color', [0.7 0.7 0.7]);
drawnow; pause(0.1);
saveas(fig6, fullfile(plotDir, '07_pca_loadings.png'));
close(fig6);

% ---- 2.9 Biplot (scores + loadings on same plot) ----
fig_biplot = figure('Position',[100 100 800 650]);
hold on;
% Normalize scores to [-1, 1] range for overlay
sc_norm = scores_pca(:,1:2) ./ max(abs(scores_pca(:,1:2)));
for ic = 1:nClasses
    idx = y_full == ic;
    scatter(sc_norm(idx,1), sc_norm(idx,2), 30, classColors(ic,:), 'filled', 'MarkerFaceAlpha', 0.5);
end
% Overlay loadings as arrows
for iv = 1:M_vars
    quiver(0, 0, loadings_pca(iv,1), loadings_pca(iv,2), 'k', 'LineWidth', 2, 'MaxHeadSize', 0.3);
    text(loadings_pca(iv,1)*1.12, loadings_pca(iv,2)*1.12, featNames{iv}, ...
        'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.1 0.1 0.1]);
end
xlabel(sprintf('PC1 (%.1f%%)', explainedVar(1)));
ylabel(sprintf('PC2 (%.1f%%)', explainedVar(2)));
title('PCA Biplot (Scores + Loadings)');
legend(classFullNames, 'Location','best');
grid on; axis equal;
drawnow; pause(0.1);
saveas(fig_biplot, fullfile(plotDir, '07b_pca_biplot.png'));
close(fig_biplot);

fprintf('   EDA plots saved.\n\n');

%% ---- 3. TRAIN / TEST SPLIT (Stratified 70/30) --------------------------
fprintf('[3] Stratified Train/Test Split (70/30)...\n');
trainIdx = false(N_total, 1);
testIdx = false(N_total, 1);

for ic = 1:nClasses
    class_idx = find(y_full == ic);
    nClass = length(class_idx);
    nTrain = round(0.7 * nClass);
    perm = class_idx(randperm(nClass));
    trainIdx(perm(1:nTrain)) = true;
    testIdx(perm(nTrain+1:end)) = true;
end

X_train = X_full(trainIdx, :);
y_train = y_full(trainIdx);
X_test = X_full(testIdx, :);
y_test = y_full(testIdx);

fprintf('   Training set: %d samples\n', sum(trainIdx));
fprintf('   Test set:     %d samples\n', sum(testIdx));
for ic = 1:nClasses
    fprintf('   Class %d (%s): Train=%d, Test=%d\n', ic, classNames{ic}, ...
        sum(y_train==ic), sum(y_test==ic));
end
fprintf('\n');

%% ---- 4. SIMCA ANALYSIS -------------------------------------------------
fprintf('[4] SIMCA Analysis...\n');
fprintf('   Running cross-validation...\n');

maxPC_simca = min(5, M_vars - 1); % max PCs to evaluate (5 for 6 variables)
nSegCV = 5;      % number of CV segments (venetian blinds)
conf_level = 0.95;

N_tr = size(X_train, 1);

% Storage for CV results
sens_cv = zeros(nClasses, maxPC_simca);
spec_cv = zeros(nClasses, maxPC_simca);
eff_cv = zeros(nClasses, maxPC_simca);
confmat_cv = cell(1, maxPC_simca);

for npc = 1:maxPC_simca
    % Venetian blinds CV
    ssmatrix = zeros(nClasses, nClasses);
    predclass_cv = zeros(N_tr, 1);
    
    for seg = 1:nSegCV
        % Venetian blinds: segment indices
        test_cv_idx = (seg:nSegCV:N_tr)';
        train_cv_idx = setdiff(1:N_tr, test_cv_idx)';
        
        Xseg_tr = X_train(train_cv_idx, :);
        yseg_tr = y_train(train_cv_idx);
        Xseg_ts = X_train(test_cv_idx, :);
        yseg_ts = y_train(test_cv_idx);
        
        % Build SIMCA model on training segment
        [simca_seg] = build_simca_model(Xseg_tr, yseg_tr, npc*ones(nClasses,1), conf_level);
        
        % Predict test segment
        [pred_seg] = predict_simca(Xseg_ts, yseg_ts, simca_seg);
        
        % Accumulate sensitivity/specificity matrix
        ssmatrix = ssmatrix + pred_seg.SIMCA.sensspec;
        predclass_cv(test_cv_idx) = pred_seg.SIMCA.predclass;
    end
    
    confmat_cv{npc} = ssmatrix;
    
    % Calculate sensitivity and specificity from accumulated matrix
    for ic = 1:nClasses
        nc_ic = sum(y_train == ic);
        sens_cv(ic, npc) = ssmatrix(ic, ic) / nc_ic;
        others = setdiff(1:nClasses, ic);
        nc_others = N_tr - nc_ic;
        spec_cv(ic, npc) = sum(ssmatrix(ic, others)) / nc_others;
    end
end

eff_cv = sqrt(sens_cv .* spec_cv);

% ---- 4.1 Plot: Sensitivity, Specificity, Efficiency in CV ----
fig7 = figure('Position',[100 100 1000 700]);
subplot(3,1,1);
plot(1:maxPC_simca, sens_cv', '-o', 'LineWidth', 1.5);
title('SIMCA Cross-Validation: Sensitivity');
xlabel('Number of PCs'); ylabel('Sensitivity');
legend(classFullNames, 'Location','best'); grid on;
ylim([0 1.05]);

subplot(3,1,2);
plot(1:maxPC_simca, spec_cv', '-o', 'LineWidth', 1.5);
title('SIMCA Cross-Validation: Specificity');
xlabel('Number of PCs'); ylabel('Specificity');
legend(classFullNames, 'Location','best'); grid on;
ylim([0 1.05]);

subplot(3,1,3);
plot(1:maxPC_simca, eff_cv', '-o', 'LineWidth', 1.5);
title('SIMCA Cross-Validation: Efficiency');
xlabel('Number of PCs'); ylabel('Efficiency = sqrt(Sens*Spec)');
legend(classFullNames, 'Location','best'); grid on;
ylim([0 1.05]);
drawnow; pause(0.1);
saveas(fig7, fullfile(plotDir, '08_simca_cv_metrics.png'));
close(fig7);

% ---- 4.2 Select optimal PCs per class (max efficiency) ----
optPC_simca = zeros(nClasses, 1);
for ic = 1:nClasses
    [~, optPC_simca(ic)] = max(eff_cv(ic, :));
end
fprintf('   Optimal PCs per class (max CV efficiency):\n');
for ic = 1:nClasses
    fprintf('   Class %d (%s): %d PCs (Eff=%.3f, Sens=%.3f, Spec=%.3f)\n', ...
        ic, classNames{ic}, optPC_simca(ic), ...
        eff_cv(ic, optPC_simca(ic)), sens_cv(ic, optPC_simca(ic)), spec_cv(ic, optPC_simca(ic)));
end

% ---- 4.3 Build final SIMCA model on full training set ----
fprintf('   Building final SIMCA model...\n');
[simca_final] = build_simca_model(X_train, y_train, optPC_simca, conf_level);

% ---- 4.4 Predict training set ----
[pred_train_simca] = predict_simca(X_train, y_train, simca_final);
confmat_train_simca = pred_train_simca.SIMCA.sensspec;
predclass_train_simca = pred_train_simca.SIMCA.predclass;

% ---- 4.5 Predict test set ----
fprintf('   Predicting test set...\n');
[pred_test_simca] = predict_simca(X_test, y_test, simca_final);
confmat_test_simca = pred_test_simca.SIMCA.sensspec;
predclass_test_simca = pred_test_simca.SIMCA.predclass;

% ---- 4.6 Compute and display metrics ----
fprintf('\n   === SIMCA RESULTS ===\n');

% Training set metrics
fprintf('   --- Training Set ---\n');
[sens_tr_s, spec_tr_s, eff_tr_s, acc_tr_s] = compute_class_metrics(y_train, predclass_train_simca, nClasses);
for ic = 1:nClasses
    fprintf('   Class %d (%s): Sens=%.3f Spec=%.3f Eff=%.3f\n', ...
        ic, classNames{ic}, sens_tr_s(ic), spec_tr_s(ic), eff_tr_s(ic));
end
fprintf('   Overall accuracy: %.1f%%\n', acc_tr_s*100);

% Test set metrics
fprintf('   --- Test Set ---\n');
[sens_ts_s, spec_ts_s, eff_ts_s, acc_ts_s] = compute_class_metrics(y_test, predclass_test_simca, nClasses);
for ic = 1:nClasses
    fprintf('   Class %d (%s): Sens=%.3f Spec=%.3f Eff=%.3f\n', ...
        ic, classNames{ic}, sens_ts_s(ic), spec_ts_s(ic), eff_ts_s(ic));
end
fprintf('   Overall accuracy: %.1f%%\n\n', acc_ts_s*100);

% ---- 4.7 Plot: Score Distance vs Orthogonal Distance per class ----
for ic = 1:nClasses
    fig_sd = figure('Position',[100 100 800 600]);
    t2lim = simca_final.PCmodels{ic}.critvals(1);
    qlim  = simca_final.PCmodels{ic}.critvals(2);
    
    % Training set
    tsq_tr = simca_final.PCmodels{ic}.tsq;
    q_tr = simca_final.PCmodels{ic}.qres;
    
    % Test set
    tsq_ts = pred_test_simca.PCmodels{ic}.tsq;
    q_ts = pred_test_simca.PCmodels{ic}.qres;
    
    hold on;
    % Plot training samples
    for jc = 1:nClasses
        idx_tr = y_train == jc;
        scatter(tsq_tr(idx_tr)./t2lim, q_tr(idx_tr)./qlim, 40, classColors(jc,:), 'o', 'LineWidth', 1.2);
    end
    % Plot test samples
    for jc = 1:nClasses
        idx_ts = y_test == jc;
        scatter(tsq_ts(idx_ts)./t2lim, q_ts(idx_ts)./qlim, 60, classColors(jc,:), 'd', 'filled', 'MarkerEdgeColor','k');
    end
    % Combined boundary (circle of radius sqrt(2))
    theta = 0:0.001:2*pi;
    plot(sqrt(2)*cos(theta), sqrt(2)*sin(theta), '-r', 'LineWidth', 1.5);
    
    axis([0 5 0 5]);
    xlabel('Score Distance (T^2 / T^2_{lim})'); ylabel('Orthogonal Distance (Q / Q_{lim})');
    title(sprintf('SIMCA: Class %d (%s) | %d PCs', ic, classFullNames{ic}, optPC_simca(ic)));
    
    legendEntries = cell(1, 2*nClasses);
    for jc = 1:nClasses
        legendEntries{jc} = sprintf('%s (train)', classNames{jc});
        legendEntries{nClasses+jc} = sprintf('%s (test)', classNames{jc});
    end
    legend(legendEntries, 'Location','best', 'FontSize', 7);
    grid on;
    drawnow; pause(0.1);
    saveas(fig_sd, fullfile(plotDir, sprintf('09_simca_SDvsOD_class%d.png', ic)));
    close(fig_sd);
end

% ---- 4.8 Plot: Confusion matrices ----
fig_cm1 = figure('Position',[100 100 1000 450]);
subplot(1,2,1);
plot_confusion_matrix(y_train, predclass_train_simca, classNames, 'SIMCA - Training Set');
subplot(1,2,2);
plot_confusion_matrix(y_test, predclass_test_simca, classNames, 'SIMCA - Test Set');
drawnow; pause(0.1);
saveas(fig_cm1, fullfile(plotDir, '10_simca_confusion_matrices.png'));
close(fig_cm1);

% ---- 4.9 Discriminant Power ----
fprintf('   Computing discriminant power...\n');
dpow = compute_discriminant_power(X_train, y_train, simca_final);
fig_dp = figure('Position',[100 100 800 400]);
bar(dpow, 'FaceColor', [0.3 0.6 0.8]);
hold on;
xl_dp = xlim; plot(xl_dp, [1 1]*prctile(dpow,95), '--g', 'LineWidth',1.5); text(xl_dp(2), prctile(dpow,95), '95th pctl', 'Color','g');
plot(xl_dp, [1 1]*prctile(dpow,99), '--r', 'LineWidth',1.5); text(xl_dp(2), prctile(dpow,99), '99th pctl', 'Color','r');
plot(xl_dp, [1 1]*mean(dpow), ':k', 'LineWidth',1.2); text(xl_dp(2), mean(dpow), 'Mean', 'Color','k');
set(gca, 'XTick', 1:M_vars, 'XTickLabel', featNames);
xtickangle(30);
xlabel('Variable'); ylabel('Discriminant Power');
title('SIMCA Discriminant Power');
grid on;
drawnow; pause(0.1);
saveas(fig_dp, fullfile(plotDir, '11_simca_discriminant_power.png'));
close(fig_dp);

% ---- 4.10 Loadings per class ----
fig_ld = figure('Position',[100 100 1200 800]);
cc = 0;
maxPC_plot = max(optPC_simca);
for ic = 1:nClasses
    for ipc = 1:optPC_simca(ic)
        cc = cc + 1;
        subplot(nClasses, maxPC_plot, (ic-1)*maxPC_plot + ipc);
        bar(simca_final.PCmodels{ic}.loads(:, ipc), 'FaceColor', classColors(ic,:));
        set(gca, 'XTick', 1:M_vars, 'XTickLabel', featNames, 'FontSize', 6);
        xtickangle(45);
        title(sprintf('C%d PC%d', ic, ipc), 'FontSize', 9);
        ylabel('Loading');
        grid on;
    end
end
try sgtitle('SIMCA: Loadings per Class', 'FontSize', 14); catch; annotation('textbox',[0.3 0.96 0.4 0.04],'String','SIMCA: Loadings per Class','FontSize',14,'FontWeight','bold','HorizontalAlignment','center','EdgeColor','none'); end
drawnow; pause(0.1);
saveas(fig_ld, fullfile(plotDir, '12_simca_loadings.png'));
close(fig_ld);

% ---- 4.11 SIMCA Classification Summary Table ----
fig_sum_s = figure('Position',[100 100 700 350]);
T_simca = table(classFullNames', optPC_simca, sens_tr_s, spec_tr_s, eff_tr_s, ...
    sens_ts_s, spec_ts_s, eff_ts_s, ...
    'VariableNames', {'Class','nPCs','Sens_Train','Spec_Train','Eff_Train',...
    'Sens_Test','Spec_Test','Eff_Test'});
uitable('Data', table2cell(T_simca), 'ColumnName', T_simca.Properties.VariableNames, ...
    'RowName', [], 'Position', [20 20 660 310], 'FontSize', 10);
annotation('textbox',[0.3 0.92 0.4 0.06],'String','SIMCA Summary','FontSize',14,...
    'FontWeight','bold','HorizontalAlignment','center','EdgeColor','none');
drawnow; pause(0.1);
saveas(fig_sum_s, fullfile(plotDir, '13_simca_summary_table.png'));
close(fig_sum_s);

fprintf('   SIMCA plots saved.\n\n');

%% ---- 5. PLS-DA ANALYSIS ------------------------------------------------
fprintf('[5] PLS-DA Analysis...\n');

% Create dummy Y matrix (one-hot encoding)
Y_train = zeros(size(X_train,1), nClasses);
for ic = 1:nClasses
    Y_train(y_train == ic, ic) = 1;
end
Y_test = zeros(size(X_test,1), nClasses);
for ic = 1:nClasses
    Y_test(y_test == ic, ic) = 1;
end

% Autoscale X
mx_tr = mean(X_train);
sx_tr = std(X_train);
X_train_sc = (X_train - mx_tr) ./ sx_tr;
X_test_sc  = (X_test  - mx_tr) ./ sx_tr;

% Mean center Y
my_tr = mean(Y_train);
Y_train_mc = Y_train - my_tr;

maxLV = min(M_vars, 6);  % max latent variables to test (bounded by number of variables)

% ---- 5.1 Cross-validation (venetian blinds, 5 segments) ----
fprintf('   Running PLS-DA cross-validation...\n');
nSegPLS = 5;
err_cv_pls = zeros(nClasses, maxLV);
corr_cv_pls = zeros(nClasses, maxLV);
misclass_cv_pls = zeros(1, maxLV);
rmsecv = zeros(nClasses, maxLV);

for nlv = 1:maxLV
    pred_cv_all = zeros(size(X_train,1), nClasses);
    
    for seg = 1:nSegPLS
        cv_ts_idx = (seg:nSegPLS:size(X_train,1))';
        cv_tr_idx = setdiff(1:size(X_train,1), cv_ts_idx)';
        
        Xcv_tr = X_train(cv_tr_idx, :);
        Ycv_tr = Y_train(cv_tr_idx, :);
        Xcv_ts = X_train(cv_ts_idx, :);
        
        % Scale with CV training stats
        mx_cv = mean(Xcv_tr);
        sx_cv = std(Xcv_tr);
        my_cv = mean(Ycv_tr);
        
        Xcv_tr_sc = (Xcv_tr - mx_cv) ./ sx_cv;
        Xcv_ts_sc = (Xcv_ts - mx_cv) ./ sx_cv;
        Ycv_tr_mc = Ycv_tr - my_cv;
        
        % Build PLS model
        plsmod = nipals_pls2(Xcv_tr_sc, Ycv_tr_mc, nlv);
        
        % Predict
        Ypred_cv = Xcv_ts_sc * plsmod.Bpls + my_cv;
        pred_cv_all(cv_ts_idx, :) = Ypred_cv;
    end
    
    % Assign classes based on max predicted Y
    [~, pred_class_cv] = max(pred_cv_all, [], 2);
    
    % Compute errors per class
    for ic = 1:nClasses
        idx_ic = y_train == ic;
        n_ic = sum(idx_ic);
        err_cv_pls(ic, nlv) = sum(pred_class_cv(idx_ic) ~= ic);
        corr_cv_pls(ic, nlv) = 100 * (n_ic - err_cv_pls(ic, nlv)) / n_ic;
        rmsecv(ic, nlv) = sqrt(mean((pred_cv_all(idx_ic, ic) - Y_train(idx_ic, ic)).^2));
    end
    misclass_cv_pls(nlv) = sum(pred_class_cv ~= y_train);
end

% ---- 5.2 Plot: CV errors and correct classification vs LVs ----
fig_plscv = figure('Position',[100 100 1000 700]);
subplot(2,2,1);
plot(1:maxLV, corr_cv_pls', '-o', 'LineWidth', 1.5);
title('PLS-DA CV: % Correct Classification');
xlabel('Latent Variables'); ylabel('% Correct');
legend(classFullNames, 'Location','best'); grid on;

subplot(2,2,3);
plot(1:maxLV, err_cv_pls', '-o', 'LineWidth', 1.5);
title('PLS-DA CV: Misclassified Samples');
xlabel('Latent Variables'); ylabel('# Misclassified');
legend(classFullNames, 'Location','best'); grid on;

subplot(2,2,2);
plot(1:maxLV, mean(corr_cv_pls)', '-o', 'LineWidth', 2, 'Color', [0.2 0.4 0.8]);
title('Mean % Correct (higher = better)');
xlabel('Latent Variables'); ylabel('Mean % Correct'); grid on;

subplot(2,2,4);
plot(1:maxLV, misclass_cv_pls, '-o', 'LineWidth', 2, 'Color', [0.8 0.2 0.2]);
title('Total Misclassified (lower = better)');
xlabel('Latent Variables'); ylabel('Total Misclassified'); grid on;
drawnow; pause(0.1);
saveas(fig_plscv, fullfile(plotDir, '14_plsda_cv_error.png'));
close(fig_plscv);

% ---- 5.2b Plot: RMSECV ----
fig_rmsecv = figure('Position',[100 100 800 400]);
plot(1:maxLV, rmsecv', '-o', 'LineWidth', 1.5);
title('PLS-DA: RMSECV per Class');
xlabel('Latent Variables'); ylabel('RMSECV');
legend(classFullNames, 'Location','best'); grid on;
drawnow; pause(0.1);
saveas(fig_rmsecv, fullfile(plotDir, '15_plsda_rmsecv.png'));
close(fig_rmsecv);

% ---- 5.3 Select optimal number of LVs ----
[~, optLV] = min(misclass_cv_pls);
% Ensure at least 2 LVs for visualization
if optLV < 2; optLV = 2; end
fprintf('   Optimal number of LVs: %d (CV misclassified: %d / %d)\n', ...
    optLV, misclass_cv_pls(optLV), size(X_train,1));

% ---- 5.4 Build final PLS-DA model ----
fprintf('   Building final PLS-DA model with %d LVs...\n', optLV);
plsda_final = nipals_pls2(X_train_sc, Y_train_mc, optLV);

% Predictions on training set
Ypred_train = X_train_sc * plsda_final.Bpls + my_tr;
[~, predclass_train_pls] = max(Ypred_train, [], 2);

% Predictions on test set
Ypred_test = X_test_sc * plsda_final.Bpls + my_tr;
[~, predclass_test_pls] = max(Ypred_test, [], 2);

% RMSEP
rmsep_pls = zeros(nClasses, 1);
for ic = 1:nClasses
    idx_ic = y_test == ic;
    rmsep_pls(ic) = sqrt(mean((Ypred_test(idx_ic, ic) - Y_test(idx_ic, ic)).^2));
end

% ---- 5.5 Display results ----
fprintf('\n   === PLS-DA RESULTS ===\n');

fprintf('   --- Training Set ---\n');
[sens_tr_p, spec_tr_p, eff_tr_p, acc_tr_p] = compute_class_metrics(y_train, predclass_train_pls, nClasses);
for ic = 1:nClasses
    fprintf('   Class %d (%s): Sens=%.3f Spec=%.3f Eff=%.3f\n', ...
        ic, classNames{ic}, sens_tr_p(ic), spec_tr_p(ic), eff_tr_p(ic));
end
fprintf('   Overall accuracy: %.1f%%\n', acc_tr_p*100);

fprintf('   --- Test Set ---\n');
[sens_ts_p, spec_ts_p, eff_ts_p, acc_ts_p] = compute_class_metrics(y_test, predclass_test_pls, nClasses);
for ic = 1:nClasses
    fprintf('   Class %d (%s): Sens=%.3f Spec=%.3f Eff=%.3f RMSEP=%.4f\n', ...
        ic, classNames{ic}, sens_ts_p(ic), spec_ts_p(ic), eff_ts_p(ic), rmsep_pls(ic));
end
fprintf('   Overall accuracy: %.1f%%\n\n', acc_ts_p*100);

% ---- 5.6 Plot: Y predicted vs samples (training) ----
fig_ytr = figure('Position',[100 100 1100 900]);
for ic = 1:nClasses
    subplot(nClasses, 1, ic);
    hold on;
    n_tr = size(X_train, 1);
    plot(1:n_tr, Ypred_train(:,ic), '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
    
    for jc = 1:nClasses
        idx = y_train == jc;
        if jc == ic
            scatter(find(idx), Ypred_train(idx,ic), 25, classColors(jc,:), 'filled');
        else
            scatter(find(idx), Ypred_train(idx,ic), 15, classColors(jc,:));
        end
    end
    xl_yp = xlim; plot(xl_yp, [0.5 0.5], '--r', 'LineWidth', 1);
    ylabel(sprintf('Y_{pred} C%d', ic));
    title(sprintf('Predicted Y for %s (Training)', classFullNames{ic}), 'FontSize', 9);
    grid on;
    if ic == nClasses; xlabel('Sample #'); end
end
try sgtitle(sprintf('PLS-DA | Preprocessing: Autoscaling | LVs = %d | Y: Mean-Centered Dummy', optLV), 'FontSize', 11, 'FontWeight','bold'); ...
catch; annotation('textbox',[0.05 0.96 0.9 0.04],'String',sprintf('PLS-DA | Preprocessing: Autoscaling | LVs = %d | Y: Mean-Centered Dummy', optLV),'FontSize',11,'FontWeight','bold','HorizontalAlignment','center','EdgeColor','none'); end
drawnow; pause(0.1);
saveas(fig_ytr, fullfile(plotDir, '16_plsda_ypred_train.png'));
close(fig_ytr);

% ---- 5.7 Plot: Y predicted vs samples (test) ----
fig_yts = figure('Position',[100 100 1100 900]);
for ic = 1:nClasses
    subplot(nClasses, 1, ic);
    hold on;
    n_ts = size(X_test, 1);
    plot(1:n_ts, Ypred_test(:,ic), '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
    
    for jc = 1:nClasses
        idx = y_test == jc;
        if jc == ic
            scatter(find(idx), Ypred_test(idx,ic), 25, classColors(jc,:), 'filled');
        else
            scatter(find(idx), Ypred_test(idx,ic), 15, classColors(jc,:));
        end
    end
    xl_yp2 = xlim; plot(xl_yp2, [0.5 0.5], '--r', 'LineWidth', 1);
    ylabel(sprintf('Y_{pred} C%d', ic));
    title(sprintf('Predicted Y for %s (Test)', classFullNames{ic}), 'FontSize', 9);
    grid on;
    if ic == nClasses; xlabel('Sample #'); end
end
try sgtitle(sprintf('PLS-DA | Preprocessing: Autoscaling | LVs = %d | Y: Mean-Centered Dummy', optLV), 'FontSize', 11, 'FontWeight','bold'); ...
catch; annotation('textbox',[0.05 0.96 0.9 0.04],'String',sprintf('PLS-DA | Preprocessing: Autoscaling | LVs = %d | Y: Mean-Centered Dummy', optLV),'FontSize',11,'FontWeight','bold','HorizontalAlignment','center','EdgeColor','none'); end
drawnow; pause(0.1);
saveas(fig_yts, fullfile(plotDir, '17_plsda_ypred_test.png'));
close(fig_yts);

% ---- 5.8 Plot: Confusion matrices (PLS-DA) ----
fig_cm2 = figure('Position',[100 100 1000 450]);
subplot(1,2,1);
plot_confusion_matrix(y_train, predclass_train_pls, classNames, 'PLS-DA - Training Set');
subplot(1,2,2);
plot_confusion_matrix(y_test, predclass_test_pls, classNames, 'PLS-DA - Test Set');
drawnow; pause(0.1);
saveas(fig_cm2, fullfile(plotDir, '18_plsda_confusion_matrices.png'));
close(fig_cm2);

% ---- 5.9 Plot: PLS Scores (LV1 vs LV2) ----
fig_plssc = figure('Position',[100 100 1000 450]);
subplot(1,2,1);
hold on;
for ic = 1:nClasses
    idx = y_train == ic;
    scatter(plsda_final.T(idx,1), plsda_final.T(idx,2), 40, classColors(ic,:), 'filled', 'MarkerEdgeColor','k','LineWidth',0.3);
end
xlabel('LV1 Score'); ylabel('LV2 Score');
title('PLS-DA Scores: Training Set');
legend(classFullNames, 'Location','best'); grid on;

% Project test data
T_test = X_test_sc * plsda_final.W * inv(plsda_final.P' * plsda_final.W);
subplot(1,2,2);
hold on;
for ic = 1:nClasses
    idx = y_test == ic;
    scatter(T_test(idx,1), T_test(idx,2), 40, classColors(ic,:), 'filled', 'MarkerEdgeColor','k','LineWidth',0.3);
end
xlabel('LV1 Score'); ylabel('LV2 Score');
title('PLS-DA Scores: Test Set');
legend(classFullNames, 'Location','best'); grid on;
drawnow; pause(0.1);
saveas(fig_plssc, fullfile(plotDir, '19_plsda_scores.png'));
close(fig_plssc);

% ---- 5.10 Plot: Regression coefficients ----
fig_bpls = figure('Position',[100 100 1000 500]);
for ic = 1:nClasses
    subplot(1, nClasses, ic);
    bar(plsda_final.Bpls(:, ic), 'FaceColor', classColors(ic,:));
    set(gca, 'XTick', 1:M_vars, 'XTickLabel', featNames, 'FontSize', 7);
    xtickangle(45);
    title(sprintf('B_{PLS} - %s', classNames{ic}), 'FontSize', 9);
    ylabel('Coefficient');
    grid on;
end
try sgtitle('PLS-DA Regression Coefficients', 'FontSize', 14); catch; annotation('textbox',[0.25 0.96 0.5 0.04],'String','PLS-DA Regression Coefficients','FontSize',14,'FontWeight','bold','HorizontalAlignment','center','EdgeColor','none'); end
drawnow; pause(0.1);
saveas(fig_bpls, fullfile(plotDir, '20_plsda_regression_coefficients.png'));
close(fig_bpls);

% ---- 5.11 VIP scores ----
VIP = compute_vip(plsda_final, X_train_sc, Y_train_mc);

fig_vip = figure('Position',[100 100 800 400]);
bar(VIP, 'FaceColor', [0.4 0.6 0.3]);
hold on;
xl_vip = xlim; plot(xl_vip, [1 1], '--r', 'LineWidth', 1.5); text(xl_vip(2), 1, 'VIP=1', 'Color','r');
plot(xl_vip, [0.8 0.8], ':k', 'LineWidth', 1); text(xl_vip(2), 0.8, 'VIP=0.8', 'Color','k');
set(gca, 'XTick', 1:M_vars, 'XTickLabel', featNames);
xtickangle(30);
xlabel('Variable'); ylabel('VIP Score');
title('PLS-DA: Variable Importance in Projection (VIP)');
grid on;
drawnow; pause(0.1);
saveas(fig_vip, fullfile(plotDir, '21_plsda_vip.png'));
close(fig_vip);

% ---- 5.12 PLS-DA Summary Table ----
fig_sum_p = figure('Position',[100 100 800 350]);
T_plsda = table(classFullNames', sens_tr_p, spec_tr_p, eff_tr_p, ...
    sens_ts_p, spec_ts_p, eff_ts_p, rmsep_pls, ...
    'VariableNames', {'Class','Sens_Train','Spec_Train','Eff_Train',...
    'Sens_Test','Spec_Test','Eff_Test','RMSEP'});
uitable('Data', table2cell(T_plsda), 'ColumnName', T_plsda.Properties.VariableNames, ...
    'RowName', [], 'Position', [20 20 760 310], 'FontSize', 10);
annotation('textbox',[0.2 0.92 0.6 0.06],'String',sprintf('PLS-DA Summary (%d LVs)',optLV),'FontSize',14,...
    'FontWeight','bold','HorizontalAlignment','center','EdgeColor','none');
drawnow; pause(0.1);
saveas(fig_sum_p, fullfile(plotDir, '22_plsda_summary_table.png'));
close(fig_sum_p);

fprintf('   PLS-DA plots saved.\n\n');

%% ---- 5b. APPLICABILITY DOMAIN ------------------------------------------
fprintf('[5b] Applicability Domain Analysis...\n');
fprintf('     Checking if test samples fall within the training domain.\n');

% Method: Leverage-based Applicability Domain (Hat matrix)
% The leverage h_i = x_i' * (X_train' * X_train)^{-1} * x_i measures
% how far each sample is from the training centroid in the model space.
% Warning threshold: h* = 3 * (p+1) / n_train (Williams plot convention)

% Using autoscaled data (same scaling as PLS-DA)
XtX_inv = inv(X_train_sc' * X_train_sc);

% Leverage for training samples
h_train = zeros(size(X_train_sc, 1), 1);
for i = 1:size(X_train_sc, 1)
    h_train(i) = X_train_sc(i,:) * XtX_inv * X_train_sc(i,:)';
end

% Leverage for test samples
h_test = zeros(size(X_test_sc, 1), 1);
for i = 1:size(X_test_sc, 1)
    h_test(i) = X_test_sc(i,:) * XtX_inv * X_test_sc(i,:)';
end

% Warning leverage threshold
h_star = 3 * (M_vars + 1) / size(X_train_sc, 1);

% Standardized residuals (PLS-DA) for Williams plot
Ypred_train_full = X_train_sc * plsda_final.Bpls + my_tr;
Ypred_test_full  = X_test_sc  * plsda_final.Bpls + my_tr;

% Use residual of assigned class
res_train = zeros(size(X_train_sc, 1), 1);
for i = 1:length(y_train)
    res_train(i) = Y_train(i, y_train(i)) - Ypred_train_full(i, y_train(i));
end
res_test = zeros(size(X_test_sc, 1), 1);
for i = 1:length(y_test)
    res_test(i) = Y_test(i, y_test(i)) - Ypred_test_full(i, y_test(i));
end

sigma_res = std(res_train);
std_res_train = res_train / sigma_res;
std_res_test  = res_test  / sigma_res;

n_test_outside = sum(h_test > h_star);
n_test_total = length(h_test);
fprintf('   Leverage threshold (h*): %.4f\n', h_star);
fprintf('   Test samples outside AD: %d / %d (%.1f%%)\n', ...
    n_test_outside, n_test_total, 100*n_test_outside/n_test_total);

% ---- Williams Plot (Leverage vs Standardized Residuals) ----
fig_ad = figure('Position',[100 100 1000 500]);
subplot(1,2,1);
hold on;
for ic = 1:nClasses
    idx_tr = y_train == ic;
    scatter(h_train(idx_tr), std_res_train(idx_tr), 30, classColors(ic,:), 'o', 'LineWidth', 1);
end
for ic = 1:nClasses
    idx_ts = y_test == ic;
    scatter(h_test(idx_ts), std_res_test(idx_ts), 50, classColors(ic,:), 'd', 'filled', 'MarkerEdgeColor', 'k');
end
xl_ad = xlim; yl_ad = ylim;
plot([h_star h_star], [-4 4], '--r', 'LineWidth', 1.5);
plot(xl_ad, [3 3], ':k', 'LineWidth', 1); plot(xl_ad, [-3 -3], ':k', 'LineWidth', 1);
text(h_star*1.02, 3.5, sprintf('h*=%.3f', h_star), 'Color', 'r', 'FontSize', 8);
xlabel('Leverage (h_i)'); ylabel('Standardized Residual');
title('Williams Plot - Applicability Domain');
legendEntries_ad = cell(1, 2*nClasses);
for jc = 1:nClasses
    legendEntries_ad{jc} = sprintf('%s (train)', classNames{jc});
    legendEntries_ad{nClasses+jc} = sprintf('%s (test)', classNames{jc});
end
legend(legendEntries_ad, 'Location','best', 'FontSize', 7);
grid on;

% ---- Leverage bar chart for test samples ----
subplot(1,2,2);
bar_colors = zeros(n_test_total, 3);
for i = 1:n_test_total
    bar_colors(i,:) = classColors(y_test(i), :);
end
bh = bar(h_test, 'FaceColor', 'flat');
bh.CData = bar_colors;
hold on;
plot(xlim, [h_star h_star], '--r', 'LineWidth', 1.5);
text(n_test_total*0.7, h_star*1.1, sprintf('h*=%.3f', h_star), 'Color', 'r', 'FontSize', 9);
xlabel('Test Sample Index'); ylabel('Leverage');
title(sprintf('Test Set Leverage (%d/%d outside AD)', n_test_outside, n_test_total));
grid on;

drawnow; pause(0.1);
saveas(fig_ad, fullfile(plotDir, '24_applicability_domain.png'));
close(fig_ad);

% ---- Hotelling T2 + Q for global PCA-based AD ----
% Build global PCA model on training set (autoscaled)
nPC_ad = optLV; % use same complexity as PLS-DA
[~, S_ad, V_ad] = svd(X_train_sc, 'econ');
P_ad = V_ad(:, 1:nPC_ad);
lambda_ad = diag(S_ad).^2 / (size(X_train_sc,1) - 1);

% Training T2 and Q
T_train_ad = X_train_sc * P_ad;
lam_diag = diag(lambda_ad(1:nPC_ad));
T2_train = diag(T_train_ad * inv(lam_diag) * T_train_ad');
E_train_ad = X_train_sc - T_train_ad * P_ad';
Q_train = sum(E_train_ad.^2, 2);

% Test T2 and Q
T_test_ad = X_test_sc * P_ad;
T2_test = diag(T_test_ad * inv(lam_diag) * T_test_ad');
E_test_ad = X_test_sc - T_test_ad * P_ad';
Q_test = sum(E_test_ad.^2, 2);

% Limits
N_ad = size(X_train_sc, 1);
A_ad = nPC_ad;
try F_ad = finv(0.95, A_ad, N_ad - A_ad); catch; z=sqrt(2)*erfinv(0.9); F_ad=z; end
T2_lim = A_ad * (N_ad - 1) / (N_ad - A_ad) * F_ad;
if length(lambda_ad) > nPC_ad
    th1 = sum(lambda_ad(nPC_ad+1:end)); th2 = sum(lambda_ad(nPC_ad+1:end).^2); th3 = sum(lambda_ad(nPC_ad+1:end).^3);
    h0_q = 1 - 2*th1*th3/(3*th2^2); if h0_q < 0.001; h0_q = 0.001; end
    ca_q = sqrt(2)*erfinv(0.9);
    Q_lim = th1*(1 + ca_q*sqrt(2*th2*h0_q^2)/th1 + th2*h0_q*(h0_q-1)/th1^2)^(1/h0_q);
else
    Q_lim = max(Q_train)*1.5;
end

fig_ad2 = figure('Position',[100 100 800 600]);
hold on;
for ic = 1:nClasses
    idx_tr = y_train == ic;
    scatter(T2_train(idx_tr)/T2_lim, Q_train(idx_tr)/Q_lim, 30, classColors(ic,:), 'o', 'LineWidth', 1);
end
for ic = 1:nClasses
    idx_ts = y_test == ic;
    scatter(T2_test(idx_ts)/T2_lim, Q_test(idx_ts)/Q_lim, 60, classColors(ic,:), 'd', 'filled', 'MarkerEdgeColor','k');
end
% AD boundary: ellipse at 1,1
theta_el = 0:0.01:2*pi;
plot(cos(theta_el), sin(theta_el), '-r', 'LineWidth', 1.5);
plot([1 1], ylim, ':r', 'LineWidth', 1);
plot(xlim, [1 1], ':r', 'LineWidth', 1);
xlabel('T^2 / T^2_{lim}'); ylabel('Q / Q_{lim}');
title(sprintf('Applicability Domain: T^2 vs Q (PCA %d PCs)', nPC_ad));
legendEntries_ad2 = cell(1, 2*nClasses);
for jc = 1:nClasses
    legendEntries_ad2{jc} = sprintf('%s (train)', classNames{jc});
    legendEntries_ad2{nClasses+jc} = sprintf('%s (test)', classNames{jc});
end
legend(legendEntries_ad2, 'Location','best', 'FontSize', 7);
grid on;
drawnow; pause(0.1);
saveas(fig_ad2, fullfile(plotDir, '25_applicability_domain_T2Q.png'));
close(fig_ad2);

% Count test samples outside T2/Q domain
n_outside_T2Q = sum(T2_test/T2_lim > 1 | Q_test/Q_lim > 1);
fprintf('   Test samples outside T2/Q domain: %d / %d (%.1f%%)\n', ...
    n_outside_T2Q, n_test_total, 100*n_outside_T2Q/n_test_total);
fprintf('   Applicability Domain plots saved.\n\n');

%% ---- 6. COMPARISON SUMMARY ---------------------------------------------
fprintf('[6] Final Comparison...\n\n');
fprintf('   %-25s  SIMCA     PLS-DA\n', '');
fprintf('   %-25s  --------  --------\n', '');
fprintf('   %-25s  %.1f%%     %.1f%%\n', 'Train Accuracy', acc_tr_s*100, acc_tr_p*100);
fprintf('   %-25s  %.1f%%     %.1f%%\n', 'Test Accuracy', acc_ts_s*100, acc_ts_p*100);
fprintf('   %-25s  ', 'Complexity');
fprintf('%s  ', sprintf('%dPC', optPC_simca));
fprintf('  %dLV\n', optLV);

% Final comparison figure
fig_comp = figure('Position',[100 100 900 500]);
subplot(1,2,1);
bar_data_sens = [sens_ts_s, sens_ts_p];
b = bar(bar_data_sens);
b(1).FaceColor = [0.3 0.5 0.8]; b(2).FaceColor = [0.8 0.4 0.3];
set(gca, 'XTickLabel', classNames);
ylabel('Sensitivity'); title('Test Set Sensitivity');
legend({'SIMCA','PLS-DA'}, 'Location','best'); grid on;
ylim([0 1.1]);

subplot(1,2,2);
bar_data_spec = [spec_ts_s, spec_ts_p];
b2 = bar(bar_data_spec);
b2(1).FaceColor = [0.3 0.5 0.8]; b2(2).FaceColor = [0.8 0.4 0.3];
set(gca, 'XTickLabel', classNames);
ylabel('Specificity'); title('Test Set Specificity');
legend({'SIMCA','PLS-DA'}, 'Location','best'); grid on;
ylim([0 1.1]);
try sgtitle('SIMCA vs PLS-DA Comparison', 'FontSize', 14); catch; annotation('textbox',[0.25 0.96 0.5 0.04],'String','SIMCA vs PLS-DA Comparison','FontSize',14,'FontWeight','bold','HorizontalAlignment','center','EdgeColor','none'); end
drawnow; pause(0.1);
saveas(fig_comp, fullfile(plotDir, '23_comparison_simca_plsda.png'));
close(fig_comp);

% Save workspace variables
save(fullfile(plotDir, 'analysis_results.mat'), ...
    'X_train','X_test','y_train','y_test', ...
    'simca_final','pred_train_simca','pred_test_simca', ...
    'plsda_final','Ypred_train','Ypred_test', ...
    'optPC_simca','optLV', ...
    'sens_tr_s','spec_tr_s','eff_tr_s','acc_tr_s', ...
    'sens_ts_s','spec_ts_s','eff_ts_s','acc_ts_s', ...
    'sens_tr_p','spec_tr_p','eff_tr_p','acc_tr_p', ...
    'sens_ts_p','spec_ts_p','eff_ts_p','acc_ts_p', ...
    'classNames','classFullNames','featNames', ...
    'explainedVar','dpow','VIP','rmsecv','rmsep_pls', ...
    'nameobj_mosti','annata', ...
    'h_train','h_test','h_star', ...
    'T2_train','T2_test','T2_lim', ...
    'Q_train','Q_test','Q_lim', ...
    'std_res_train','std_res_test');

fprintf('\n============================================================\n');
fprintf('  ANALYSIS COMPLETE\n');
fprintf('  All plots saved in: %s\n', plotDir);
fprintf('  Results saved in: analysis_results.mat\n');
fprintf('============================================================\n');


%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================

function model = build_simca_model(X, y, ncomp_vec, cl)
% Build SIMCA models (one PCA per class) without PLS Toolbox dependency
    nClasses = max(y);
    [nsamp, nvar] = size(X);
    
    if length(ncomp_vec) == 1
        ncomp_vec = ncomp_vec * ones(nClasses, 1);
    end
    
    sensspec = zeros(nClasses, nClasses);
    
    for ic = 1:nClasses
        cl_ind = find(y == ic);
        Xd1 = X(cl_ind, :);
        Xd2 = X;
        
        mx_c = mean(Xd1, 'omitnan');
        sx_c = std(Xd1, 0, 1, 'omitnan');
        sx_c(sx_c == 0) = 1;
        
        model.PCmodels{ic}.prepr = {mx_c, sx_c};
        
        Xd1_sc = (Xd1 - mx_c) ./ sx_c;
        Xd2_sc = (Xd2 - mx_c) ./ sx_c;
        
        nc = min(ncomp_vec(ic), min(size(Xd1_sc))-1);
        if nc < 1; nc = 1; end
        ncomp_vec(ic) = nc;
        
        [u, s, v] = svd(Xd1_sc, 'econ');
        T_class = u(:, 1:nc) * s(1:nc, 1:nc);
        P_class = v(:, 1:nc);
        
        lambda_diag = diag(s).^2 / (size(Xd1,1) - 1);
        eigs_all = lambda_diag;
        
        tot_var = sum(lambda_diag);
        ev = 100 * lambda_diag / tot_var;
        cv_var = cumsum(ev);
        
        T1 = Xd2_sc * P_class;
        lambda_mat = diag(lambda_diag(1:nc));
        
        tsq = diag(T1 * inv(lambda_mat) * T1');
        
        Xd2_recon = T1 * P_class';
        E = Xd2_sc - Xd2_recon;
        q = sum(E.^2, 2);
        
        N_class = size(Xd1, 1);
        A = nc;
        try
            F_crit = finv(cl, A, N_class - A);
        catch
            z = sqrt(2) * erfinv(2*cl - 1);
            chi2_approx = A * (1 - 2/(9*A) + z*sqrt(2/(9*A)))^3;
            F_crit = chi2_approx / A;
        end
        t2lim = A * (N_class - 1) / (N_class - A) * F_crit;
        
        if length(eigs_all) > nc
            theta1 = sum(eigs_all(nc+1:end));
            theta2 = sum(eigs_all(nc+1:end).^2);
            theta3 = sum(eigs_all(nc+1:end).^3);
        else
            theta1 = 0; theta2 = 0; theta3 = 0;
        end
        
        if theta1 == 0
            qlim = 0;
        else
            h0 = 1 - 2*theta1*theta3 / (3*theta2^2);
            if h0 < 0.001; h0 = 0.001; end
            ca = sqrt(2) * erfinv(2*cl - 1);
            h1 = ca * sqrt(2*theta2*h0^2) / theta1;
            h2 = theta2*h0*(h0-1) / theta1^2;
            qlim = theta1 * (1 + h1 + h2)^(1/h0);
        end
        
        cr_i = zeros(nsamp, 1);
        cl_acc = zeros(nsamp, 1);
        
        for j = 1:nsamp
            if qlim > 0 && t2lim > 0
                cr_i(j) = sqrt((tsq(j)/t2lim)^2 + (q(j)/qlim)^2);
            else
                cr_i(j) = tsq(j) / max(t2lim, eps);
            end
            
            if cr_i(j) <= sqrt(2)
                cl_acc(j) = 1;
                if y(j) == ic
                    sensspec(ic, y(j)) = sensspec(ic, y(j)) + 1;
                end
            else
                if y(j) ~= ic
                    sensspec(ic, y(j)) = sensspec(ic, y(j)) + 1;
                end
            end
        end
        
        model.PCmodels{ic}.res = E;
        model.PCmodels{ic}.scores = T1;
        model.PCmodels{ic}.loads = P_class;
        model.PCmodels{ic}.eigs = eigs_all;
        model.PCmodels{ic}.variance = {ev, cv_var};
        model.PCmodels{ic}.critvals = [t2lim, qlim];
        model.PCmodels{ic}.tsq = tsq;
        model.PCmodels{ic}.qres = q;
        model.SIMCA.accept{ic} = cl_acc;
        model.SIMCA.crit(:, ic) = cr_i;
    end
    
    model.SIMCA.sensspec = sensspec;
    model.SIMCA.ncomp = ncomp_vec(:)';
    model.SIMCA.nclass = nClasses;
    model.SIMCA.cl = cl;
    
    [~, final_cl] = min(model.SIMCA.crit, [], 2);
    model.SIMCA.predclass = final_cl;
end


function pred = predict_simca(Xpred, ypred, simcamod)
% Predict class membership using a trained SIMCA model
    nClasses = simcamod.SIMCA.nclass;
    ncomp = simcamod.SIMCA.ncomp;
    nspred = size(Xpred, 1);
    
    sensspec = zeros(nClasses, nClasses);
    has_labels = ~isempty(ypred);
    
    for ic = 1:nClasses
        mx_c = simcamod.PCmodels{ic}.prepr{1};
        sx_c = simcamod.PCmodels{ic}.prepr{2};
        P = simcamod.PCmodels{ic}.loads;
        eigs_all = simcamod.PCmodels{ic}.eigs;
        
        Xp_sc = (Xpred - mx_c) ./ sx_c;
        
        nc = ncomp(ic);
        Tp = Xp_sc * P;
        lambda_mat = diag(eigs_all(1:nc));
        
        tsq = diag(Tp * inv(lambda_mat) * Tp');
        E = Xp_sc - Tp * P';
        q = sum(E.^2, 2);
        
        t2lim = simcamod.PCmodels{ic}.critvals(1);
        qlim = simcamod.PCmodels{ic}.critvals(2);
        
        cr_i = zeros(nspred, 1);
        cl_acc = zeros(nspred, 1);
        
        for j = 1:nspred
            if qlim > 0 && t2lim > 0
                cr_i(j) = sqrt((tsq(j)/t2lim)^2 + (q(j)/qlim)^2);
            else
                cr_i(j) = tsq(j) / max(t2lim, eps);
            end
            
            if cr_i(j) <= sqrt(2)
                cl_acc(j) = 1;
                if has_labels && ypred(j) == ic
                    sensspec(ic, ypred(j)) = sensspec(ic, ypred(j)) + 1;
                end
            else
                if has_labels && ypred(j) ~= ic
                    sensspec(ic, ypred(j)) = sensspec(ic, ypred(j)) + 1;
                end
            end
        end
        
        pred.PCmodels{ic}.scores = Tp;
        pred.PCmodels{ic}.loads = P;
        pred.PCmodels{ic}.critvals = [t2lim, qlim];
        pred.PCmodels{ic}.tsq = tsq;
        pred.PCmodels{ic}.qres = q;
        pred.SIMCA.accept{ic} = cl_acc;
        pred.SIMCA.crit(:, ic) = cr_i;
    end
    
    if has_labels
        pred.SIMCA.sensspec = sensspec;
    end
    pred.SIMCA.ncomp = ncomp;
    pred.SIMCA.nclass = nClasses;
    
    [~, final_cl] = min(pred.SIMCA.crit, [], 2);
    pred.SIMCA.predclass = final_cl;
end


function model = nipals_pls2(X, Y, ncomp)
% NIPALS PLS2 algorithm
    [n, p] = size(X);
    [~, q] = size(Y);
    
    T = zeros(n, ncomp);
    P = zeros(p, ncomp);
    W = zeros(p, ncomp);
    Q = zeros(q, ncomp);
    bvec = zeros(ncomp, 1);
    
    E = X;
    F = Y;
    
    for a = 1:ncomp
        [~, maxcol] = max(sum(F.^2));
        u = F(:, maxcol);
        
        for iter = 1:500
            w = E' * u;
            w = w / norm(w);
            t = E * w;
            qq = F' * t / (t' * t);
            
            if q == 1
                u = F * qq / (qq' * qq);
                break;
            end
            
            u_new = F * qq / (qq' * qq);
            
            if norm(u_new - u) / (norm(u_new) + eps) < 1e-12
                u = u_new;
                break;
            end
            u = u_new;
        end
        
        pp = E' * t / (t' * t);
        b = u' * t / (t' * t);
        
        T(:, a) = t;
        P(:, a) = pp;
        W(:, a) = w;
        Q(:, a) = qq;
        bvec(a) = b;
        
        E = E - t * pp';
        F = F - b * t * qq';
    end
    
    model.T = T;
    model.P = P;
    model.W = W;
    model.Q = Q;
    model.B = bvec;
    model.Bpls = W * inv(P' * W) * diag(bvec) * Q';
end


function VIP = compute_vip(plsmodel, X, Y)
% Compute Variable Importance in Projection (VIP) scores
    W = plsmodel.W;
    T = plsmodel.T;
    Q = plsmodel.Q;
    
    [p, ncomp] = size(W);
    
    SS = zeros(ncomp, 1);
    for a = 1:ncomp
        b = plsmodel.B(a);
        SS(a) = b^2 * (T(:,a)' * T(:,a)) * (Q(:,a)' * Q(:,a));
    end
    SStotal = sum(SS);
    
    VIP = zeros(p, 1);
    for j = 1:p
        s = 0;
        for a = 1:ncomp
            s = s + SS(a) * (W(j,a) / norm(W(:,a)))^2;
        end
        VIP(j) = sqrt(p * s / SStotal);
    end
end


function dpow = compute_discriminant_power(X, y, simcamod)
% Calculate SIMCA discriminant power for each variable
    nClasses = simcamod.SIMCA.nclass;
    M = size(X, 2);
    s2in = zeros(1, M);
    s2not = zeros(1, M);
    
    for ic = 1:nClasses
        iin = find(y == ic);
        inot = find(y ~= ic);
        res = simcamod.PCmodels{ic}.res;
        q_res = res.^2;
        A = simcamod.SIMCA.ncomp(ic);
        
        s2in = s2in + (M/(M-A)) * sum(q_res(iin,:), 1) / length(iin);
        s2not = s2not + (M/(M-A)) * sum(q_res(inot,:), 1) / length(inot);
    end
    
    dpow = sqrt(s2not ./ max(s2in, eps)) - 1;
    dpow = max(dpow, 0);
end


function [sens, spec, eff, acc] = compute_class_metrics(y_true, y_pred, nClasses)
% Compute sensitivity, specificity, efficiency for each class
    sens = zeros(nClasses, 1);
    spec = zeros(nClasses, 1);
    
    for ic = 1:nClasses
        TP = sum(y_true == ic & y_pred == ic);
        FN = sum(y_true == ic & y_pred ~= ic);
        TN = sum(y_true ~= ic & y_pred ~= ic);
        FP = sum(y_true ~= ic & y_pred == ic);
        
        sens(ic) = TP / max(TP + FN, 1);
        spec(ic) = TN / max(TN + FP, 1);
    end
    
    eff = sqrt(sens .* spec);
    acc = sum(y_true == y_pred) / length(y_true);
end


function plot_confusion_matrix(y_true, y_pred, classLabels, titleStr)
% Plot a professional confusion matrix
    nClasses = length(classLabels);
    cm = zeros(nClasses);
    
    for i = 1:nClasses
        for j = 1:nClasses
            cm(i, j) = sum(y_true == i & y_pred == j);
        end
    end
    
    imagesc(cm);
    colormap(flipud(bone));
    colorbar;
    
    total_per_row = sum(cm, 2);
    for i = 1:nClasses
        for j = 1:nClasses
            pct = 100 * cm(i,j) / max(total_per_row(i), 1);
            if cm(i,j) > 0
                if i == j
                    textColor = [0 0.5 0];
                else
                    textColor = [0.8 0 0];
                end
                text(j, i, sprintf('%d\n(%.0f%%)', cm(i,j), pct), ...
                    'HorizontalAlignment','center', 'FontSize', 9, ...
                    'FontWeight', 'bold', 'Color', textColor);
            else
                text(j, i, '0', 'HorizontalAlignment','center', ...
                    'FontSize', 9, 'Color', [0.5 0.5 0.5]);
            end
        end
    end
    
    set(gca, 'XTick', 1:nClasses, 'XTickLabel', classLabels, ...
             'YTick', 1:nClasses, 'YTickLabel', classLabels);
    xlabel('Predicted Class'); ylabel('True Class');
    title(titleStr);
    
    acc = 100 * trace(cm) / sum(cm(:));
    text(0.5, nClasses + 0.6, sprintf('Accuracy: %.1f%%', acc), 'FontSize', 10, 'FontWeight', 'bold');
end
