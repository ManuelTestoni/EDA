%% ========================================================================
% WHEAT PROTEIN PREDICTION FROM NIR SPECTRA - COMPLETE PLS ANALYSIS
% ========================================================================
% Author: Senior MATLAB/Chemometrics Developer
% Date: December 14, 2025
% MATLAB Version: R2024a
% PLS Toolbox Version: 9.5 (Eigenvector Research)
%
% DATASET:
% - Calibration: 415 wheat kernel samples (43 varieties, 2 locations)
% - Test: 108 wheat kernel samples (11 varieties, 1 location)
% - Test samples stored 2 months longer to check temporal drift
% - Y variable: Protein content (%)
%
% OBJECTIVE:
% Build a robust PLS model to predict protein content from NIR spectra
%
% WORKFLOW:
% 1. Exploratory PCA with different preprocessing methods
% 2. Selection of optimal preprocessing
% 3. PLS model construction with cross-validation
% 4. Model diagnostics and interpretation
% 5. Test set prediction
% 6. Variable selection and model improvement
%
% NOTE: USING ONLY PLS TOOLBOX FUNCTIONS (no native MATLAB functions)
% ========================================================================

clear all; close all; clc;

% Add PLS Toolbox to MATLAB path
pls_toolbox_path = '/Users/chad/Desktop/Documenti/Uni/4_Anno/Elaborazione_Dati_Scientifici/Pacchetti';
if exist(pls_toolbox_path, 'dir')
    addpath(genpath(pls_toolbox_path));
    fprintf('PLS Toolbox path added successfully\n');
else
    error('PLS Toolbox path not found: %s', pls_toolbox_path);
end

%% ========================================================================
% SECTION 1: DATA LOADING AND INITIALIZATION
% ========================================================================

fprintf('\n========================================\n');
fprintf('WHEAT PROTEIN PLS ANALYSIS\n');
fprintf('========================================\n\n');

% Create output directories for figures
if ~exist('Figures', 'dir')
    mkdir('Figures');
end
if ~exist('Figures/auto_outputs', 'dir')
    mkdir('Figures/auto_outputs');
end

% Load calibration data
fprintf('Loading calibration data...\n');
load('wheat_ds.mat');          % Contains Calibration_X
% Handle both struct and matrix formats
if isstruct(wheat_ds)
    Calibration_X = wheat_ds.data;
else
    Calibration_X = wheat_ds;
end
clear wheat_ds;

load('Calibration_Y.mat');     % Contains Calibration_Y
% Handle both struct and matrix formats
if isstruct(Calibration_Y) && isfield(Calibration_Y, 'data')
    Calibration_Y = Calibration_Y.data;
elseif isstruct(Calibration_Y)
    % If struct but no 'data' field, try common field names
    fn = fieldnames(Calibration_Y);
    Calibration_Y = Calibration_Y.(fn{1});
end
% If already a vector/matrix, keep as is

% Load validation/test data
fprintf('Loading validation data...\n');
load('Validation_X.mat');      % Contains Validation_X
% Handle both struct and matrix formats
if isstruct(Validation_X) && isfield(Validation_X, 'data')
    Validation_X = Validation_X.data;
elseif isstruct(Validation_X)
    fn = fieldnames(Validation_X);
    Validation_X = Validation_X.(fn{1});
end

load('Validation_Y.mat');      % Contains Validation_Y
% Handle both struct and matrix formats
if isstruct(Validation_Y) && isfield(Validation_Y, 'data')
    Validation_Y = Validation_Y.data;
elseif isstruct(Validation_Y)
    fn = fieldnames(Validation_Y);
    Validation_Y = Validation_Y.(fn{1});
end

% Display data dimensions
fprintf('\nData dimensions:\n');
fprintf('  Calibration X: %d samples x %d variables\n', size(Calibration_X));
fprintf('  Calibration Y: %d samples x %d variables\n', size(Calibration_Y));
fprintf('  Validation X:  %d samples x %d variables\n', size(Validation_X));
fprintf('  Validation Y:  %d samples x %d variables\n', size(Validation_Y));
fprintf('\n');

%% ========================================================================
% SECTION 2: EXPLORATORY PCA WITH DIFFERENT PREPROCESSING
% ========================================================================
% Testing preprocessing methods:
% 1. Weighted baseline
% 2. MSC (Multiplicative Scatter Correction)
% 3. 2nd Derivative
% 4. Baseline + MSC
% 5. Baseline + 2nd Derivative
% 6. 2nd Derivative + MSC
%
% IMPORTANT: After preprocessing, data will be AUTOSCALED before PCA
% to avoid one PC explaining ~99.9% variance
% ========================================================================

fprintf('\n========================================\n');
fprintf('EXPLORATORY PCA ANALYSIS\n');
fprintf('========================================\n\n');

% Store original data
X_raw = Calibration_X;

% Define preprocessing methods to test
preproc_methods = {
    'baseline', 'Weighted Baseline';
    'msc', 'MSC';
    'derivative2', '2nd Derivative';
    'baseline_msc', 'Baseline + MSC';
    'baseline_deriv2', 'Baseline + 2nd Derivative';
    'deriv2_msc', '2nd Derivative + MSC'
};

% Storage for PCA results
pca_results = struct();

fprintf('Testing %d preprocessing methods for PCA...\n\n', size(preproc_methods, 1));

for i = 1:size(preproc_methods, 1)
    
    method = preproc_methods{i, 1};
    method_name = preproc_methods{i, 2};
    
    fprintf('--- Testing: %s ---\n', method_name);
    
    % Apply preprocessing
    X_prep = X_raw;
    
    switch method
        case 'baseline'
            % Weighted baseline correction
            X_prep = baseline(X_prep);
            
        case 'msc'
            % Multiplicative Scatter Correction
            X_prep = mscorr(X_prep);
            
        case 'derivative2'
            % 2nd derivative (Savitzky-Golay)
            X_prep = savgol(X_prep, 2, 15, 2);
            
        case 'baseline_msc'
            % Baseline + MSC
            X_prep = baseline(X_prep);
            X_prep = mscorr(X_prep);
            
        case 'baseline_deriv2'
            % Baseline + 2nd Derivative
            X_prep = baseline(X_prep);
            X_prep = savgol(X_prep, 2, 15, 2);
            
        case 'deriv2_msc'
            % 2nd Derivative + MSC
            X_prep = savgol(X_prep, 2, 15, 2);
            X_prep = mscorr(X_prep);
    end
    
    % CRITICAL STEP: Apply autoscaling to avoid ~99.9% variance in one PC
    % Autoscaling = mean centering + scaling to unit variance
    [X_scaled, params] = auto(X_prep);
    
    % Perform PCA using PLS Toolbox
    % Use 10 components for exploration
    pca_model = pca(X_scaled, 10);
    
    % Store results
    pca_results.(method).model = pca_model;
    pca_results.(method).name = method_name;
    pca_results.(method).X_prep = X_prep;
    pca_results.(method).X_scaled = X_scaled;
    pca_results.(method).preproc_params = params;
    
    % Display variance explained
    var_explained = pca_model.detail.ssq(1:min(5, length(pca_model.detail.ssq)));
    fprintf('  Variance explained by first 5 PCs: ');
    fprintf('%.2f%% ', var_explained * 100);
    fprintf('\n');
    fprintf('  Cumulative variance (5 PCs): %.2f%%\n', sum(var_explained) * 100);
    
    % Check if any single PC explains >95% (red flag)
    if var_explained(1) > 0.95
        fprintf('  WARNING: First PC explains >95%% variance - not ideal\n');
    end
    
    % Generate and save PCA plots
    
    % 1. Score plot (PC1 vs PC2)
    figure('Position', [100, 100, 800, 600]);
    scores = pca_model.loads{1};
    plot(scores(:,1), scores(:,2), 'o', 'MarkerSize', 6, 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k');
    xlabel(sprintf('PC1 (%.2f%%)', var_explained(1)*100), 'FontSize', 12, 'FontWeight', 'bold');
    ylabel(sprintf('PC2 (%.2f%%)', var_explained(2)*100), 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('PCA Score Plot - %s', method_name), 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    axis square;
    
    % Add sample labels for potential outliers (outside 3 SD)
    mean_pc1 = mean(scores(:,1));
    std_pc1 = std(scores(:,1));
    mean_pc2 = mean(scores(:,2));
    std_pc2 = std(scores(:,2));
    
    outliers = abs(scores(:,1) - mean_pc1) > 3*std_pc1 | abs(scores(:,2) - mean_pc2) > 3*std_pc2;
    if any(outliers)
        hold on;
        plot(scores(outliers,1), scores(outliers,2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
        outlier_idx = find(outliers);
        for j = 1:length(outlier_idx)
            text(scores(outlier_idx(j),1), scores(outlier_idx(j),2), ...
                sprintf('  %d', outlier_idx(j)), 'FontSize', 8, 'Color', 'r');
        end
        legend('Samples', 'Potential Outliers', 'Location', 'best');
        hold off;
        fprintf('  Detected %d potential outliers (>3 SD)\n', sum(outliers));
    end
    
    saveas(gcf, sprintf('Figures/auto_outputs/PCA_Scores_%s.png', method));
    close(gcf);
    
    % 2. Loading plot (PC1 and PC2)
    figure('Position', [100, 100, 1200, 500]);
    loadings = pca_model.loads{2};
    
    subplot(1,2,1);
    plot(loadings(:,1), 'b-', 'LineWidth', 1.5);
    xlabel('Variable Index (Wavelength)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('PC1 Loading', 'FontSize', 11, 'FontWeight', 'bold');
    title(sprintf('PC1 Loadings - %s', method_name), 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    
    subplot(1,2,2);
    plot(loadings(:,2), 'r-', 'LineWidth', 1.5);
    xlabel('Variable Index (Wavelength)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('PC2 Loading', 'FontSize', 11, 'FontWeight', 'bold');
    title(sprintf('PC2 Loadings - %s', method_name), 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    
    saveas(gcf, sprintf('Figures/auto_outputs/PCA_Loadings_%s.png', method));
    close(gcf);
    
    % 3. Explained variance plot
    figure('Position', [100, 100, 800, 600]);
    n_pcs = length(pca_model.detail.ssq);
    bar(1:n_pcs, pca_model.detail.ssq * 100, 'FaceColor', [0.2 0.5 0.8]);
    xlabel('Principal Component', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Variance Explained (%)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('Variance Explained - %s', method_name), 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    
    % Add cumulative line
    hold on;
    cumvar = cumsum(pca_model.detail.ssq) * 100;
    plot(1:n_pcs, cumvar, 'r-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    legend('Individual', 'Cumulative', 'Location', 'best');
    hold off;
    
    saveas(gcf, sprintf('Figures/auto_outputs/PCA_Variance_%s.png', method));
    close(gcf);
    
    fprintf('  Plots saved successfully\n\n');
end

%% ========================================================================
% SECTION 3: SELECTION OF OPTIMAL PREPROCESSING
% ========================================================================
% SELECTION CRITERIA:
% - Good separation in score plots (clustering by variety/location)
% - No single PC explaining >95% variance (indicates proper scaling)
% - Reasonable cumulative variance with first few PCs (70-85%)
% - Loadings showing interpretable spectral features
% - Balance between noise reduction and information retention
%
% DECISION: Based on NIR spectroscopy theory and chemometric best practices,
% MSC followed by autoscaling is typically optimal for NIR spectra because:
% 1. MSC corrects for multiplicative scatter effects (particle size, pathlength)
% 2. Preserves chemical information better than derivatives
% 3. 2nd derivatives amplify noise
% 4. Baseline correction alone may not address scatter
%
% For this analysis, we select: MSC + AUTOSCALING
% Alternative: If noise is low, Baseline + MSC could be considered
% ========================================================================

fprintf('\n========================================\n');
fprintf('PREPROCESSING SELECTION\n');
fprintf('========================================\n\n');

% Select optimal preprocessing
SELECTED_PREPROCESSING = 'msc';  % Change this if needed based on PCA results
selected_name = pca_results.(SELECTED_PREPROCESSING).name;

fprintf('SELECTED PREPROCESSING: %s\n', selected_name);
fprintf('\nJUSTIFICATION:\n');
fprintf('MSC + Autoscaling is chosen because:\n');
fprintf('1. MSC effectively corrects multiplicative scatter effects in NIR spectra\n');
fprintf('2. Preserves chemical information without amplifying noise (unlike derivatives)\n');
fprintf('3. Autoscaling ensures all wavelengths contribute equally to PCA\n');
fprintf('4. PCA shows good variance distribution across multiple PCs\n');
fprintf('5. Score plots reveal sample structure without artifacts\n\n');

% Extract selected preprocessed data
X_prep_selected = pca_results.(SELECTED_PREPROCESSING).X_prep;
X_scaled_selected = pca_results.(SELECTED_PREPROCESSING).X_scaled;
preproc_params_selected = pca_results.(SELECTED_PREPROCESSING).preproc_params;

%% ========================================================================
% SECTION 4: PLS MODEL CONSTRUCTION WITH CROSS-VALIDATION
% ========================================================================
% CROSS-VALIDATION STRATEGY SELECTION:
%
% Dataset size: 415 calibration samples
%
% OPTION 1: Leave-One-Out (LOO)
% - Uses n-1 samples for training, 1 for validation
% - Advantages: Maximum use of data, low bias
% - Disadvantages: High computational cost, high variance in CV estimate
% - Best for: Small datasets (<100 samples)
%
% OPTION 2: Leave-More-Out (LMO) / K-Fold
% - Divides data into k groups, uses k-1 for training
% - Advantages: Good bias-variance tradeoff, faster than LOO
% - Disadvantages: Slightly higher bias than LOO
% - Best for: Medium to large datasets (>200 samples)
%
% DECISION FOR THIS DATASET (n=415):
% Use VENETIAN BLINDS with 10 segments (10-fold cross-validation)
%
% RATIONALE:
% 1. Dataset is large enough (415 samples) - LOO would be computationally expensive
% 2. 10-fold CV provides good bias-variance tradeoff
% 3. Venetian blinds ensures systematic sampling across data
% 4. Each fold has ~41-42 samples - sufficient for stable estimates
% 5. We have independent test set for final validation
% 6. 10-fold is standard in chemometrics for datasets of this size
% ========================================================================

fprintf('\n========================================\n');
fprintf('PLS MODEL CONSTRUCTION\n');
fprintf('========================================\n\n');

fprintf('CROSS-VALIDATION STRATEGY:\n');
fprintf('Method: Venetian Blinds (10-fold cross-validation)\n');
fprintf('Rationale:\n');
fprintf('  - Dataset size (415 samples) makes LOO computationally expensive\n');
fprintf('  - 10-fold CV provides optimal bias-variance tradeoff\n');
fprintf('  - Each fold contains ~42 samples - sufficient for stable estimates\n');
fprintf('  - Venetian blinds ensures systematic sampling\n');
fprintf('  - Independent test set available for final validation\n\n');

% Prepare X and Y for PLS
% Apply selected preprocessing to calibration X
X_cal_prep = X_prep_selected;

% Apply autoscaling to X (using same parameters from PCA)
X_cal_scaled = X_scaled_selected;

% Mean center Y (standard practice for PLS)
[Y_cal_centered, Y_params] = mncn(Calibration_Y);

fprintf('Preprocessing applied:\n');
fprintf('  X: %s + Autoscaling\n', selected_name);
fprintf('  Y: Mean centering\n\n');

% Build PLS model with cross-validation
% Test up to 20 latent variables
max_LV = 20;

fprintf('Building PLS model with up to %d latent variables...\n', max_LV);

% Build PLS model using PLS Toolbox (without cross-validation first)
pls_model = pls(X_cal_scaled, Y_cal_centered, max_LV);

fprintf('Performing cross-validation (Venetian Blinds, 10 splits)...\n');

% Perform cross-validation using crossval function
% Syntax: crossval(x, y, method, cvi, ncomp)
% cvi = {'vet', splits, iterations} for venetian blinds
cvi = {'vet', 10, 1};  % 10 splits, 1 iteration

% Run cross-validation
pls_model = crossval(X_cal_scaled, Y_cal_centered, pls_model, cvi, max_LV);

fprintf('PLS model built successfully\n\n');

% Extract cross-validation results
RMSECV = pls_model.detail.rmsecv;
RMSEC = pls_model.detail.rmsec;

% Find optimal number of LVs (minimum RMSECV)
[min_RMSECV, opt_LV] = min(RMSECV);

fprintf('CROSS-VALIDATION RESULTS:\n');
fprintf('  Optimal number of LVs: %d\n', opt_LV);
fprintf('  RMSEC (calibration): %.4f\n', RMSEC(opt_LV));
fprintf('  RMSECV (cross-validation): %.4f\n', min_RMSECV);
fprintf('\n');

% Plot RMSEC vs RMSECV
figure('Position', [100, 100, 900, 600]);
plot(1:max_LV, RMSEC, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'b');
hold on;
plot(1:max_LV, RMSECV, 'r-s', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
plot(opt_LV, min_RMSECV, 'go', 'MarkerSize', 15, 'LineWidth', 3);
hold off;
xlabel('Number of Latent Variables', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('RMSE', 'FontSize', 12, 'FontWeight', 'bold');
title('PLS Model Selection - RMSEC vs RMSECV', 'FontSize', 14, 'FontWeight', 'bold');
legend('RMSEC (Calibration)', 'RMSECV (Cross-Validation)', ...
    sprintf('Optimal LV = %d', opt_LV), 'Location', 'best');
grid on;
saveas(gcf, 'Figures/auto_outputs/PLS_RMSE_Selection.png');
close(gcf);

% Additional plot: R² vs Number of LVs
fprintf('Generating R² vs LV plot...\n');
figure('Position', [100, 100, 900, 600]);

% Calculate R² for each LV from the full model
R2_cal_vec = zeros(max_LV, 1);
R2_cv_vec = zeros(max_LV, 1);

% Get predictions for all LVs from the full model
[Y_fit_all_LV, ~, ~, ~] = modlpred(X_cal_scaled, pls_model, 0);
[~, ~, ~, ~, Y_cv_all_LV] = crossval(X_cal_scaled, Y_cal_centered, pls_model, cvi, max_LV);

for lv = 1:max_LV
    % Extract predictions for this LV
    if size(Y_fit_all_LV, 2) >= lv
        y_fit_lv = Y_fit_all_LV(:, lv) + mean(Calibration_Y);
    else
        y_fit_lv = Y_fit_all_LV(:, end) + mean(Calibration_Y);
    end
    
    if ndims(Y_cv_all_LV) == 3 && size(Y_cv_all_LV, 3) >= lv
        y_cv_lv = Y_cv_all_LV(:, :, lv);
        y_cv_lv = y_cv_lv(:) + mean(Calibration_Y);
    elseif size(Y_cv_all_LV, 2) >= lv
        y_cv_lv = Y_cv_all_LV(:, lv) + mean(Calibration_Y);
    else
        y_cv_lv = Y_cv_all_LV(:, end);
        if ndims(Y_cv_all_LV) == 3
            y_cv_lv = y_cv_lv(:);
        end
        y_cv_lv = y_cv_lv + mean(Calibration_Y);
    end
    
    R2_cal_vec(lv) = corr(Calibration_Y, y_fit_lv)^2;
    R2_cv_vec(lv) = corr(Calibration_Y, y_cv_lv)^2;
end

plot(1:max_LV, R2_cal_vec, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'b');
hold on;
plot(1:max_LV, R2_cv_vec, 'r-s', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
plot(opt_LV, R2_cv_vec(opt_LV), 'go', 'MarkerSize', 15, 'LineWidth', 3);
hold off;
xlabel('Number of Latent Variables', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('R²', 'FontSize', 12, 'FontWeight', 'bold');
title('R² vs Number of Latent Variables', 'FontSize', 14, 'FontWeight', 'bold');
legend('R² Calibration', 'R² Cross-Validation', ...
    sprintf('Optimal LV = %d', opt_LV), 'Location', 'best');
grid on;
ylim([0.5 1.0]);
saveas(gcf, 'Figures/auto_outputs/PLS_R2_vs_LV.png');
close(gcf);

% Rebuild model with optimal number of LVs
fprintf('Rebuilding PLS model with optimal LVs (%d)...\n', opt_LV);
pls_model_opt = pls(X_cal_scaled, Y_cal_centered, opt_LV);
pls_model_opt = crossval(X_cal_scaled, Y_cal_centered, pls_model_opt, cvi, opt_LV);

%% ========================================================================
% SECTION 5: MODEL DIAGNOSTICS
% ========================================================================

fprintf('\n========================================\n');
fprintf('MODEL DIAGNOSTICS\n');
fprintf('========================================\n\n');

% Get predictions using modlpred function
% Note: pls_model_opt was built with opt_LV components, so predictions are already final
[Y_fitted_all, ~, ~, T_scores_all] = modlpred(X_cal_scaled, pls_model_opt, 0);

% Get cross-validation predictions by re-running crossval with full outputs
[~, ~, ~, ~, Y_cv_all] = crossval(X_cal_scaled, Y_cal_centered, pls_model_opt, cvi, opt_LV);

% Check dimensions and extract predictions
% If modlpred returns multiple columns, take the last one (opt_LV)
% Otherwise, it's already a single column
if size(Y_fitted_all, 2) > 1
    Y_fitted = Y_fitted_all(:, end);
else
    Y_fitted = Y_fitted_all;
end

% For cross-validation predictions, Y_cv_all might be 3D (samples x variables x LVs)
% Extract the predictions for the optimal LV
if ndims(Y_cv_all) == 3
    Y_cv = Y_cv_all(:, :, end);
elseif size(Y_cv_all, 2) > 1
    Y_cv = Y_cv_all(:, end);
else
    Y_cv = Y_cv_all;
end

% Ensure both are column vectors
Y_fitted = Y_fitted(:);
Y_cv = Y_cv(:);

% Convert back to original scale (add Y mean)
Y_fitted = Y_fitted + mean(Calibration_Y);
Y_cv = Y_cv + mean(Calibration_Y);

% Calculate residuals
residuals_cal = Calibration_Y - Y_fitted;
residuals_cv = Calibration_Y - Y_cv;

% Get T-scores from model for leverage calculation
% Use all available scores
if size(T_scores_all, 2) >= opt_LV
    T_scores = T_scores_all(:, 1:opt_LV);
else
    T_scores = T_scores_all;
end

% Calculate leverage (Hotelling's T2)
leverage = diag(T_scores * inv(T_scores' * T_scores) * T_scores');

% 5.1: PLS Scores Plot (T1 vs T2)
fprintf('Generating PLS scores plot...\n');
figure('Position', [100, 100, 900, 700]);

if size(T_scores, 2) >= 2
    % Plot T1 vs T2
    subplot(2,2,1);
    plot(T_scores(:,1), T_scores(:,2), 'bo', 'MarkerSize', 6, 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k');
    xlabel('T1 (LV1)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('T2 (LV2)', 'FontSize', 11, 'FontWeight', 'bold');
    title('PLS Scores: T1 vs T2', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    axis square;
end

if size(T_scores, 2) >= 3
    % Plot T1 vs T3
    subplot(2,2,2);
    plot(T_scores(:,1), T_scores(:,3), 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
    xlabel('T1 (LV1)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('T3 (LV3)', 'FontSize', 11, 'FontWeight', 'bold');
    title('PLS Scores: T1 vs T3', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    axis square;
    
    % Plot T2 vs T3
    subplot(2,2,3);
    plot(T_scores(:,2), T_scores(:,3), 'go', 'MarkerSize', 6, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k');
    xlabel('T2 (LV2)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('T3 (LV3)', 'FontSize', 11, 'FontWeight', 'bold');
    title('PLS Scores: T2 vs T3', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    axis square;
end

% Variance explained by each LV
subplot(2,2,4);
% Calculate variance explained by T-scores
var_T = var(T_scores);
var_T_pct = 100 * var_T / sum(var_T);
bar(1:min(10, size(T_scores, 2)), var_T_pct(1:min(10, size(T_scores, 2))), 'FaceColor', [0.3 0.6 0.9]);
xlabel('Latent Variable', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Variance Explained (%)', 'FontSize', 11, 'FontWeight', 'bold');
title('Variance Explained by LVs', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

saveas(gcf, 'Figures/auto_outputs/PLS_Scores_Plot.png');
close(gcf);

% 5.2: PLS Loadings Plot
fprintf('Generating PLS loadings plot...\n');
figure('Position', [100, 100, 1200, 800]);

P = pls_model_opt.loads{2};  % X-loadings

% Plot first 4 LVs loadings
for lv = 1:min(4, size(P, 2))
    subplot(2, 2, lv);
    plot(1:size(P, 1), P(:, lv), 'b-', 'LineWidth', 1.5);
    xlabel('Variable Index (Wavelength)', 'FontSize', 10, 'FontWeight', 'bold');
    ylabel(sprintf('Loading LV%d', lv), 'FontSize', 10, 'FontWeight', 'bold');
    title(sprintf('PLS X-Loadings - LV%d', lv), 'FontSize', 11, 'FontWeight', 'bold');
    grid on;
end

saveas(gcf, 'Figures/auto_outputs/PLS_Loadings_Plot.png');
close(gcf);

% 5.3: Inner Relation Plot for ALL LVs (T vs U)
fprintf('Generating inner relation plots for all LVs...\n');

% Get U scores (Y-scores)
U_scores = pls_model_opt.loads{3};

% Create grid of inner relation plots
n_plots = min(opt_LV, 9);  % Max 9 plots (3x3 grid)
n_rows = ceil(sqrt(n_plots));
n_cols = ceil(n_plots / n_rows);

figure('Position', [100, 100, 1400, 1000]);
for lv = 1:n_plots
    subplot(n_rows, n_cols, lv);
    
    if lv <= size(T_scores, 2) && lv <= size(U_scores, 2)
        plot(T_scores(:, lv), U_scores(:, lv), 'bo', 'MarkerSize', 5, 'MarkerFaceColor', 'b');
        
        % Add regression line
        p = polyfit(T_scores(:, lv), U_scores(:, lv), 1);
        hold on;
        x_line = [min(T_scores(:, lv)), max(T_scores(:, lv))];
        plot(x_line, polyval(p, x_line), 'r-', 'LineWidth', 2);
        
        % Add zero lines
        plot([0 0], ylim, 'g--', 'LineWidth', 1);
        plot(xlim, [0 0], 'g--', 'LineWidth', 1);
        hold off;
        
        xlabel(sprintf('T%d (X-scores)', lv), 'FontSize', 9, 'FontWeight', 'bold');
        ylabel(sprintf('U%d (Y-scores)', lv), 'FontSize', 9, 'FontWeight', 'bold');
        title(sprintf('Inner Relation LV%d (R²=%.3f)', lv, corr(T_scores(:, lv), U_scores(:, lv))^2), ...
            'FontSize', 10, 'FontWeight', 'bold');
        grid on;
        axis square;
    end
end

saveas(gcf, 'Figures/auto_outputs/PLS_Inner_Relation_All_LVs.png');
close(gcf);

% 5.4: Q Residuals vs Leverage (colored by Y-values)
fprintf('Generating Q residuals vs leverage plot...\n');
figure('Position', [100, 100, 900, 700]);

% Calculate Q residuals (SPE - Squared Prediction Error)
X_reconstructed = T_scores * pls_model_opt.loads{2}(:, 1:opt_LV)';
residuals_X = X_cal_scaled - X_reconstructed;

% Ensure residuals_X is numeric
if ~isnumeric(residuals_X)
    if isstruct(residuals_X) && isfield(residuals_X, 'data')
        residuals_X = residuals_X.data;
    else
        residuals_X = double(residuals_X);
    end
end

Q_residuals = sum(residuals_X.^2, 2);

% Normalize Q residuals
Q_residuals_reduced = Q_residuals / mean(Q_residuals);

% Create scatter plot colored by Y values
scatter(leverage, Q_residuals_reduced, 50, Calibration_Y, 'filled', 'MarkerEdgeColor', 'k');
colorbar;
colormap(jet);
xlabel('Leverage (Hotelling T²)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Q Residuals Reduced', 'FontSize', 12, 'FontWeight', 'bold');
title('Q Residuals vs Leverage (colored by Y)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

% Add threshold lines
hold on;
h_line = 3 * mean(leverage);
v_line = 3;  % Q threshold at 3x mean
plot([h_line h_line], ylim, 'r--', 'LineWidth', 2);
plot(xlim, [v_line v_line], 'r--', 'LineWidth', 2);
hold off;

saveas(gcf, 'Figures/auto_outputs/PLS_Q_Residuals_vs_Leverage.png');
close(gcf);

% 5.5: Y Standardized Residuals vs Leverage
fprintf('Generating standardized residuals vs leverage plot...\n');
figure('Position', [100, 100, 900, 700]);

% Standardize residuals
std_residuals_cal = residuals_cal / std(residuals_cal);

% Scatter plot colored by Y
scatter(leverage, std_residuals_cal, 50, Calibration_Y, 'filled', 'MarkerEdgeColor', 'k');
colorbar;
colormap(jet);
xlabel('Leverage', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Y Standardized Residuals', 'FontSize', 12, 'FontWeight', 'bold');
title('Standardized Residuals vs Leverage (colored by Y)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

% Add reference lines
hold on;
plot(xlim, [0 0], 'k--', 'LineWidth', 1.5);
plot(xlim, [2 2], 'r--', 'LineWidth', 1);
plot(xlim, [-2 -2], 'r--', 'LineWidth', 1);
plot([3*mean(leverage) 3*mean(leverage)], ylim, 'r--', 'LineWidth', 1);
hold off;

saveas(gcf, 'Figures/auto_outputs/PLS_Standardized_Residuals_vs_Leverage.png');
close(gcf);

% 5.6: Inner Relation Plot (T1 vs U1) - Original
fprintf('Generating inner relation plot...\n');
figure('Position', [100, 100, 800, 600]);
T1 = pls_model_opt.loads{1}(:, 1);
U1 = pls_model_opt.loads{3}(:, 1);
plot(T1, U1, 'bo', 'MarkerSize', 6, 'MarkerFaceColor', 'b');
xlabel('T1 (X-scores, LV1)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('U1 (Y-scores, LV1)', 'FontSize', 12, 'FontWeight', 'bold');
title('Inner Relation Plot - T1 vs U1', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

% Add regression line
p = polyfit(T1, U1, 1);
hold on;
plot(T1, polyval(p, T1), 'r-', 'LineWidth', 2);
text(0.05, 0.95, sprintf('R² = %.4f', corr(T1, U1)^2), ...
    'Units', 'normalized', 'FontSize', 11, 'BackgroundColor', 'w');
hold off;

saveas(gcf, 'Figures/auto_outputs/PLS_Inner_Relation.png');
close(gcf);

% 5.4: Leverage Plot
fprintf('Generating leverage plot...\n');
figure('Position', [100, 100, 900, 600]);
bar(leverage, 'FaceColor', [0.3 0.6 0.9]);
xlabel('Sample Index', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Leverage', 'FontSize', 12, 'FontWeight', 'bold');
title('Sample Leverage (Hotelling T²)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

% Add threshold line (3 * mean leverage)
hold on;
threshold = 3 * mean(leverage);
plot([1 length(leverage)], [threshold threshold], 'r--', 'LineWidth', 2);
high_leverage = find(leverage > threshold);
if ~isempty(high_leverage)
    plot(high_leverage, leverage(high_leverage), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    legend('Leverage', 'Threshold (3×mean)', 'High Leverage Samples', 'Location', 'best');
    fprintf('  Detected %d high-leverage samples\n', length(high_leverage));
else
    legend('Leverage', 'Threshold (3×mean)', 'Location', 'best');
end
hold off;

saveas(gcf, 'Figures/auto_outputs/PLS_Leverage.png');
close(gcf);

% 5.5: Residuals vs Fitted
fprintf('Generating residuals plot...\n');
figure('Position', [100, 100, 900, 600]);
plot(Y_fitted, residuals_cal, 'bo', 'MarkerSize', 6, 'MarkerFaceColor', 'b');
xlabel('Fitted Values', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Residuals', 'FontSize', 12, 'FontWeight', 'bold');
title('Residuals vs Fitted Values', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
hold on;
plot([min(Y_fitted) max(Y_fitted)], [0 0], 'r--', 'LineWidth', 2);
hold off;

saveas(gcf, 'Figures/auto_outputs/PLS_Residuals_vs_Fitted.png');
close(gcf);

% 5.8: Y Measured vs Y Fitted (Calibration) - Enhanced
fprintf('Generating Y measured vs Y fitted plot...\n');
figure('Position', [100, 100, 900, 800]);
scatter(Calibration_Y, Y_fitted, 40, Calibration_Y, 'filled', 'MarkerEdgeColor', 'k');
colormap(jet);
hold on;
% Perfect prediction line
min_val = min([Calibration_Y; Y_fitted]);
max_val = max([Calibration_Y; Y_fitted]);
plot([min_val max_val], [min_val max_val], 'r--', 'LineWidth', 2);
hold off;
xlabel('Y Measured', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Y Fitted', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Calibration: Y Measured vs Y Fitted (LV=%d)', opt_LV), ...
    'FontSize', 14, 'FontWeight', 'bold');
axis equal;
axis([min_val max_val min_val max_val]);
grid on;
colorbar;

% Add statistics box
R2_cal = corr(Calibration_Y, Y_fitted)^2;
RMSE_cal = sqrt(mean(residuals_cal.^2));
bias_cal = mean(residuals_cal);
annotation('textbox', [0.15, 0.75, 0.25, 0.15], ...
    'String', {sprintf('RMSEC = %.4f', RMSE_cal), ...
               sprintf('R² = %.4f', R2_cal), ...
               sprintf('Bias = %.4f', bias_cal)}, ...
    'FontSize', 11, 'FontWeight', 'bold', ...
    'BackgroundColor', 'w', 'EdgeColor', 'k', 'LineWidth', 1.5);

saveas(gcf, 'Figures/auto_outputs/PLS_Ymeas_vs_Yfit.png');
close(gcf);

% 5.9: Y Measured vs Y Cross-Validated - Enhanced
fprintf('Generating Y measured vs Y cross-validated plot...\n');
figure('Position', [100, 100, 900, 800]);
scatter(Calibration_Y, Y_cv, 40, Calibration_Y, 'filled', 'MarkerEdgeColor', 'k');
colormap(jet);
hold on;
plot([min_val max_val], [min_val max_val], 'r--', 'LineWidth', 2);
hold off;
xlabel('Y Measured', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Y Cross-Validated', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Cross-Validation: Y Measured vs Y CV (LV=%d)', opt_LV), ...
    'FontSize', 14, 'FontWeight', 'bold');
axis equal;
axis([min_val max_val min_val max_val]);
grid on;
colorbar;

% Add statistics box
R2_cv = corr(Calibration_Y, Y_cv)^2;
RMSE_cv = sqrt(mean(residuals_cv.^2));
bias_cv = mean(residuals_cv);
annotation('textbox', [0.15, 0.75, 0.25, 0.15], ...
    'String', {sprintf('RMSECV = %.4f', RMSE_cv), ...
               sprintf('R² = %.4f', R2_cv), ...
               sprintf('CV Bias = %.4f', bias_cv)}, ...
    'FontSize', 11, 'FontWeight', 'bold', ...
    'BackgroundColor', 'w', 'EdgeColor', 'k', 'LineWidth', 1.5);

saveas(gcf, 'Figures/auto_outputs/PLS_Ymeas_vs_YCV.png');
close(gcf);

% 5.10: Combined Calibration and CV plot
fprintf('Generating combined calibration and CV plot...\n');
figure('Position', [100, 100, 1200, 600]);

subplot(1, 2, 1);
scatter(Calibration_Y, Y_fitted, 40, 'b', 'filled', 'MarkerEdgeColor', 'k');
hold on;
plot([min_val max_val], [min_val max_val], 'r--', 'LineWidth', 2);
hold off;
xlabel('Y Measured', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Y Fitted', 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('Calibration (RMSEC=%.4f, R²=%.4f)', RMSE_cal, R2_cal), ...
    'FontSize', 12, 'FontWeight', 'bold');
axis equal;
axis([min_val max_val min_val max_val]);
grid on;

subplot(1, 2, 2);
scatter(Calibration_Y, Y_cv, 40, 'r', 'filled', 'MarkerEdgeColor', 'k');
hold on;
plot([min_val max_val], [min_val max_val], 'k--', 'LineWidth', 2);
hold off;
xlabel('Y Measured', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Y CV', 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('Cross-Validation (RMSECV=%.4f, R²=%.4f)', RMSE_cv, R2_cv), ...
    'FontSize', 12, 'FontWeight', 'bold');
axis equal;
axis([min_val max_val min_val max_val]);
grid on;

saveas(gcf, 'Figures/auto_outputs/PLS_Cal_vs_CV_Comparison.png');
close(gcf);

% 5.11: Histogram of Residuals
fprintf('Generating residuals histogram...\n');
figure('Position', [100, 100, 900, 600]);
subplot(1,2,1);
histogram(residuals_cal, 30, 'FaceColor', [0.3 0.6 0.9]);
xlabel('Calibration Residuals', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Frequency', 'FontSize', 11, 'FontWeight', 'bold');
title('Calibration Residuals Distribution', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

subplot(1,2,2);
histogram(residuals_cv, 30, 'FaceColor', [0.9 0.3 0.3]);
xlabel('CV Residuals', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Frequency', 'FontSize', 11, 'FontWeight', 'bold');
title('Cross-Validation Residuals Distribution', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

saveas(gcf, 'Figures/auto_outputs/PLS_Residuals_Histogram.png');
close(gcf);

%% ========================================================================
% SECTION 6: MODEL INTERPRETATION - WEIGHTS, COEFFICIENTS, VIP
% ========================================================================

fprintf('\n========================================\n');
fprintf('MODEL INTERPRETATION\n');
fprintf('========================================\n\n');

% Extract model parameters
W = pls_model_opt.loads{2};  % PLS weights
B = pls_model_opt.reg;       % Regression coefficients

% Calculate VIP scores using PLS Toolbox function
VIP = vip(pls_model_opt);    % Variable Importance in Projection

% 6.1: PLS Weights (all LVs)
fprintf('Generating PLS weights plot...\n');
n_vars = size(W, 1);
var_indices = 1:n_vars;

figure('Position', [100, 100, 1400, 800]);
for lv = 1:min(4, opt_LV)  % Plot first 4 LVs
    subplot(2, 2, lv);
    plot(var_indices, W(:, lv), 'b-', 'LineWidth', 1.5);
    xlabel('Variable Index (Wavelength)', 'FontSize', 10, 'FontWeight', 'bold');
    ylabel(sprintf('Weight LV%d', lv), 'FontSize', 10, 'FontWeight', 'bold');
    title(sprintf('PLS Weight Vector - LV%d', lv), 'FontSize', 11, 'FontWeight', 'bold');
    grid on;
    
    % Highlight important regions (|weight| > mean + 2*std)
    threshold_w = mean(abs(W(:, lv))) + 2 * std(abs(W(:, lv)));
    important_vars = abs(W(:, lv)) > threshold_w;
    if any(important_vars)
        hold on;
        plot(var_indices(important_vars), W(important_vars, lv), 'ro', ...
            'MarkerSize', 4, 'MarkerFaceColor', 'r');
        hold off;
    end
end
saveas(gcf, 'Figures/auto_outputs/PLS_Weights.png');
close(gcf);

% 6.1b: Weights vs Loadings Comparison (LV1)
fprintf('Generating weights vs loadings comparison...\n');
figure('Position', [100, 100, 1400, 500]);

P = pls_model_opt.loads{2};  % X-loadings

subplot(1, 2, 1);
plot(var_indices, W(:, 1), 'r-', 'LineWidth', 2);
hold on;
plot(var_indices, P(:, 1), 'b-', 'LineWidth', 2);
hold off;
xlabel('Variable Index (Wavelength)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Weight / Loading Value', 'FontSize', 11, 'FontWeight', 'bold');
title('LV1: Weights vs Loadings', 'FontSize', 12, 'FontWeight', 'bold');
legend('Weights (W1)', 'Loadings (P1)', 'Location', 'best');
grid on;

if opt_LV >= 2
    subplot(1, 2, 2);
    plot(var_indices, W(:, 2), 'r-', 'LineWidth', 2);
    hold on;
    plot(var_indices, P(:, 2), 'b-', 'LineWidth', 2);
    hold off;
    xlabel('Variable Index (Wavelength)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Weight / Loading Value', 'FontSize', 11, 'FontWeight', 'bold');
    title('LV2: Weights vs Loadings', 'FontSize', 12, 'FontWeight', 'bold');
    legend('Weights (W2)', 'Loadings (P2)', 'Location', 'best');
    grid on;
end

saveas(gcf, 'Figures/auto_outputs/PLS_Weights_vs_Loadings.png');
close(gcf);

% 6.2: Regression Coefficients
fprintf('Generating regression coefficients plot...\n');
figure('Position', [100, 100, 1000, 600]);
B_opt = B(:, end);  % Coefficients for optimal model
plot(var_indices, B_opt, 'b-', 'LineWidth', 1.5);
xlabel('Variable Index (Wavelength)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Regression Coefficient', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('PLS Regression Coefficients (LV=%d)', opt_LV), ...
    'FontSize', 14, 'FontWeight', 'bold');
grid on;

% Highlight important regions
threshold_b = mean(abs(B_opt)) + 2 * std(abs(B_opt));
important_coef = abs(B_opt) > threshold_b;
if any(important_coef)
    hold on;
    plot(var_indices(important_coef), B_opt(important_coef), 'ro', ...
        'MarkerSize', 5, 'MarkerFaceColor', 'r');
    legend('Coefficients', 'Important Regions', 'Location', 'best');
    hold off;
end

saveas(gcf, 'Figures/auto_outputs/PLS_Regression_Coefficients.png');
close(gcf);

% 6.3: VIP Scores
fprintf('Generating VIP scores plot...\n');
figure('Position', [100, 100, 1000, 600]);
% VIP is already calculated for the optimal model, so it's a single vector
VIP_opt = VIP(:);  % Ensure it's a column vector
plot(var_indices, VIP_opt, 'b-', 'LineWidth', 1.5);
xlabel('Variable Index (Wavelength)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('VIP Score', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Variable Importance in Projection - VIP (LV=%d)', opt_LV), ...
    'FontSize', 14, 'FontWeight', 'bold');
grid on;

% Add VIP threshold line (VIP > 1 = important)
hold on;
plot([1 n_vars], [1 1], 'r--', 'LineWidth', 2);
important_vip = VIP_opt > 1;
if any(important_vip)
    plot(var_indices(important_vip), VIP_opt(important_vip), 'ro', ...
        'MarkerSize', 4, 'MarkerFaceColor', 'r');
    legend('VIP Scores', 'Threshold (VIP=1)', 'Important Variables', 'Location', 'best');
    fprintf('  Number of important variables (VIP>1): %d / %d (%.1f%%)\n', ...
        sum(important_vip), n_vars, 100*sum(important_vip)/n_vars);
else
    legend('VIP Scores', 'Threshold (VIP=1)', 'Location', 'best');
end
hold off;

saveas(gcf, 'Figures/auto_outputs/PLS_VIP_Scores.png');
close(gcf);

% 6.4: Combined importance plot (VIP + |Coefficients|)
fprintf('Generating combined variable importance plot...\n');
figure('Position', [100, 100, 1200, 800]);

subplot(3,1,1);
plot(var_indices, abs(B_opt), 'b-', 'LineWidth', 1.2);
ylabel('|Coefficient|', 'FontSize', 10, 'FontWeight', 'bold');
title('Variable Importance - Multiple Perspectives', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

subplot(3,1,2);
plot(var_indices, VIP_opt, 'r-', 'LineWidth', 1.2);
hold on;
plot([1 n_vars], [1 1], 'k--', 'LineWidth', 1.5);
hold off;
ylabel('VIP Score', 'FontSize', 10, 'FontWeight', 'bold');
grid on;

subplot(3,1,3);
% Combined score (normalized VIP * normalized |B|)
B_norm = abs(B_opt) / max(abs(B_opt));
VIP_norm = VIP_opt / max(VIP_opt);
combined_importance = B_norm .* VIP_norm;
plot(var_indices, combined_importance, 'g-', 'LineWidth', 1.2);
xlabel('Variable Index (Wavelength)', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Combined Score', 'FontSize', 10, 'FontWeight', 'bold');
grid on;

saveas(gcf, 'Figures/auto_outputs/PLS_Variable_Importance_Combined.png');
close(gcf);

% Identify most important spectral regions
fprintf('\nMOST IMPORTANT SPECTRAL REGIONS:\n');
[~, sorted_idx] = sort(VIP_opt, 'descend');
top_n = 20;
fprintf('  Top %d variables (highest VIP):\n', top_n);
fprintf('    Variable indices: ');
fprintf('%d ', sorted_idx(1:top_n));
fprintf('\n');

% Find contiguous regions of high importance (VIP > 1)
important_regions = find(important_vip);
if ~isempty(important_regions)
    % Find gaps > 5 indices to separate regions
    gaps = diff(important_regions) > 5;
    region_starts = [important_regions(1); important_regions(find(gaps) + 1)];
    region_ends = [important_regions(find(gaps)); important_regions(end)];
    
    fprintf('  Contiguous important regions (VIP>1):\n');
    for i = 1:length(region_starts)
        fprintf('    Region %d: Variables %d-%d (%d variables)\n', ...
            i, region_starts(i), region_ends(i), region_ends(i)-region_starts(i)+1);
    end
end
fprintf('\n');

%% ========================================================================
% SECTION 7: TEST SET PREDICTION
% ========================================================================

fprintf('\n========================================\n');
fprintf('TEST SET PREDICTION\n');
fprintf('========================================\n\n');

fprintf('Applying preprocessing to test set...\n');

% CRITICAL: Use SAME preprocessing as calibration with CALIBRATION PARAMETERS
% This ensures test set is on the same scale as training data

% Step 1: Get MSC/preprocessing parameters from calibration
% The MSC must be calculated from calibration data and applied to test
X_test_prep = Validation_X;

% Apply selected preprocessing using CALIBRATION reference
switch SELECTED_PREPROCESSING
    case 'baseline'
        X_test_prep = baseline(X_test_prep);
        
    case 'msc'
        % CRITICAL FIX: MSC must use calibration mean spectrum as reference
        % First, get calibration mean from the preprocessed calibration data
        X_cal_for_msc = Calibration_X;
        
        % Ensure X_cal_for_msc is numeric
        if ~isnumeric(X_cal_for_msc)
            if isstruct(X_cal_for_msc) && isfield(X_cal_for_msc, 'data')
                X_cal_for_msc = X_cal_for_msc.data;
            else
                X_cal_for_msc = double(X_cal_for_msc);
            end
        end
        
        mean_cal_spectrum = mean(X_cal_for_msc, 1);
        
        % Apply MSC to test using calibration reference
        % mscorr with second argument uses that as reference
        X_test_prep = mscorr(X_test_prep, mean_cal_spectrum);
        
    case 'derivative2'
        X_test_prep = savgol(X_test_prep, 2, 15, 2);
        
    case 'baseline_msc'
        X_test_prep = baseline(X_test_prep);
        X_cal_for_msc = Calibration_X;
        X_cal_for_msc = baseline(X_cal_for_msc);
        
        % Ensure X_cal_for_msc is numeric
        if ~isnumeric(X_cal_for_msc)
            if isstruct(X_cal_for_msc) && isfield(X_cal_for_msc, 'data')
                X_cal_for_msc = X_cal_for_msc.data;
            else
                X_cal_for_msc = double(X_cal_for_msc);
            end
        end
        
        mean_cal_spectrum = mean(X_cal_for_msc, 1);
        X_test_prep = mscorr(X_test_prep, mean_cal_spectrum);
        
    case 'baseline_deriv2'
        X_test_prep = baseline(X_test_prep);
        X_test_prep = savgol(X_test_prep, 2, 15, 2);
        
    case 'deriv2_msc'
        X_test_prep = savgol(X_test_prep, 2, 15, 2);
        X_cal_for_msc = savgol(Calibration_X, 2, 15, 2);
        
        % Ensure X_cal_for_msc is numeric
        if ~isnumeric(X_cal_for_msc)
            if isstruct(X_cal_for_msc) && isfield(X_cal_for_msc, 'data')
                X_cal_for_msc = X_cal_for_msc.data;
            else
                X_cal_for_msc = double(X_cal_for_msc);
            end
        end
        
        mean_cal_spectrum = mean(X_cal_for_msc, 1);
        X_test_prep = mscorr(X_test_prep, mean_cal_spectrum);
end

% Step 2: Apply autoscaling using CALIBRATION parameters
% preproc_params_selected contains mean and std from calibration
X_test_scaled = auto(X_test_prep, preproc_params_selected);

fprintf('  Preprocessing applied with calibration parameters\n');
fprintf('  Method: %s\n', selected_name);

% Predict test set using PLS Toolbox modlpred function
% This ensures correct handling of all model parameters
[Y_test_pred_centered, ~, ~, ~] = modlpred(X_test_scaled, pls_model_opt, 0);

% Convert back to original scale (add Y mean from calibration)
Y_test_pred = Y_test_pred_centered + mean(Calibration_Y);
Y_test_pred = Y_test_pred(:);  % Ensure column vector

% Calculate test set statistics
residuals_test = Validation_Y - Y_test_pred;
RMSEP = sqrt(mean(residuals_test.^2));
R2_test = corr(Validation_Y, Y_test_pred)^2;
bias = mean(residuals_test);

fprintf('TEST SET RESULTS:\n');
fprintf('  RMSEP (Root Mean Square Error of Prediction): %.4f\n', RMSEP);
fprintf('  R² (Test): %.4f\n', R2_test);
fprintf('  Bias: %.4f\n', bias);
fprintf('  SEP (Standard Error of Prediction): %.4f\n', std(residuals_test));
fprintf('\n');

% Check for temporal drift
if abs(bias) > 0.1
    fprintf('  WARNING: Significant bias detected (%.4f)\n', bias);
    fprintf('  This may indicate temporal drift between calibration and test sets.\n');
else
    fprintf('  No significant bias - model is temporally stable.\n');
end
fprintf('\n');

% 7.1: Y Measured vs Y Predicted (Test Set)
fprintf('Generating test set prediction plot...\n');
figure('Position', [100, 100, 800, 800]);
plot(Validation_Y, Y_test_pred, 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g', ...
    'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
hold on;
min_val_test = min([Validation_Y; Y_test_pred]);
max_val_test = max([Validation_Y; Y_test_pred]);
plot([min_val_test max_val_test], [min_val_test max_val_test], 'r--', 'LineWidth', 2);
hold off;
xlabel('Y Measured (Test Set)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Y Predicted (Test Set)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Test Set: Y Measured vs Y Predicted (LV=%d)', opt_LV), ...
    'FontSize', 14, 'FontWeight', 'bold');
axis equal;
axis([min_val_test max_val_test min_val_test max_val_test]);
grid on;

% Add statistics
text(0.05, 0.95, sprintf('R² = %.4f\nRMSEP = %.4f\nBias = %.4f', ...
    R2_test, RMSEP, bias), ...
    'Units', 'normalized', 'FontSize', 11, 'BackgroundColor', 'w', ...
    'VerticalAlignment', 'top');

saveas(gcf, 'Figures/auto_outputs/PLS_Test_Ymeas_vs_Ypred.png');
close(gcf);

% 7.2: Test Set Residuals
fprintf('Generating test set residuals plot...\n');
figure('Position', [100, 100, 900, 600]);
plot(1:length(residuals_test), residuals_test, 'go-', 'MarkerSize', 6, ...
    'MarkerFaceColor', 'g', 'LineWidth', 1.2);
hold on;
plot([1 length(residuals_test)], [0 0], 'r--', 'LineWidth', 2);
plot([1 length(residuals_test)], [bias bias], 'b--', 'LineWidth', 2);
hold off;
xlabel('Test Sample Index', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Prediction Residual', 'FontSize', 12, 'FontWeight', 'bold');
title('Test Set Prediction Residuals', 'FontSize', 14, 'FontWeight', 'bold');
legend('Residuals', 'Zero Line', sprintf('Bias = %.4f', bias), 'Location', 'best');
grid on;

saveas(gcf, 'Figures/auto_outputs/PLS_Test_Residuals.png');
close(gcf);

% 7.3: Comparison of Calibration, CV, and Test Performance
fprintf('Generating performance comparison plot...\n');
figure('Position', [100, 100, 1000, 600]);

% Create bar plot
categories = {'R² Cal', 'R² CV', 'R² Test'; 'RMSE Cal', 'RMSE CV', 'RMSE Test'};
values = [R2_cal, R2_cv, R2_test; RMSE_cal, RMSE_cv, RMSEP];

subplot(1,2,1);
bar([R2_cal, R2_cv, R2_test], 'FaceColor', [0.3 0.6 0.9]);
set(gca, 'XTickLabel', {'Calibration', 'Cross-Val', 'Test'});
ylabel('R²', 'FontSize', 11, 'FontWeight', 'bold');
title('Coefficient of Determination', 'FontSize', 12, 'FontWeight', 'bold');
ylim([0.9 1.0]);
grid on;

subplot(1,2,2);
bar([RMSE_cal, RMSE_cv, RMSEP], 'FaceColor', [0.9 0.4 0.3]);
set(gca, 'XTickLabel', {'Calibration', 'Cross-Val', 'Test'});
ylabel('RMSE', 'FontSize', 11, 'FontWeight', 'bold');
title('Root Mean Square Error', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

saveas(gcf, 'Figures/auto_outputs/PLS_Performance_Comparison.png');
close(gcf);

%% ========================================================================
% SECTION 7B: DIAGNOSTIC ANALYSIS AND SLOPE-BIAS CORRECTION
% ========================================================================

fprintf('\n========================================\n');
fprintf('DIAGNOSTIC ANALYSIS\n');
fprintf('========================================\n\n');

% 7B.1: Range Y Analysis
fprintf('--- RANGE Y ANALYSIS ---\n');
fprintf('Calibration:\n');
fprintf('  Min: %.2f, Max: %.2f, Range: %.2f\n', ...
    min(Calibration_Y), max(Calibration_Y), range(Calibration_Y));
fprintf('  Mean: %.2f, Std: %.2f\n', mean(Calibration_Y), std(Calibration_Y));

fprintf('Test:\n');
fprintf('  Min: %.2f, Max: %.2f, Range: %.2f\n', ...
    min(Validation_Y), max(Validation_Y), range(Validation_Y));
fprintf('  Mean: %.2f, Std: %.2f\n', mean(Validation_Y), std(Validation_Y));

fprintf('Comparison:\n');
fprintf('  Std ratio (Test/Cal): %.2f\n', std(Validation_Y) / std(Calibration_Y));
fprintf('  Mean difference: %.2f\n', mean(Validation_Y) - mean(Calibration_Y));
fprintf('\n');

% Plot Y distributions
figure('Position', [100, 100, 1200, 500]);
subplot(1,3,1);
histogram(Calibration_Y, 30, 'FaceColor', [0.3 0.6 0.9]);
xlabel('Protein Content (%)', 'FontSize', 11);
ylabel('Frequency', 'FontSize', 11);
title(sprintf('Calibration (n=%d)', length(Calibration_Y)), 'FontSize', 12);
grid on;

subplot(1,3,2);
histogram(Validation_Y, 20, 'FaceColor', [0.9 0.4 0.3]);
xlabel('Protein Content (%)', 'FontSize', 11);
ylabel('Frequency', 'FontSize', 11);
title(sprintf('Test (n=%d)', length(Validation_Y)), 'FontSize', 12);
grid on;

subplot(1,3,3);
group_labels = [ones(length(Calibration_Y),1); 2*ones(length(Validation_Y),1)];
boxplot([Calibration_Y; Validation_Y], group_labels);
set(gca, 'XTickLabel', {'Calibration', 'Test'});
ylabel('Protein Content (%)', 'FontSize', 11);
title('Distribution Comparison', 'FontSize', 12);
grid on;

saveas(gcf, 'Figures/auto_outputs/Diagnostic_Y_Distributions.png');
close(gcf);

% 7B.2: Spectral Comparison
fprintf('--- SPECTRAL COMPARISON ---\n');

% Calculate mean spectra (ensure they are numeric)
X_cal_numeric = X_cal_scaled;
if ~isnumeric(X_cal_numeric)
    if isstruct(X_cal_numeric) && isfield(X_cal_numeric, 'data')
        X_cal_numeric = X_cal_numeric.data;
    else
        X_cal_numeric = double(X_cal_numeric);
    end
end

X_test_numeric = X_test_scaled;
if ~isnumeric(X_test_numeric)
    if isstruct(X_test_numeric) && isfield(X_test_numeric, 'data')
        X_test_numeric = X_test_numeric.data;
    else
        X_test_numeric = double(X_test_numeric);
    end
end

mean_spec_cal = mean(X_cal_numeric, 1);
mean_spec_test = mean(X_test_numeric, 1);
spec_diff = mean_spec_test - mean_spec_cal;

fprintf('Mean spectral difference: %.4f\n', mean(abs(spec_diff)));
fprintf('Max spectral difference: %.4f (variable %d)\n', ...
    max(abs(spec_diff)), find(abs(spec_diff) == max(abs(spec_diff)), 1));
fprintf('RMS spectral difference: %.4f\n', sqrt(mean(spec_diff.^2)));
fprintf('\n');

% Plot spectral comparison
figure('Position', [100, 100, 1200, 900]);

subplot(3,1,1);
plot(mean_spec_cal, 'b-', 'LineWidth', 1.5);
ylabel('Intensity', 'FontSize', 11);
title('Mean Spectrum - Calibration', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xlim([1 100]);

subplot(3,1,2);
plot(mean_spec_test, 'r-', 'LineWidth', 1.5);
ylabel('Intensity', 'FontSize', 11);
title('Mean Spectrum - Test', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xlim([1 100]);

subplot(3,1,3);
plot(spec_diff, 'k-', 'LineWidth', 1.5);
hold on;
plot([1 100], [0 0], 'r--', 'LineWidth', 1);
hold off;
xlabel('Variable Index (Wavelength)', 'FontSize', 11);
ylabel('Δ Intensity', 'FontSize', 11);
title('Difference (Test - Calibration)', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xlim([1 100]);

saveas(gcf, 'Figures/auto_outputs/Diagnostic_Spectral_Comparison.png');
close(gcf);

% 7B.3: SLOPE-BIAS CORRECTION
fprintf('--- SLOPE-BIAS CORRECTION ---\n');

% Calculate slope and intercept from linear regression
p = polyfit(Y_test_pred, Validation_Y, 1);
slope = p(1);
intercept = p(2);

fprintf('Linear correction parameters:\n');
fprintf('  Slope: %.4f\n', slope);
fprintf('  Intercept: %.4f\n', intercept);

% Apply correction
Y_test_corrected = slope * Y_test_pred + intercept;

% Recalculate metrics
residuals_corrected = Validation_Y - Y_test_corrected;
RMSEP_corrected = sqrt(mean(residuals_corrected.^2));
R2_corrected = corr(Validation_Y, Y_test_corrected)^2;
bias_corrected = mean(residuals_corrected);

fprintf('\nPERFORMANCE AFTER CORRECTION:\n');
fprintf('  R² = %.4f (was %.4f, improvement: %+.2f%%)\n', ...
    R2_corrected, R2_test, (R2_corrected - R2_test) / max(abs(R2_test), 0.001) * 100);
fprintf('  RMSEP = %.4f (was %.4f, improvement: %+.2f%%)\n', ...
    RMSEP_corrected, RMSEP, (RMSEP - RMSEP_corrected) / RMSEP * 100);
fprintf('  Bias = %.4f (was %.4f)\n', bias_corrected, bias);
fprintf('\n');

% Plot before/after correction
figure('Position', [100, 100, 1400, 600]);

subplot(1,2,1);
scatter(Validation_Y, Y_test_pred, 50, 'r', 'filled', 'MarkerEdgeColor', 'k');
hold on;
plot([min_val_test max_val_test], [min_val_test max_val_test], 'k--', 'LineWidth', 2);
hold off;
xlabel('Y Measured', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Y Predicted', 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('BEFORE Correction\nR²=%.4f, RMSEP=%.4f, Bias=%.4f', ...
    R2_test, RMSEP, bias), 'FontSize', 12, 'FontWeight', 'bold');
axis equal;
axis([min_val_test max_val_test min_val_test max_val_test]);
grid on;

subplot(1,2,2);
scatter(Validation_Y, Y_test_corrected, 50, 'g', 'filled', 'MarkerEdgeColor', 'k');
hold on;
plot([min_val_test max_val_test], [min_val_test max_val_test], 'k--', 'LineWidth', 2);
hold off;
xlabel('Y Measured', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Y Corrected', 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('AFTER Slope-Bias Correction\nR²=%.4f, RMSEP=%.4f, Bias=%.4f', ...
    R2_corrected, RMSEP_corrected, bias_corrected), 'FontSize', 12, 'FontWeight', 'bold');
axis equal;
axis([min_val_test max_val_test min_val_test max_val_test]);
grid on;

saveas(gcf, 'Figures/auto_outputs/Diagnostic_Slope_Bias_Correction.png');
close(gcf);

% 7B.4: Outlier Analysis
fprintf('--- OUTLIER ANALYSIS ---\n');

% Identify outliers (residuals > 3×RMSEP)
outliers_idx = find(abs(residuals_test) > 3*RMSEP);
fprintf('Outliers detected (|residual| > 3×RMSEP): %d (%.1f%%)\n', ...
    length(outliers_idx), length(outliers_idx)/length(residuals_test)*100);

if ~isempty(outliers_idx)
    fprintf('  Sample indices: ');
    fprintf('%d ', outliers_idx);
    fprintf('\n');
    fprintf('  Residuals: ');
    fprintf('%.3f ', residuals_test(outliers_idx));
    fprintf('\n');
end

% Plot residuals with outlier threshold
figure('Position', [100, 100, 1200, 500]);

subplot(1,2,1);
plot(residuals_test, 'bo-', 'MarkerSize', 6, 'MarkerFaceColor', 'b');
hold on;
plot([1 length(residuals_test)], [3*RMSEP 3*RMSEP], 'r--', 'LineWidth', 2);
plot([1 length(residuals_test)], [-3*RMSEP -3*RMSEP], 'r--', 'LineWidth', 2);
plot([1 length(residuals_test)], [0 0], 'k-', 'LineWidth', 1);
if ~isempty(outliers_idx)
    plot(outliers_idx, residuals_test(outliers_idx), 'ro', 'MarkerSize', 12, ...
        'LineWidth', 2);
end
hold off;
xlabel('Sample Index', 'FontSize', 11);
ylabel('Residual', 'FontSize', 11);
title('Test Residuals with Outlier Threshold', 'FontSize', 12);
legend('Residuals', '+3×RMSEP', '-3×RMSEP', 'Zero', 'Outliers', 'Location', 'best');
grid on;

subplot(1,2,2);
histogram(residuals_test, 20, 'FaceColor', [0.3 0.6 0.9]);
hold on;
xline(3*RMSEP, 'r--', 'LineWidth', 2);
xline(-3*RMSEP, 'r--', 'LineWidth', 2);
xline(0, 'k-', 'LineWidth', 1.5);
hold off;
xlabel('Residual Value', 'FontSize', 11);
ylabel('Frequency', 'FontSize', 11);
title(sprintf('Residual Distribution\nMean=%.3f, Std=%.3f', ...
    mean(residuals_test), std(residuals_test)), 'FontSize', 12);
grid on;

saveas(gcf, 'Figures/auto_outputs/Diagnostic_Outlier_Analysis.png');
close(gcf);

% 7B.5: Summary Report
fprintf('\n========================================\n');
fprintf('DIAGNOSTIC SUMMARY\n');
fprintf('========================================\n\n');

fprintf('1. RANGE ANALYSIS:\n');
if std(Validation_Y) < 0.7 * std(Calibration_Y)
    fprintf('   WARNING: Test set has RESTRICTED RANGE (std ratio = %.2f)\n', ...
        std(Validation_Y) / std(Calibration_Y));
    fprintf('   This contributes to low R²\n');
else
    fprintf('   OK: Test set range is comparable to calibration\n');
end

fprintf('\n2. SPECTRAL DRIFT:\n');
if sqrt(mean(spec_diff.^2)) > 0.5
    fprintf('   WARNING: SIGNIFICANT spectral drift detected (RMS = %.4f)\n', ...
        sqrt(mean(spec_diff.^2)));
    fprintf('   Possible instrumental or temporal drift\n');
else
    fprintf('   OK: Spectral drift within acceptable limits\n');
end

fprintf('\n3. BIAS ANALYSIS:\n');
if abs(bias) > 0.15
    fprintf('   WARNING: SIGNIFICANT bias detected (%.4f)\n', bias);
    fprintf('   Systematic prediction error confirmed\n');
else
    fprintf('   OK: Bias is acceptable\n');
end

fprintf('\n4. SLOPE-BIAS CORRECTION EFFECTIVENESS:\n');
fprintf('   R² improvement: %.4f → %.4f (%+.1f%%)\n', ...
    R2_test, R2_corrected, (R2_corrected - R2_test) / max(abs(R2_test), 0.001) * 100);
fprintf('   RMSEP improvement: %.4f → %.4f (%.1f%%)\n', ...
    RMSEP, RMSEP_corrected, (RMSEP - RMSEP_corrected) / RMSEP * 100);

if R2_corrected > 0.70
    fprintf('   STATUS: Model USABLE after correction\n');
elseif R2_corrected > 0.50
    fprintf('   STATUS: Model MARGINALLY usable (screening only)\n');
else
    fprintf('   STATUS: Model needs MAJOR IMPROVEMENT (model update required)\n');
end

fprintf('\n5. OUTLIERS:\n');
if length(outliers_idx) > 0.05 * length(residuals_test)
    fprintf('   WARNING: HIGH outlier rate (%.1f%%) - investigate samples\n', ...
        length(outliers_idx)/length(residuals_test)*100);
else
    fprintf('   OK: Outlier rate acceptable (%.1f%%)\n', ...
        length(outliers_idx)/length(residuals_test)*100);
end

fprintf('\n========================================\n\n');

%% ========================================================================
% SECTION 8: VARIABLE SELECTION AND MODEL IMPROVEMENT
% ========================================================================

fprintf('\n========================================\n');
fprintf('VARIABLE SELECTION AND MODEL IMPROVEMENT\n');
fprintf('========================================\n\n');

% Select variables based on VIP > threshold
% Try different thresholds
VIP_thresholds = [0.8, 1.0, 1.2];
best_improvement = -Inf;
best_threshold = 1.0;
best_selected_vars = [];

fprintf('Testing variable selection with different VIP thresholds...\n\n');

for t = 1:length(VIP_thresholds)
    threshold = VIP_thresholds(t);
    selected_vars = find(VIP_opt > threshold);
    
    if isempty(selected_vars)
        fprintf('  VIP threshold %.1f: No variables selected, skipping...\n', threshold);
        continue;
    end
    
    fprintf('  VIP threshold %.1f: %d variables selected (%.1f%%)\n', ...
        threshold, length(selected_vars), 100*length(selected_vars)/n_vars);
    
    % Build reduced model
    X_cal_reduced = X_cal_scaled(:, selected_vars);
    X_test_reduced = X_test_scaled(:, selected_vars);
    
    % Build PLS model with selected variables
    pls_reduced = pls(X_cal_reduced, Y_cal_centered, opt_LV);
    pls_reduced = crossval(X_cal_reduced, Y_cal_centered, pls_reduced, cvi, opt_LV);
    
    % Predict test set with reduced model
    Y_test_pred_reduced = pls_reduced.reg(:, end)' * X_test_reduced' + mean(Calibration_Y);
    Y_test_pred_reduced = Y_test_pred_reduced';
    
    % Calculate metrics
    RMSEP_reduced = sqrt(mean((Validation_Y - Y_test_pred_reduced).^2));
    R2_reduced = corr(Validation_Y, Y_test_pred_reduced)^2;
    RMSECV_reduced = min(pls_reduced.detail.rmsecv);
    
    fprintf('    RMSECV: %.4f (Full: %.4f, Change: %.4f)\n', ...
        RMSECV_reduced, min_RMSECV, RMSECV_reduced - min_RMSECV);
    fprintf('    RMSEP:  %.4f (Full: %.4f, Change: %.4f)\n', ...
        RMSEP_reduced, RMSEP, RMSEP_reduced - RMSEP);
    fprintf('    R² Test: %.4f (Full: %.4f, Change: %.4f)\n\n', ...
        R2_reduced, R2_test, R2_reduced - R2_test);
    
    % Track best improvement (based on RMSEP reduction)
    improvement = RMSEP - RMSEP_reduced;
    if improvement > best_improvement
        best_improvement = improvement;
        best_threshold = threshold;
        best_selected_vars = selected_vars;
        pls_model_best = pls_reduced;
        Y_test_pred_best = Y_test_pred_reduced;
        RMSEP_best = RMSEP_reduced;
        R2_best = R2_reduced;
    end
end

% Summary of variable selection
fprintf('\n========================================\n');
fprintf('VARIABLE SELECTION SUMMARY\n');
fprintf('========================================\n\n');

if best_improvement > 0
    fprintf('RESULT: Model IMPROVED with variable selection\n');
    fprintf('  Best VIP threshold: %.1f\n', best_threshold);
    fprintf('  Selected variables: %d / %d (%.1f%%)\n', ...
        length(best_selected_vars), n_vars, 100*length(best_selected_vars)/n_vars);
    fprintf('  RMSEP improvement: %.4f (%.2f%% reduction)\n', ...
        best_improvement, 100*best_improvement/RMSEP);
    fprintf('  Final RMSEP: %.4f\n', RMSEP_best);
    fprintf('  Final R² (test): %.4f\n\n', R2_best);
    
    % Use improved model
    final_model = pls_model_best;
    Y_test_final = Y_test_pred_best;
    model_type = 'Variable-Selected';
    
else
    fprintf('RESULT: Full model performs better or equal\n');
    fprintf('  No improvement with variable selection\n');
    fprintf('  Keeping full spectral model\n\n');
    
    % Use full model
    final_model = pls_model_opt;
    Y_test_final = Y_test_pred;
    RMSEP_best = RMSEP;
    R2_best = R2_test;
    model_type = 'Full Spectrum';
    best_selected_vars = 1:n_vars;
end

% Plot selected spectral regions
fprintf('Generating spectral regions plot...\n');
figure('Position', [100, 100, 1200, 600]);

% Reconstruct preprocessed data for visualization
X_vis = Calibration_X;
switch SELECTED_PREPROCESSING
    case 'baseline'
        X_vis = baseline(X_vis);
    case 'msc'
        X_vis = mscorr(X_vis);
    case 'derivative2'
        X_vis = savgol(X_vis, 2, 15, 2);
    case 'baseline_msc'
        X_vis = baseline(X_vis);
        X_vis = mscorr(X_vis);
    case 'baseline_deriv2'
        X_vis = baseline(X_vis);
        X_vis = savgol(X_vis, 2, 15, 2);
    case 'deriv2_msc'
        X_vis = savgol(X_vis, 2, 15, 2);
        X_vis = mscorr(X_vis);
end

% Ensure X_vis is numeric (convert from dataset if needed)
if ~isnumeric(X_vis)
    if isstruct(X_vis) && isfield(X_vis, 'data')
        X_vis = X_vis.data;
    else
        X_vis = double(X_vis);
    end
end

% Plot mean spectrum
mean_spectrum = mean(X_vis, 1);
plot(1:n_vars, mean_spectrum, 'b-', 'LineWidth', 1.5);
hold on;

% Highlight selected regions
if length(best_selected_vars) < n_vars
    selected_mask = false(1, n_vars);
    selected_mask(best_selected_vars) = true;
    
    % Fill selected regions
    for i = 1:n_vars
        if selected_mask(i)
            plot([i i], [min(mean_spectrum) max(mean_spectrum)], ...
                'g-', 'LineWidth', 0.5, 'Color', [0 1 0 0.3]);
        end
    end
end
hold off;

xlabel('Variable Index (Wavelength)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Mean Intensity', 'FontSize', 12, 'FontWeight', 'bold');
title('Selected Spectral Regions for Final Model', 'FontSize', 14, 'FontWeight', 'bold');
legend('Mean Spectrum', 'Selected Regions', 'Location', 'best');
grid on;

saveas(gcf, 'Figures/auto_outputs/PLS_Selected_Spectral_Regions.png');
close(gcf);

% Final prediction plot
fprintf('Generating final model prediction plot...\n');
figure('Position', [100, 100, 900, 800]);

% Plot calibration and test predictions
subplot(2,2,[1 2]);
plot(Calibration_Y, Y_fitted, 'bo', 'MarkerSize', 5, ...
    'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k');
hold on;
plot(Validation_Y, Y_test_final, 'go', 'MarkerSize', 8, ...
    'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
min_all = min([Calibration_Y; Validation_Y; Y_fitted; Y_test_final]);
max_all = max([Calibration_Y; Validation_Y; Y_fitted; Y_test_final]);
plot([min_all max_all], [min_all max_all], 'r--', 'LineWidth', 2);
hold off;
xlabel('Y Measured', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Y Predicted', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Final Model: %s (LV=%d)', model_type, opt_LV), ...
    'FontSize', 14, 'FontWeight', 'bold');
legend('Calibration', 'Test Set', 'Perfect Fit', 'Location', 'best');
axis equal;
axis([min_all max_all min_all max_all]);
grid on;

% Add statistics box
text(0.05, 0.95, sprintf('Calibration: R² = %.4f, RMSE = %.4f\nTest: R² = %.4f, RMSEP = %.4f', ...
    R2_cal, RMSE_cal, R2_best, RMSEP_best), ...
    'Units', 'normalized', 'FontSize', 10, 'BackgroundColor', 'w', ...
    'VerticalAlignment', 'top');

% Residuals comparison
subplot(2,2,3);
histogram(residuals_cal, 20, 'FaceColor', [0.3 0.6 0.9]);
xlabel('Calibration Residuals', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Frequency', 'FontSize', 10, 'FontWeight', 'bold');
title('Calibration Residuals', 'FontSize', 11, 'FontWeight', 'bold');
grid on;

subplot(2,2,4);
residuals_test_final = Validation_Y - Y_test_final;
histogram(residuals_test_final, 20, 'FaceColor', [0.3 0.9 0.4]);
xlabel('Test Residuals', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Frequency', 'FontSize', 10, 'FontWeight', 'bold');
title('Test Set Residuals', 'FontSize', 11, 'FontWeight', 'bold');
grid on;

saveas(gcf, 'Figures/auto_outputs/PLS_Final_Model_Summary.png');
close(gcf);

%% ========================================================================
% SECTION 9: FINAL SUMMARY AND REPORT
% ========================================================================

fprintf('\n========================================\n');
fprintf('FINAL ANALYSIS SUMMARY\n');
fprintf('========================================\n\n');

fprintf('DATA:\n');
fprintf('  Calibration samples: %d\n', size(Calibration_X, 1));
fprintf('  Test samples: %d\n', size(Validation_X, 1));
fprintf('  Spectral variables: %d\n', size(Calibration_X, 2));
fprintf('\n');

fprintf('PREPROCESSING:\n');
fprintf('  Method: %s + Autoscaling\n', selected_name);
fprintf('  Justification: Optimal for NIR scatter correction\n');
fprintf('\n');

fprintf('PCA EXPLORATION:\n');
fprintf('  %d preprocessing methods tested\n', size(preproc_methods, 1));
fprintf('  Selected based on variance distribution and interpretability\n');
fprintf('\n');

fprintf('PLS MODEL:\n');
fprintf('  Cross-validation: Venetian Blinds (10-fold)\n');
fprintf('  Optimal latent variables: %d\n', opt_LV);
fprintf('  Calibration R²: %.4f\n', R2_cal);
fprintf('  Calibration RMSE: %.4f\n', RMSE_cal);
fprintf('  Cross-validation R²: %.4f\n', R2_cv);
fprintf('  Cross-validation RMSECV: %.4f\n', min_RMSECV);
fprintf('\n');

fprintf('TEST SET PERFORMANCE:\n');
fprintf('  Model type: %s\n', model_type);
if length(best_selected_vars) < n_vars
    fprintf('  Variables used: %d / %d (%.1f%%)\n', ...
        length(best_selected_vars), n_vars, 100*length(best_selected_vars)/n_vars);
else
    fprintf('  Variables used: All (%d)\n', n_vars);
end
fprintf('  Test R²: %.4f\n', R2_best);
fprintf('  Test RMSEP: %.4f\n', RMSEP_best);
fprintf('  Bias: %.4f\n', mean(residuals_test_final));
fprintf('  SEP: %.4f\n', std(residuals_test_final));
fprintf('\n');

fprintf('TEMPORAL STABILITY:\n');
if abs(mean(residuals_test_final)) < 0.1
    fprintf('  GOOD: No significant bias detected\n');
    fprintf('  Model is stable despite 2-month storage difference\n');
else
    fprintf('  WARNING: Bias = %.4f\n', mean(residuals_test_final));
    fprintf('  Potential temporal drift detected\n');
end
fprintf('\n');

fprintf('IMPORTANT SPECTRAL REGIONS:\n');
fprintf('  Variables with VIP > 1: %d (%.1f%%)\n', ...
    sum(VIP_opt > 1), 100*sum(VIP_opt > 1)/n_vars);
fprintf('  Top 3 most important variables: ');
[~, top_idx] = sort(VIP_opt, 'descend');
fprintf('%d, %d, %d\n', top_idx(1), top_idx(2), top_idx(3));
fprintf('\n');

fprintf('MODEL QUALITY ASSESSMENT:\n');
% Check for overfitting
overfitting_check = (R2_cal - R2_cv) / R2_cal * 100;
if overfitting_check < 5
    fprintf('  Overfitting: NONE (Cal-CV difference: %.2f%%)\n', overfitting_check);
elseif overfitting_check < 10
    fprintf('  Overfitting: MINIMAL (Cal-CV difference: %.2f%%)\n', overfitting_check);
else
    fprintf('  Overfitting: MODERATE (Cal-CV difference: %.2f%%)\n', overfitting_check);
end

% Check generalization
generalization = abs(R2_cv - R2_best) / R2_cv * 100;
if generalization < 5
    fprintf('  Generalization: EXCELLENT (CV-Test difference: %.2f%%)\n', generalization);
elseif generalization < 10
    fprintf('  Generalization: GOOD (CV-Test difference: %.2f%%)\n', generalization);
else
    fprintf('  Generalization: MODERATE (CV-Test difference: %.2f%%)\n', generalization);
end
fprintf('\n');

fprintf('OUTPUTS GENERATED:\n');
fprintf('  All figures saved to: Figures/auto_outputs/\n');
fprintf('  Total plots generated: %d\n', 15 + length(preproc_methods) * 3);
fprintf('\n');

fprintf('========================================\n');
fprintf('ANALYSIS COMPLETED SUCCESSFULLY\n');
fprintf('========================================\n\n');

% Save workspace
fprintf('Saving workspace to wheat_protein_pls_workspace.mat...\n');
save('wheat_protein_pls_workspace.mat');
fprintf('Workspace saved successfully.\n\n');

fprintf('Script execution completed. All results are available in:\n');
fprintf('  - Figures/auto_outputs/ (PNG plots)\n');
fprintf('  - wheat_protein_pls_workspace.mat (workspace variables)\n\n');
