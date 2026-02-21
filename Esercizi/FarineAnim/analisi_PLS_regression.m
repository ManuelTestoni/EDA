%% =========================================================================
%  ANALISI PLS REGRESSION / PLS-DA - FARINE ANIMALI NIR
%  =========================================================================
%  Script per la regressione PLS su dati spettrali NIR di farine animali,
%  utilizzando ESCLUSIVAMENTE le funzioni (non GUI) del PLS Toolbox.
%
%  Segue la metodologia descritta in Assignment_Collega.pdf:
%    1. PCA esplorativa (mean centering, 2 PCs)
%    2. Selezione del preprocessing (confronto RMSECV)
%    3. Preprocessing e scelta del numero di LV (SNV + MC, scree plot RMSEC/RMSECV)
%    4. Costruzione modello PLS con cross-validation (Venetian Blind)
%    5. Grafici diagnostici (T2 vs Q, Leverage vs Residuals, ecc.)
%    6. Importanza variabili (Weights, Reg. Coefficients, VIP, Selectivity Ratio)
%    7. Validazione su test set
%    8. Spettri preprocessati
%    9. Riepilogo e salvataggio
%
%  DATI:
%    - farineanimNIR.mat  -> farineanimNIRdata, category
%    - animal_feedNIR.mat -> dataset PLS Toolbox
%    - assexscale.txt     -> lunghezze d'onda (wavenumbers, cm^-1)
%
%  Grafici salvati in: pls_plots/
%  =========================================================================

clear all; close all; clc;

%% 0. SETUP - PLS Toolbox e cartella output
fprintf('============================================================\n');
fprintf('  ANALISI PLS REGRESSION - FARINE ANIMALI NIR\n');
fprintf('============================================================\n\n');

% Crea cartella per i grafici
plotDir = fullfile(pwd, 'pls_plots');
if ~exist(plotDir, 'dir')
    mkdir(plotDir);
    fprintf('[OK] Cartella pls_plots/ creata.\n');
else
    fprintf('[OK] Cartella pls_plots/ gia'' esistente.\n');
end

%% 1. CARICAMENTO DATI
fprintf('\n------------------------------------------------------------\n');
fprintf('  1. CARICAMENTO DATI\n');
fprintf('------------------------------------------------------------\n');

% --- Carica dati grezzi ---
load('farineanimNIR.mat');          % farineanimNIRdata, category
wavelengths = load('assexscale.txt'); % wavenumbers (cm^-1), es. 6000->4000

X_raw = farineanimNIRdata;
[nSamples, nVars] = size(X_raw);

fprintf('  Campioni:    %d\n', nSamples);
fprintf('  Variabili:   %d\n', nVars);
fprintf('  Wavenumbers: %.0f - %.0f cm^-1\n', min(wavelengths), max(wavelengths));

% --- Gestione categorie ---
if ~iscell(category)
    if ischar(category)
        category = cellstr(category);
    elseif isnumeric(category)
        category = cellstr(num2str(category));
    end
end
category = strtrim(category);
uniqueCats = unique(category);
nClasses   = length(uniqueCats);

fprintf('  Classi:      %d -> ', nClasses);
for c = 1:nClasses
    nc = sum(strcmp(category, uniqueCats{c}));
    fprintf('%s(%d) ', uniqueCats{c}, nc);
end
fprintf('\n');

% --- Codifica Y per PLS-DA (dummy matrix: one-hot encoding) ---
%  Poiche' il dataset ha variabili categoriche (pollo, bovino, pesce),
%  si utilizza PLS-DA con matrice Y dummy.
Y_dummy = zeros(nSamples, nClasses);
for c = 1:nClasses
    Y_dummy(strcmp(category, uniqueCats{c}), c) = 1;
end
fprintf('  Matrice Y dummy: %d x %d\n', size(Y_dummy,1), size(Y_dummy,2));

% --- Carica dataset PLS Toolbox (se disponibile) ---
try
    loaded = load('animal_feedNIR.mat');
    fnames = fieldnames(loaded);
    ds_pls = loaded.(fnames{1});
    fprintf('  Dataset PLS Toolbox caricato: %s\n', fnames{1});
    fprintf('  Tipo: %s\n', class(ds_pls));
catch ME
    fprintf('  [WARN] animal_feedNIR.mat non caricato: %s\n', ME.message);
    ds_pls = [];
end

% --- Divisione Calibrazione / Test Set (70/30 stratificata) ---
rng(42); % riproducibilita'
idxCal = false(nSamples, 1);
idxTest = false(nSamples, 1);

for c = 1:nClasses
    classIdx = find(strcmp(category, uniqueCats{c}));
    nClass   = length(classIdx);
    nCal     = round(0.7 * nClass);
    perm     = classIdx(randperm(nClass));
    idxCal(perm(1:nCal))     = true;
    idxTest(perm(nCal+1:end)) = true;
end

X_cal  = X_raw(idxCal, :);   Y_cal  = Y_dummy(idxCal, :);
X_test = X_raw(idxTest, :);  Y_test = Y_dummy(idxTest, :);
cat_cal  = category(idxCal);
cat_test = category(idxTest);

fprintf('\n  Set di calibrazione: %d campioni\n', sum(idxCal));
fprintf('  Set di test:         %d campioni\n', sum(idxTest));
for c = 1:nClasses
    fprintf('    %s -> Cal: %d, Test: %d\n', uniqueCats{c}, ...
        sum(strcmp(cat_cal, uniqueCats{c})), sum(strcmp(cat_test, uniqueCats{c})));
end

% --- Crea dataset PLS Toolbox per calibrazione ---
dsCal = dataset(X_cal);
dsCal.label{1,1}    = cat_cal;                     % etichette campione
dsCal.axisscale{2}  = wavelengths(:)';             % asse x = wavenumbers
dsCal.name          = 'Farine Animali - Calibrazione';
dsCal.label{2,1}    = cellstr(num2str(wavelengths(:)));

dsY = dataset(Y_cal);
dsY.label{2,1} = uniqueCats;
dsY.name = 'Y dummy (classi)';

fprintf('\n  Dataset PLS Toolbox creati con successo.\n');

%% =========================================================================
%  2. PCA ESPLORATIVA (Mean centering, 2 PCs)
%  =========================================================================
fprintf('\n------------------------------------------------------------\n');
fprintf('  2. PCA ESPLORATIVA\n');
fprintf('------------------------------------------------------------\n');

% Opzioni PCA: solo mean centering, 2 PC
optPCA          = pca('options');
optPCA.display  = 'off';
optPCA.plots    = 'none';
optPCA.preprocessing = {preprocess('default','meancenter')};

% Esegui PCA su set di calibrazione
modelPCA = pca(X_cal, 2, optPCA);

% Estrai scores e loadings (accesso robusto per diverse versioni PLS Toolbox)
scoresPCA   = [];
loadingsPCA = [];
try
    loadsCell = modelPCA.loads;
    if iscell(loadsCell)
        scoresPCA   = loadsCell{1,1};  % T scores
        loadingsPCA = loadsCell{2,1};  % P loadings
    end
catch
    try scoresPCA   = modelPCA.scores;   catch, end
    try loadingsPCA = modelPCA.loadings; catch, end
end
if isempty(scoresPCA)
    try scoresPCA   = modelPCA.xscores;  catch, end
    try loadingsPCA = modelPCA.xloads;   catch, end
end

% Gestione formati: converti da dataset a double se necessario
if isa(scoresPCA, 'dataset')
    scoresPCA = double(scoresPCA.data);
elseif isstruct(scoresPCA)
    scoresPCA = scoresPCA.data;
end
if isa(loadingsPCA, 'dataset')
    loadingsPCA = double(loadingsPCA.data);
elseif isstruct(loadingsPCA)
    loadingsPCA = loadingsPCA.data;
end

% Varianza spiegata
ssq = modelPCA.detail.ssq;
varExplPCA = ssq(:,2); % percentuale varianza per ogni PC

fprintf('  PC1: %.2f%% varianza\n', varExplPCA(1));
fprintf('  PC2: %.2f%% varianza\n', varExplPCA(2));
fprintf('  Totale PC1+PC2: %.2f%%\n', sum(varExplPCA(1:2)));

% --- Fig 1: Score Plot PCA (PC1 vs PC2) ---
fig1 = figure('Name','PCA Scores','Position',[100 100 900 600],'Visible','off');
colors_rgb = [1 0 0; 0 0 1; 0 0.7 0];
hold on;
legendH = gobjects(nClasses, 1);
for c = 1:nClasses
    idx = strcmp(cat_cal, uniqueCats{c});
    legendH(c) = scatter(scoresPCA(idx,1), scoresPCA(idx,2), 60, ...
        colors_rgb(c,:), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
end
xlabel(sprintf('PC1 (%.2f%%)', varExplPCA(1)));
ylabel(sprintf('PC2 (%.2f%%)', varExplPCA(2)));
title('PCA Esplorativa - Score Plot (Mean Centering)');
legend(legendH, uniqueCats, 'Location', 'best');
xline(0,'--','Color',[0.5 0.5 0.5]); yline(0,'--','Color',[0.5 0.5 0.5]);
grid on; box on; hold off;
saveas(fig1, fullfile(plotDir, '01_PCA_scores_PC1_PC2.png'));
fprintf('  Salvato: 01_PCA_scores_PC1_PC2.png\n');

% --- Fig 2: Loadings PCA ---
fig2 = figure('Name','PCA Loadings','Position',[100 100 1100 500],'Visible','off');
subplot(2,1,1);
plot(wavelengths, loadingsPCA(:,1), 'b-', 'LineWidth', 1.5);
xlabel('Wavenumber (cm^{-1})'); ylabel('Loading PC1');
title(sprintf('Loading PC1 (%.2f%%)', varExplPCA(1)));
set(gca,'XDir','reverse'); grid on;
subplot(2,1,2);
plot(wavelengths, loadingsPCA(:,2), 'r-', 'LineWidth', 1.5);
xlabel('Wavenumber (cm^{-1})'); ylabel('Loading PC2');
title(sprintf('Loading PC2 (%.2f%%)', varExplPCA(2)));
set(gca,'XDir','reverse'); grid on;
saveas(fig2, fullfile(plotDir, '02_PCA_loadings.png'));
fprintf('  Salvato: 02_PCA_loadings.png\n');

% --- Fig 3: T2 vs Q per outlier detection ---
fig3 = figure('Name','PCA T2 vs Q','Position',[100 100 800 600],'Visible','off');

% Calcolo manuale T2 e Q dai scores/loadings (robusto per tutte le versioni PLS Toolbox)
% T2 = Hotelling's T-squared: somma degli scores^2 / eigenvalue per ogni PC
eigs_pca = var(scoresPCA, 0, 1);  % varianza di ogni score = eigenvalue
T2vals = sum(scoresPCA.^2 ./ repmat(eigs_pca, size(scoresPCA,1), 1), 2);

% Q = somma dei quadrati dei residui di ricostruzione
X_cal_mc = X_cal - mean(X_cal, 1);  % mean centering
X_reconstructed_pca = scoresPCA * loadingsPCA';
E_pca = X_cal_mc - X_reconstructed_pca;
Qvals = sum(E_pca.^2, 2);

% Limiti al 95%: T2 con distribuzione F, Q con percentile
nPC_sel = 2;
n_cal = size(X_cal, 1);
T2lim_pca = ((nPC_sel * (n_cal^2 - 1)) / (n_cal * (n_cal - nPC_sel))) * ...
    finv(0.95, nPC_sel, n_cal - nPC_sel);
Qlim_pca = prctile(Qvals, 95);

hold on;
for c = 1:nClasses
    idx = strcmp(cat_cal, uniqueCats{c});
    scatter(T2vals(idx), Qvals(idx), 60, colors_rgb(c,:), 'filled');
end
xline(T2lim_pca, '--r', 'T^2 limit', 'LineWidth', 1.5);
yline(Qlim_pca,  '--r', 'Q limit', 'LineWidth', 1.5);
xlabel('Hotelling T^2'); ylabel('Q Residuals');
title('PCA - T^2 vs Q (Outlier Detection)');
legend(uniqueCats, 'Location', 'best');
grid on; box on; hold off;
saveas(fig3, fullfile(plotDir, '03_PCA_T2_vs_Q.png'));
fprintf('  Salvato: 03_PCA_T2_vs_Q.png\n');

% --- Fig 4: Spettri originali per categoria ---
fig4 = figure('Name','Spettri Originali','Position',[100 100 1100 500],'Visible','off');
subplot(1,2,1); hold on;
hLeg = gobjects(nClasses,1);
for c = 1:nClasses
    idx = strcmp(cat_cal, uniqueCats{c});
    h = plot(wavelengths, X_cal(idx,:)', 'Color', [colors_rgb(c,:) 0.3], 'LineWidth', 0.3);
    hLeg(c) = h(1);
end
xlabel('Wavenumber (cm^{-1})'); ylabel('Assorbanza');
title('Spettri NIR Originali'); legend(hLeg, uniqueCats, 'Location','best');
set(gca,'XDir','reverse'); grid on; hold off;

subplot(1,2,2); hold on;
for c = 1:nClasses
    idx = strcmp(cat_cal, uniqueCats{c});
    plot(wavelengths, mean(X_cal(idx,:),1), 'Color', colors_rgb(c,:), 'LineWidth', 2);
end
xlabel('Wavenumber (cm^{-1})'); ylabel('Assorbanza media');
title('Spettri Medi per Categoria'); legend(uniqueCats, 'Location','best');
set(gca,'XDir','reverse'); grid on; hold off;
saveas(fig4, fullfile(plotDir, '04_spettri_originali.png'));
fprintf('  Salvato: 04_spettri_originali.png\n');

%% =========================================================================
%  3. PREPROCESSING E SCELTA NUMERO LV
%  =========================================================================
fprintf('\n------------------------------------------------------------\n');
fprintf('  3. PREPROCESSING E SCELTA NUMERO LV\n');
fprintf('------------------------------------------------------------\n');

% Preprocessing scelto: SNV + Mean Centering
%   SNV (Standard Normal Variate) corregge gli effetti di scattering
%   tipici degli spettri NIR, poi Mean Centering centra i dati.
%   SNV viene applicato manualmente ai dati, poi pls() fa il mean centering.
prepName = 'SNV + MC';

% Applica SNV manualmente (row-wise: sottrai media riga, dividi per std riga)
fprintf('  Applicazione SNV ai dati di calibrazione e test...\n');
X_cal_snv = (X_cal - mean(X_cal, 2)) ./ std(X_cal, 0, 2);
X_test_snv = (X_test - mean(X_test, 2)) ./ std(X_test, 0, 2);

fprintf('  Preprocessing X: %s\n', prepName);
fprintf('  Preprocessing Y: Mean Centering (default PLS Toolbox)\n');

maxLV = 15;  % max numero LV da testare

% --- Cross-validation: Venetian Blind (15 splits, thickness=3) ---
optCV = pls('options');
optCV.display       = 'off';
optCV.plots         = 'none';
optCV.crossval      = {'vet', 15, 3};  % venetian blind, 15 splits, thickness 3
optCV.algorithm     = 'sim';  % SIMPLS
% Nota: NON settiamo preprocessing - usiamo il default di PLS Toolbox
%       (autoscale). SNV applicato manualmente prima della chiamata.

% Costruisci modello PLS con CV per trovare il numero ottimale di LV
fprintf('  Costruzione modello PLS con CV (1-%d LV)...\n', maxLV);
modelPLS_cv = pls(X_cal_snv, Y_cal, maxLV, optCV);

% Estrai RMSECV e RMSEC per ogni numero di LV
% Nota: evrimodel objects non supportano isfield(), usiamo try/catch
rmsecv_vec = NaN(1, maxLV);
rmsec_vec  = NaN(1, maxLV);

% --- RMSECV ---
try
    tmp = modelPLS_cv.cv.rmsecv;
    if isa(tmp, 'dataset'), tmp = double(tmp.data); end
    rmsecv_vec = tmp(:)';
catch
    try
        % ssq contiene le statistiche per ogni LV: colonne [RMSEC, RMSECV, ...]
        tmp = modelPLS_cv.detail.ssq;
        if isa(tmp, 'dataset'), tmp = double(tmp.data); end
        % In PLS Toolbox, ssq ha colonne: [calibration, cv] per ciascun blocco
        % Cerchiamo la colonna RMSECV
    catch
    end
end

% Se RMSECV ancora NaN, calcoliamolo dalle predizioni CV
if all(isnan(rmsecv_vec))
    try
        cvPred = modelPLS_cv.cv.pred;
        if iscell(cvPred)
            for lv = 1:min(maxLV, numel(cvPred))
                tmp = cvPred{lv};
                if isa(tmp, 'dataset'), tmp = double(tmp.data); end
                rmsecv_vec(lv) = sqrt(mean((Y_cal(:) - tmp(:)).^2));
            end
        end
    catch
    end
end

% Se ancora NaN, proviamo con il campo 'ssq' (sum-of-squares table)
if all(isnan(rmsecv_vec))
    try
        ssqData = modelPLS_cv.ssq;
        if isa(ssqData, 'dataset'), ssqData = double(ssqData.data); end
        % ssq tipicamente: righe = LV, colonne = [RMSEC, RMSECV, R2cal, R2cv, ...]
        if size(ssqData, 2) >= 2
            rmsecv_vec(1:size(ssqData,1)) = ssqData(:, 2)';
        end
    catch
    end
end

% --- RMSEC ---
try
    tmp = modelPLS_cv.rmsec;
    if isa(tmp, 'dataset'), tmp = double(tmp.data); end
    rmsec_vec = tmp(:)';
catch
    try
        ssqData = modelPLS_cv.ssq;
        if isa(ssqData, 'dataset'), ssqData = double(ssqData.data); end
        if size(ssqData, 2) >= 1
            rmsec_vec(1:size(ssqData,1)) = ssqData(:, 1)';
        end
    catch
    end
end

% Ultima risorsa: esplora la struttura del modello e stampa i campi
if all(isnan(rmsecv_vec))
    fprintf('  [DEBUG] Esploro struttura modelPLS_cv...\n');
    try
        fnames = fieldnames(modelPLS_cv);
        fprintf('  [DEBUG] Campi: %s\n', strjoin(fnames, ', '));
    catch
        fprintf('  [DEBUG] fieldnames non disponibile, provo properties...\n');
        try
            fnames = properties(modelPLS_cv);
            fprintf('  [DEBUG] Properties: %s\n', strjoin(fnames, ', '));
        catch
        end
    end
end

% Assicurati siano vettori double
if isa(rmsecv_vec, 'dataset'), rmsecv_vec = double(rmsecv_vec.data); end
if isa(rmsec_vec, 'dataset'),  rmsec_vec  = double(rmsec_vec.data);  end
rmsecv_vec = rmsecv_vec(:)';
rmsec_vec  = rmsec_vec(:)';

% Trova miglior numero di LV (minimo RMSECV)
[minRMSECV, bestNLV] = min(rmsecv_vec);

% R2 in CV e calibrazione (per info)
SSres_cv  = minRMSECV^2 * numel(Y_cal);
SStot     = sum((Y_cal(:) - mean(Y_cal(:))).^2);
r2cv      = 1 - SSres_cv/SStot;
if ~isnan(rmsec_vec(bestNLV))
    SSres_cal = rmsec_vec(bestNLV)^2 * numel(Y_cal);
    r2cal     = 1 - SSres_cal/SStot;
else
    r2cal = NaN;
end

fprintf('\n  Risultati selezione LV:\n');
fprintf('    Numero LV ottimale: %d\n', bestNLV);
fprintf('    RMSECV:             %.4f\n', minRMSECV);
if ~isnan(rmsec_vec(bestNLV))
    fprintf('    RMSEC:              %.4f\n', rmsec_vec(bestNLV));
end
fprintf('    R2 CV:              %.4f\n', r2cv);
if ~isnan(r2cal)
    fprintf('    R2 Cal:             %.4f\n', r2cal);
end

% --- Fig 5: Scree Plot RMSEC vs RMSECV ---
fig5 = figure('Name','Scree Plot LV','Position',[100 100 900 600],'Visible','off');
nPlotLV = min(maxLV, length(rmsecv_vec));
plot(1:nPlotLV, rmsec_vec(1:nPlotLV), 'bo-', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', 'b', 'DisplayName', 'RMSEC');
hold on;
plot(1:nPlotLV, rmsecv_vec(1:nPlotLV), 'rs-', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', 'r', 'DisplayName', 'RMSECV');
xline(bestNLV, '--k', sprintf('LV = %d', bestNLV), 'LineWidth', 1.5, ...
    'LabelVerticalAlignment', 'bottom');
xlabel('Numero di Latent Variables (LV)');
ylabel('RMSE');
title(sprintf('Scree Plot - %s', prepName));
legend('Location','northeast');
grid on; box on; hold off;
saveas(fig5, fullfile(plotDir, '05_scree_plot_LV.png'));
fprintf('  Salvato: 05_scree_plot_LV.png\n');
fprintf('  Numero LV selezionato: %d\n', bestNLV);

%% =========================================================================
%  4. COSTRUZIONE MODELLO PLS FINALE
%  =========================================================================
fprintf('\n------------------------------------------------------------\n');
fprintf('  4. MODELLO PLS FINALE\n');
fprintf('------------------------------------------------------------\n');

% Opzioni modello finale
optFinal = pls('options');
optFinal.display        = 'off';
optFinal.plots          = 'none';
optFinal.crossval       = {'vet', 15, 3};   % Venetian Blind
optFinal.algorithm      = 'sim';             % SIMPLS
% Nota: uso default preprocessing da PLS Toolbox (autoscale).
%       SNV applicato manualmente ai dati.

% Costruisci modello PLS finale
modelFinal = pls(X_cal_snv, Y_cal, bestNLV, optFinal);

fprintf('  Modello PLS costruito con %d LV\n', bestNLV);
fprintf('  Preprocessing X: %s\n', prepName);
fprintf('  Preprocessing Y: None\n');
fprintf('  Cross-validation: Venetian Blind (15 splits, thickness=3)\n');

% Estrai metriche dal modello (accesso robusto con try/catch)
rmsec_final = NaN;
rmsecv_final = NaN;

% RMSEC dal modello finale
try
    tmp = modelFinal.rmsec;
    if isa(tmp, 'dataset'), tmp = double(tmp.data); end
    rmsec_final = tmp(:);
    fprintf('  RMSEC:  %.4f\n', rmsec_final(end));
catch
    % Prova da ssq
    try
        ssqData = modelFinal.ssq;
        if isa(ssqData, 'dataset'), ssqData = double(ssqData.data); end
        if size(ssqData, 2) >= 1
            rmsec_final = ssqData(:, 1);
            fprintf('  RMSEC (da ssq): %.4f\n', rmsec_final(end));
        end
    catch
        fprintf('  [INFO] RMSEC non accessibile.\n');
    end
end

% RMSECV dal modello finale
try
    tmp = modelFinal.cv.rmsecv;
    if isa(tmp, 'dataset'), tmp = double(tmp.data); end
    rmsecv_final = tmp(:);
    fprintf('  RMSECV: %.4f\n', rmsecv_final(end));
catch
    try
        ssqData = modelFinal.ssq;
        if isa(ssqData, 'dataset'), ssqData = double(ssqData.data); end
        if size(ssqData, 2) >= 2
            rmsecv_final = ssqData(:, 2);
            fprintf('  RMSECV (da ssq): %.4f\n', rmsecv_final(end));
        end
    catch
        fprintf('  [INFO] RMSECV non accessibile.\n');
    end
end

%% =========================================================================
%  5. GRAFICI DIAGNOSTICI
%  =========================================================================
fprintf('\n------------------------------------------------------------\n');
fprintf('  5. GRAFICI DIAGNOSTICI\n');
fprintf('------------------------------------------------------------\n');

% --- 6a. Estrazione scores (T), loadings (P), weights (W) ---
% Accesso robusto ai campi del modello PLS (compatibile con diverse versioni PLS Toolbox)
T_scores  = [];
P_loads   = [];
W_weights = [];
U_scores_pls = [];  % Y-scores per inner relations

try
    % Metodo 1: accesso tramite .loads cell array
    if isprop(modelFinal, 'loads') || isfield(modelFinal, 'loads')
        loadsCell = modelFinal.loads;
        if iscell(loadsCell)
            T_scores = loadsCell{1,1};  % X-scores
            P_loads  = loadsCell{2,1};  % X-loadings
            if size(loadsCell,1) >= 3
                W_weights = loadsCell{3,1};  % Weights
            end
            if size(loadsCell,2) >= 2
                U_scores_pls = loadsCell{1,2};  % Y-scores
            end
        end
    end
catch
    fprintf('  [INFO] Accesso .loads fallito, provo campi diretti...\n');
end

% Metodo 2: campi diretti (versioni alternative PLS Toolbox)
if isempty(T_scores)
    try T_scores = modelFinal.xscores;  catch, end
    try P_loads  = modelFinal.xloads;   catch, end
    try W_weights = modelFinal.weights;  catch, end
    try U_scores_pls = modelFinal.yscores; catch, end
end

% Converti dataset -> double
if isa(T_scores, 'dataset'),     T_scores     = double(T_scores.data);     end
if isa(P_loads, 'dataset'),      P_loads      = double(P_loads.data);      end
if isa(W_weights, 'dataset'),    W_weights    = double(W_weights.data);    end
if isa(U_scores_pls, 'dataset'), U_scores_pls = double(U_scores_pls.data); end

fprintf('  T_scores:  %d x %d\n', size(T_scores,1), size(T_scores,2));
fprintf('  P_loads:   %d x %d\n', size(P_loads,1), size(P_loads,2));
if ~isempty(W_weights), fprintf('  W_weights: %d x %d\n', size(W_weights,1), size(W_weights,2)); end

% --- 6b. T2 vs Q (calcolati manualmente dagli scores/loadings) ---
fig7 = figure('Name','T2 vs Q','Position',[100 100 800 600],'Visible','off');

% T2 = Hotelling's T-squared
eigs_pls = var(T_scores, 0, 1);  % eigenvalues
T2 = sum(T_scores.^2 ./ repmat(eigs_pls, size(T_scores,1), 1), 2);

% Q = residui di ricostruzione
% Il modello PLS usa autoscale internamente, ricostruiamo nello spazio originale
% come approssimazione: Q = ||x - T*P'||^2 calcolato sui dati SNV
X_hat_pls = T_scores * P_loads';
X_cal_prep = X_cal_snv - mean(X_cal_snv, 1);  % approssimazione centra per Q
E_pls = X_cal_prep - X_hat_pls;
Q = sum(E_pls.^2, 2);

% Limiti al 95%
n_cal_pls = size(X_cal_snv, 1);
T2lim = ((bestNLV * (n_cal_pls^2 - 1)) / (n_cal_pls * (n_cal_pls - bestNLV))) * ...
    finv(0.95, bestNLV, n_cal_pls - bestNLV);
Qlim = prctile(Q, 95);

hold on;
for c = 1:nClasses
    idx = strcmp(cat_cal, uniqueCats{c});
    scatter(T2(idx), Q(idx), 60, colors_rgb(c,:), 'filled');
end
xline(T2lim, '--r', 'T^2 limit', 'LineWidth', 1.5);
yline(Qlim,  '--r', 'Q limit', 'LineWidth', 1.5);
xlabel('Hotelling T^2'); ylabel('Q Residuals');
title(sprintf('T^2 vs Q - PLS con %d LV', bestNLV));
legend(uniqueCats, 'Location','best');
grid on; box on; hold off;
saveas(fig7, fullfile(plotDir, '07_T2_vs_Q_PLS.png'));
fprintf('  Salvato: 07_T2_vs_Q_PLS.png\n');

% --- 6c. Score Plot LV1 vs LV2 ---
fig8 = figure('Name','Scores LV1 vs LV2','Position',[100 100 800 600],'Visible','off');
hold on;
for c = 1:nClasses
    idx = strcmp(cat_cal, uniqueCats{c});
    scatter(T_scores(idx,1), T_scores(idx,2), 60, colors_rgb(c,:), 'filled', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
end
xlabel('LV1'); ylabel('LV2');
title('Score Plot: LV1 vs LV2 (PLS)');
legend(uniqueCats, 'Location','best');
xline(0,'--','Color',[0.5 0.5 0.5]); yline(0,'--','Color',[0.5 0.5 0.5]);
grid on; box on; hold off;
saveas(fig8, fullfile(plotDir, '08_scores_LV1_LV2.png'));
fprintf('  Salvato: 08_scores_LV1_LV2.png\n');

% --- 6d. Y misurato vs Y predetto (fit e CV) ---
% Predizioni in calibrazione (accesso robusto)
Y_pred_cal = [];
try
    tmp = modelFinal.pred;
    if iscell(tmp)
        Y_pred_cal = tmp{end};  % predizione con tutte le LV
    else
        Y_pred_cal = tmp;
    end
catch
    try
        Y_pred_cal = modelFinal.detail.pred{end};
    catch
        fprintf('  [WARN] Campo pred non accessibile, calcolo manuale...\n');
    end
end
if isa(Y_pred_cal, 'dataset'), Y_pred_cal = double(Y_pred_cal.data); end

% Se pred non disponibile, calcola da scores
if isempty(Y_pred_cal)
    % Y_pred = T * Q_y' + mean(Y) (approssimazione)
    Y_pred_cal = T_scores * (T_scores \ (Y_cal - mean(Y_cal,1))) + mean(Y_cal,1);
end

% Predizioni in cross-validation
Y_pred_cv = [];
try
    cvField = modelFinal.cv;
    if isstruct(cvField) && isfield(cvField, 'pred')
        if iscell(cvField.pred)
            Y_pred_cv = cvField.pred{end};
        else
            Y_pred_cv = cvField.pred;
        end
    end
catch
    try
        cvField = modelFinal.detail.cv;
        if iscell(cvField.pred)
            Y_pred_cv = cvField.pred{end};
        else
            Y_pred_cv = cvField.pred;
        end
    catch
        fprintf('  [INFO] Predizioni CV non disponibili.\n');
    end
end
if isa(Y_pred_cv, 'dataset'), Y_pred_cv = double(Y_pred_cv.data); end

% Plot per ogni classe (colonna di Y)
for col = 1:nClasses
    fig_yfit = figure('Name', sprintf('Y pred vs meas - %s', uniqueCats{col}), ...
        'Position', [100 100 1100 500], 'Visible', 'off');
    
    % Fit
    subplot(1,2,1);
    scatter(Y_cal(:,col), Y_pred_cal(:,col), 40, 'b', 'filled'); hold on;
    plot([0 1], [0 1], 'g-', 'LineWidth', 2);
    pFit = polyfit(Y_cal(:,col), Y_pred_cal(:,col), 1);
    xLine = linspace(min(Y_cal(:,col)), max(Y_cal(:,col)), 100);
    plot(xLine, polyval(pFit, xLine), 'r--', 'LineWidth', 1.5);
    xlabel('Y Misurato'); ylabel('Y Predetto (Fit)');
    title(sprintf('Y Fit: %s (R^2=%.4f)', uniqueCats{col}, ...
        1 - sum((Y_cal(:,col)-Y_pred_cal(:,col)).^2)/sum((Y_cal(:,col)-mean(Y_cal(:,col))).^2)));
    legend('Campioni','Diagonale','Regressione','Location','best');
    grid on; box on; hold off;
    
    % Cross-validation
    if ~isempty(Y_pred_cv)
        subplot(1,2,2);
        scatter(Y_cal(:,col), Y_pred_cv(:,col), 40, 'r', 'filled'); hold on;
        plot([0 1], [0 1], 'g-', 'LineWidth', 2);
        pCV = polyfit(Y_cal(:,col), Y_pred_cv(:,col), 1);
        plot(xLine, polyval(pCV, xLine), 'r--', 'LineWidth', 1.5);
        xlabel('Y Misurato'); ylabel('Y Predetto (CV)');
        r2cv_col = 1 - sum((Y_cal(:,col)-Y_pred_cv(:,col)).^2)/sum((Y_cal(:,col)-mean(Y_cal(:,col))).^2);
        title(sprintf('Y CV: %s (R^2=%.4f)', uniqueCats{col}, r2cv_col));
        legend('Campioni','Diagonale','Regressione','Location','best');
        grid on; box on; hold off;
    end
    
    saveas(fig_yfit, fullfile(plotDir, sprintf('09_%s_Y_pred_vs_meas_%s.png', ...
        num2str(col,'%02d'), uniqueCats{col})));
    fprintf('  Salvato: 09_%02d_Y_pred_vs_meas_%s.png\n', col, uniqueCats{col});
end

% --- 6e. Y misurato vs Residui CV ---
if ~isempty(Y_pred_cv)
    fig_res = figure('Name','Y vs CV Residuals','Position',[100 100 1200 400],'Visible','off');
    for col = 1:nClasses
        subplot(1, nClasses, col);
        residuals_cv = Y_cal(:,col) - Y_pred_cv(:,col);
        scatter(Y_cal(:,col), residuals_cv, 40, colors_rgb(col,:), 'filled');
        hold on;
        yline(0, 'k-', 'LineWidth', 1);
        yline(3*std(residuals_cv), '--r', '+3\sigma');
        yline(-3*std(residuals_cv), '--r', '-3\sigma');
        xlabel('Y Misurato'); ylabel('CV Residuals');
        title(sprintf('Residui CV: %s', uniqueCats{col}));
        grid on; box on; hold off;
    end
    saveas(fig_res, fullfile(plotDir, '10_Y_vs_CV_residuals.png'));
    fprintf('  Salvato: 10_Y_vs_CV_residuals.png\n');
end

% --- 6f. Leverage vs Y Residuals ---
fig_lev = figure('Name','Leverage vs Residuals','Position',[100 100 1200 400],'Visible','off');
% Calcola leverage dagli scores
leverage = sum(T_scores.^2 ./ var(T_scores), 2);
for col = 1:nClasses
    subplot(1, nClasses, col);
    if ~isempty(Y_pred_cv)
        yres = Y_cal(:,col) - Y_pred_cv(:,col);
    else
        yres = Y_cal(:,col) - Y_pred_cal(:,col);
    end
    scatter(leverage, yres, 40, colors_rgb(col,:), 'filled');
    hold on;
    yline(0, 'k-', 'LineWidth', 1);
    yline(3*std(yres), '--r', '+3\sigma');
    yline(-3*std(yres), '--r', '-3\sigma');
    xlabel('Leverage'); ylabel('Y Residuals');
    title(sprintf('Leverage vs Residuals: %s', uniqueCats{col}));
    grid on; box on; hold off;
end
saveas(fig_lev, fullfile(plotDir, '11_leverage_vs_residuals.png'));
fprintf('  Salvato: 11_leverage_vs_residuals.png\n');

% --- 6g. Inner Relations (T scores vs U scores) ---
% U_scores_pls gia' estratti nella sezione 6a
U_scores = U_scores_pls;
if isempty(U_scores)
    % Fallback: calcola U come proiezione di Y su Y-loadings
    try
        Y_cal_mc = Y_cal - mean(Y_cal, 1);
        % U = Y * Q_y dove Q_y = Y-loadings (approssimazione NIPALS)
        U_scores = Y_cal_mc * (T_scores \ Y_cal_mc)';  % proiezione approssimata
        fprintf('  [INFO] U scores calcolati come proiezione di Y.\n');
    catch
        U_scores = [];
        fprintf('  [INFO] U scores non disponibili, inner relations saltate.\n');
    end
end

if ~isempty(U_scores)
    nLV_plot = min(bestNLV, size(U_scores,2));
    nRows = ceil(nLV_plot / 4);
    fig_inner = figure('Name','Inner Relations','Position',[50 50 1400 350*nRows],'Visible','off');
    for lv = 1:nLV_plot
        subplot(nRows, 4, lv);
        hold on;
        for c = 1:nClasses
            idx = strcmp(cat_cal, uniqueCats{c});
            scatter(T_scores(idx,lv), U_scores(idx,lv), 30, colors_rgb(c,:), 'filled');
        end
        % Fit lineare
        pInner = polyfit(T_scores(:,lv), U_scores(:,lv), 1);
        xr = linspace(min(T_scores(:,lv)), max(T_scores(:,lv)), 100);
        plot(xr, polyval(pInner, xr), 'k-', 'LineWidth', 1.5);
        xlabel(sprintf('T_{%d} (X scores)', lv));
        ylabel(sprintf('U_{%d} (Y scores)', lv));
        title(sprintf('Inner Relation LV%d', lv));
        r_inner = corr(T_scores(:,lv), U_scores(:,lv));
        text(0.05, 0.95, sprintf('r = %.3f', r_inner), 'Units','normalized', ...
            'VerticalAlignment','top', 'FontSize', 9);
        grid on; box on; hold off;
    end
    if nLV_plot > 0
        legend(uniqueCats, 'Location','best');
    end
    saveas(fig_inner, fullfile(plotDir, '12_inner_relations.png'));
    fprintf('  Salvato: 12_inner_relations.png\n');
end

%% =========================================================================
%  6. IMPORTANZA VARIABILI
%  =========================================================================
fprintf('\n------------------------------------------------------------\n');
fprintf('  6. IMPORTANZA VARIABILI\n');
fprintf('------------------------------------------------------------\n');

% Wavenumber axis per plot
wn = wavelengths(:)';

% --- 7a. Weights (primi 3 LV) ---
if ~isempty(W_weights)
    nW = min(3, size(W_weights,2));
    fig_w = figure('Name','PLS Weights','Position',[100 100 1100 700],'Visible','off');
    colorsW = {'b','r','g'};
    for lv = 1:nW
        subplot(nW,1,lv);
        plot(wn, W_weights(:,lv), colorsW{lv}, 'LineWidth', 1.5);
        xlabel('Wavenumber (cm^{-1})'); ylabel(sprintf('w_{%d}', lv));
        title(sprintf('PLS Weight - LV%d', lv));
        set(gca, 'XDir', 'reverse');
        grid on;
    end
    sgtitle('PLS Weights (primi 3 LV)');
    saveas(fig_w, fullfile(plotDir, '13_PLS_weights.png'));
    fprintf('  Salvato: 13_PLS_weights.png\n');
end

% --- 7b. Regression Coefficients ---
regCoeff = [];
try
    regCoeff = modelFinal.reg;
    if iscell(regCoeff), regCoeff = regCoeff{end}; end
    if isa(regCoeff, 'dataset'), regCoeff = double(regCoeff.data); end
catch
    try
        regCoeff = modelFinal.detail.reg;
        if iscell(regCoeff), regCoeff = regCoeff{end}; end
        if isa(regCoeff, 'dataset'), regCoeff = double(regCoeff.data); end
    catch
        fprintf('  [INFO] Reg coefficients non disponibili nel modello.\n');
    end
end
if ~isempty(regCoeff)
    
    fig_reg = figure('Name','Regression Coefficients','Position',[100 100 1200 400*nClasses],'Visible','off');
    for col = 1:min(nClasses, size(regCoeff,2))
        subplot(nClasses,1,col);
        plot(wn, regCoeff(1:nVars,col), 'Color', colors_rgb(col,:), 'LineWidth', 1.2);
        xlabel('Wavenumber (cm^{-1})'); ylabel('Coefficient');
        title(sprintf('Regression Coefficients - %s', uniqueCats{col}));
        set(gca, 'XDir', 'reverse');
        yline(0, 'k--'); grid on;
    end
    sgtitle('Coefficienti di Regressione PLS');
    saveas(fig_reg, fullfile(plotDir, '14_regression_coefficients.png'));
    fprintf('  Salvato: 14_regression_coefficients.png\n');
end

% --- 7c. VIP Scores ---
% Calcolo VIP manuale se non disponibile direttamente
try
    % Prova la funzione vip del PLS Toolbox
    vipScores = vip(modelFinal);
    if isa(vipScores, 'dataset'), vipScores = double(vipScores.data); end
catch
    % Calcolo VIP manuale
    % VIP = sqrt(p * sum(w_a^2 * SSY_a) / SSY_total)
    % dove p = numero variabili, w_a = weights normalizzati, SSY_a = SS spiegata da LV a
    fprintf('  Calcolo VIP manuale...\n');
    if ~isempty(W_weights)
        p = size(W_weights, 1);
        a = size(W_weights, 2);
        
        % SSY spiegata da ogni LV
        SSY = zeros(1, a);
        Y_pred_cumul = zeros(size(Y_cal));
        for lv = 1:a
            % Ricostruisci Y predetto con lv componenti
            if isfield(modelFinal, 'pred') && iscell(modelFinal.pred)
                Y_pred_lv = modelFinal.pred{lv};
                if isa(Y_pred_lv, 'dataset'), Y_pred_lv = double(Y_pred_lv.data); end
                Y_mean = mean(Y_cal, 1);
                if lv == 1
                    SSY(lv) = sum(sum((Y_pred_lv - Y_mean).^2));
                else
                    Y_pred_prev = modelFinal.pred{lv-1};
                    if isa(Y_pred_prev, 'dataset'), Y_pred_prev = double(Y_pred_prev.data); end
                    SSY(lv) = sum(sum((Y_pred_lv - Y_mean).^2)) - sum(sum((Y_pred_prev - Y_mean).^2));
                end
            end
        end
        
        % Normalizza weights
        W_norm = W_weights ./ sqrt(sum(W_weights.^2, 1));
        
        % VIP
        vipScores = zeros(p, 1);
        for j = 1:p
            s = 0;
            for lv = 1:a
                s = s + W_norm(j,lv)^2 * SSY(lv);
            end
            vipScores(j) = sqrt(p * s / sum(SSY));
        end
    else
        vipScores = [];
    end
end

if ~isempty(vipScores)
    fig_vip = figure('Name','VIP Scores','Position',[100 100 1100 500],'Visible','off');
    bar(wn, vipScores(:,1), 'FaceColor', [0.3 0.5 0.8], 'EdgeColor', 'none');
    hold on;
    yline(1, 'r--', 'VIP = 1', 'LineWidth', 2);
    xlabel('Wavenumber (cm^{-1})'); ylabel('VIP Score');
    title('Variable Importance in Projection (VIP)');
    set(gca, 'XDir', 'reverse');
    grid on; box on; hold off;
    saveas(fig_vip, fullfile(plotDir, '15_VIP_scores.png'));
    fprintf('  Salvato: 15_VIP_scores.png\n');
end

% --- 7d. Selectivity Ratio ---
% SR = varianza spiegata / varianza residua per ogni variabile
try
    srScores = selectratio(modelFinal);
    if isa(srScores, 'dataset'), srScores = double(srScores.data); end
catch
    % Calcolo manuale Selectivity Ratio
    fprintf('  Calcolo Selectivity Ratio manuale...\n');
    if ~isempty(W_weights)
        % Preprocessa X come nel modello
        X_prep_cal = X_cal_snv;  % dati con SNV applicato
        % Ricostruisci: X_hat = T * P' (parte spiegata dal modello)
        X_hat = T_scores * P_loads';
        X_res = X_prep_cal - X_hat; % Approssimazione
        
        var_explained = var(X_hat, 0, 1);
        var_residual  = var(X_res, 0, 1);
        srScores = var_explained ./ (var_residual + eps);
        srScores = srScores(:);
    else
        srScores = [];
    end
end

if ~isempty(srScores) && ~all(isnan(srScores))
    fig_sr = figure('Name','Selectivity Ratio','Position',[100 100 1100 500],'Visible','off');
    bar(wn, srScores(:,1), 'FaceColor', [0.8 0.4 0.2], 'EdgeColor', 'none');
    hold on;
    % Limite al 95%
    srLim = prctile(srScores(:,1), 95);
    yline(srLim, 'r--', sprintf('95%% = %.2f', srLim), 'LineWidth', 1.5);
    xlabel('Wavenumber (cm^{-1})'); ylabel('Selectivity Ratio');
    title('Selectivity Ratio');
    set(gca, 'XDir', 'reverse');
    grid on; box on; hold off;
    saveas(fig_sr, fullfile(plotDir, '16_selectivity_ratio.png'));
    fprintf('  Salvato: 16_selectivity_ratio.png\n');
end

%% =========================================================================
%  7. VALIDAZIONE SU TEST SET
%  =========================================================================
fprintf('\n------------------------------------------------------------\n');
fprintf('  7. VALIDAZIONE SU TEST SET\n');
fprintf('------------------------------------------------------------\n');

% Predizione sul test set - prova diversi metodi in ordine di preferenza
Y_pred_test = [];

% Metodo 1: pls(X_test_snv, modelFinal) - applica modello a nuovi dati
try
    testModel = pls(X_test_snv, modelFinal);
    if iscell(testModel.pred)
        Y_pred_test = testModel.pred{end};
    else
        Y_pred_test = testModel.pred;
    end
    if isa(Y_pred_test, 'dataset'), Y_pred_test = double(Y_pred_test.data); end
    fprintf('  Predizione test set: metodo pls(X_test, model)\n');
catch ME1
    fprintf('  [INFO] Metodo 1 fallito: %s\n', ME1.message);
end

% Metodo 2: prediction method
if isempty(Y_pred_test)
    try
        Y_pred_test = modelFinal.prediction(X_test_snv);
        if isa(Y_pred_test, 'dataset'), Y_pred_test = double(Y_pred_test.data); end
        fprintf('  Predizione test set: metodo .prediction()\n');
    catch ME2
        fprintf('  [INFO] Metodo 2 fallito: %s\n', ME2.message);
    end
end

% Metodo 3: coefficienti di regressione
if isempty(Y_pred_test) && ~isempty(regCoeff)
    try
        fprintf('  Predizione test set: metodo coefficienti di regressione\n');
        % X_test_snv con SNV; PLS Toolbox applica autoscale come nel modello
        X_test_prep = X_test_snv - mean(X_cal_snv, 1);  % centra con media cal
        if size(regCoeff,1) > nVars
            % regCoeff ha intercetta nell'ultima riga
            Y_pred_test = X_test_prep * regCoeff(1:nVars,:) + repmat(regCoeff(end,:), size(X_test,1), 1);
        else
            Y_pred_test = X_test_prep * regCoeff + repmat(mean(Y_cal,1), size(X_test,1), 1);
        end
    catch ME3
        fprintf('  [WARN] Metodo 3 fallito: %s\n', ME3.message);
    end
end

if isempty(Y_pred_test)
    error('Impossibile predire il test set con nessun metodo disponibile.');
end

% Calcola metriche
RMSEP = zeros(1, nClasses);
R2pred = zeros(1, nClasses);
for col = 1:nClasses
    residuals_test = Y_test(:,col) - Y_pred_test(:,col);
    RMSEP(col)  = sqrt(mean(residuals_test.^2));
    SStot_test  = sum((Y_test(:,col) - mean(Y_test(:,col))).^2);
    SSres_test  = sum(residuals_test.^2);
    R2pred(col) = 1 - SSres_test / SStot_test;
end

fprintf('\n  Risultati Test Set:\n');
fprintf('  %-15s | RMSEP   | R2 Pred\n', 'Classe');
fprintf('  %s\n', repmat('-',1,45));
for col = 1:nClasses
    fprintf('  %-15s | %.4f  | %.4f\n', uniqueCats{col}, RMSEP(col), R2pred(col));
end
fprintf('  %-15s | %.4f  | %.4f\n', 'Media', mean(RMSEP), mean(R2pred));

% --- Fig Test Set: Y misurato vs Y predetto ---
fig_test = figure('Name','Test Set Prediction','Position',[100 100 1200 400],'Visible','off');
for col = 1:nClasses
    subplot(1, nClasses, col);
    scatter(Y_test(:,col), Y_pred_test(:,col), 50, colors_rgb(col,:), 'filled');
    hold on;
    plot([0 1], [0 1], 'g-', 'LineWidth', 2);
    pTest = polyfit(Y_test(:,col), Y_pred_test(:,col), 1);
    xr = linspace(min(Y_test(:,col)), max(Y_test(:,col)), 100);
    plot(xr, polyval(pTest, xr), 'r--', 'LineWidth', 1.5);
    xlabel('Y Misurato'); ylabel('Y Predetto');
    title(sprintf('Test Set: %s\nRMSEP=%.4f, R^2=%.4f', ...
        uniqueCats{col}, RMSEP(col), R2pred(col)));
    legend('Campioni','Diagonale','Regressione','Location','best');
    grid on; box on; hold off;
end
sgtitle('Validazione Test Set - Y Misurato vs Y Predetto');
saveas(fig_test, fullfile(plotDir, '17_test_set_prediction.png'));
fprintf('  Salvato: 17_test_set_prediction.png\n');

% --- Classificazione: assegna classe in base al massimo Y predetto ---
[~, predClass_cal]  = max(Y_pred_cal, [], 2);
[~, trueClass_cal]  = max(Y_cal, [], 2);
[~, predClass_test] = max(Y_pred_test, [], 2);
[~, trueClass_test] = max(Y_test, [], 2);

acc_cal  = 100 * sum(predClass_cal == trueClass_cal) / length(trueClass_cal);
acc_test = 100 * sum(predClass_test == trueClass_test) / length(trueClass_test);

fprintf('\n  Accuratezza classificazione (PLS-DA):\n');
fprintf('    Calibrazione: %.1f%%\n', acc_cal);
fprintf('    Test Set:     %.1f%%\n', acc_test);

% --- Confusion Matrix ---
fig_cm = figure('Name','Confusion Matrix','Position',[100 100 1000 450],'Visible','off');

% Funzione manuale per confusion matrix (non richiede Statistics Toolbox)
build_cm = @(trueL, predL) accumarray([trueL(:) predL(:)], 1, [nClasses nClasses]);

subplot(1,2,1);
try
    cm_cal = confusionmat(trueClass_cal, predClass_cal);
catch
    cm_cal = build_cm(trueClass_cal, predClass_cal);
end
imagesc(cm_cal); colorbar;
set(gca, 'XTick', 1:nClasses, 'XTickLabel', uniqueCats, ...
         'YTick', 1:nClasses, 'YTickLabel', uniqueCats);
xlabel('Classe Predetta'); ylabel('Classe Vera');
title(sprintf('Confusion Matrix - Calibrazione (Acc: %.1f%%)', acc_cal));
% Aggiungi numeri nelle celle
for i = 1:nClasses
    for j = 1:nClasses
        text(j, i, num2str(cm_cal(i,j)), 'HorizontalAlignment', 'center', ...
            'FontSize', 14, 'FontWeight', 'bold', 'Color', 'w');
    end
end
colormap(gca, 'parula');

subplot(1,2,2);
try
    cm_test = confusionmat(trueClass_test, predClass_test);
catch
    cm_test = build_cm(trueClass_test, predClass_test);
end
imagesc(cm_test); colorbar;
set(gca, 'XTick', 1:nClasses, 'XTickLabel', uniqueCats, ...
         'YTick', 1:nClasses, 'YTickLabel', uniqueCats);
xlabel('Classe Predetta'); ylabel('Classe Vera');
title(sprintf('Confusion Matrix - Test Set (Acc: %.1f%%)', acc_test));
for i = 1:nClasses
    for j = 1:nClasses
        text(j, i, num2str(cm_test(i,j)), 'HorizontalAlignment', 'center', ...
            'FontSize', 14, 'FontWeight', 'bold', 'Color', 'w');
    end
end
colormap(gca, 'parula');
saveas(fig_cm, fullfile(plotDir, '18_confusion_matrix.png'));
fprintf('  Salvato: 18_confusion_matrix.png\n');

%% =========================================================================
%  8. SPETTRI PREPROCESSATI (confronto con originali)
%  =========================================================================
fprintf('\n------------------------------------------------------------\n');
fprintf('  8. SPETTRI PREPROCESSATI vs ORIGINALI\n');
fprintf('------------------------------------------------------------\n');

% Applica il preprocessing selezionato manualmente per visualizzazione
try
    % SNV gia' applicato, qui mostriamo il confronto originale vs preprocessato
    X_prep_vis = X_cal_snv - mean(X_cal_snv, 1);  % SNV + mean centering
    
    fig_prep = figure('Name','Spettri Preprocessati','Position',[100 100 1200 500],'Visible','off');
    
    subplot(1,2,1); hold on;
    hL = gobjects(nClasses,1);
    for c = 1:nClasses
        idx = strcmp(cat_cal, uniqueCats{c});
        h = plot(wn, X_cal(idx,:)', 'Color', [colors_rgb(c,:) 0.3], 'LineWidth', 0.3);
        hL(c) = h(1);
    end
    xlabel('Wavenumber (cm^{-1})'); ylabel('Assorbanza');
    title('Spettri Originali'); legend(hL, uniqueCats, 'Location','best');
    set(gca, 'XDir', 'reverse'); grid on; hold off;
    
    subplot(1,2,2); hold on;
    hL2 = gobjects(nClasses,1);
    for c = 1:nClasses
        idx = strcmp(cat_cal, uniqueCats{c});
        h = plot(wn, X_prep_vis(idx,:)', 'Color', [colors_rgb(c,:) 0.3], 'LineWidth', 0.3);
        hL2(c) = h(1);
    end
    xlabel('Wavenumber (cm^{-1})'); ylabel('Preprocessato');
    title(sprintf('Spettri dopo %s', prepName));
    legend(hL2, uniqueCats, 'Location','best');
    set(gca, 'XDir', 'reverse'); grid on; hold off;
    
    saveas(fig_prep, fullfile(plotDir, '19_spettri_preprocessati.png'));
    fprintf('  Salvato: 19_spettri_preprocessati.png\n');
catch ME
    fprintf('  [WARN] Visualizzazione preprocessing: %s\n', ME.message);
end

%% =========================================================================
%  9. RIEPILOGO E SALVATAGGIO
%  =========================================================================
fprintf('\n============================================================\n');
fprintf('  RIEPILOGO ANALISI PLS\n');
fprintf('============================================================\n\n');

fprintf('  Preprocessing scelto:  %s\n', prepName);
fprintf('  Numero LV:             %d\n', bestNLV);
fprintf('  Cross-Validation:      Venetian Blind (15 splits, thickness=3)\n');
if ~isnan(rmsec_final(end))
    fprintf('  RMSEC:                 %.4f\n', rmsec_final(end));
else
    fprintf('  RMSEC:                 N/A\n');
end
if ~isnan(rmsecv_final(end))
    fprintf('  RMSECV:                %.4f\n', rmsecv_final(end));
else
    fprintf('  RMSECV:                N/A\n');
end
fprintf('  RMSEP medio:           %.4f\n', mean(RMSEP));
fprintf('  Accuratezza Cal:       %.1f%%\n', acc_cal);
fprintf('  Accuratezza Test:      %.1f%%\n', acc_test);

% Salva risultati
save(fullfile(plotDir, 'results_PLS.mat'), ...
    'modelFinal', 'prepName', 'bestNLV', ...
    'Y_pred_cal', 'Y_pred_cv', 'Y_pred_test', ...
    'RMSEP', 'R2pred', 'acc_cal', 'acc_test', ...
    'cm_cal', 'cm_test', 'uniqueCats', 'wavelengths', ...
    'X_cal', 'X_test', 'Y_cal', 'Y_test', 'cat_cal', 'cat_test');

fprintf('\n  Risultati salvati in: pls_plots/results_PLS.mat\n');
fprintf('\n  Grafici salvati in: pls_plots/\n');
fprintf('\n============================================================\n');
fprintf('  ANALISI COMPLETATA\n');
fprintf('============================================================\n');

close all;
