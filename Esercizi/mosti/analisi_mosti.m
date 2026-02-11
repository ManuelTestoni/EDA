%% ========================================================================
%  ANALISI CHEMOMETRICA COMPLETA - MOSTI DI UVA
%  ========================================================================
%  Autore:  Consulente Chemometrico
%  Data:    Febbraio 2026
%  Scopo:   Analisi esplorativa (PCA) e classificazione (SIMCA, PLS-DA)
%           dei mosti di uva per varietale (5 classi, 2 annate)
%  Tools:   MATLAB + PLS_Toolbox (Eigenvector Research)
%  ========================================================================
%  APPROCCIO METODOLOGICO:
%  1) PCA esplorativa con autoscaling per identificare pattern varietali
%     e valutare l'effetto annata vs. varietale.
%  2) Split stratificato 70/30 (train/test) con seed fisso.
%  3) SIMCA: modello locale per classe, PC ottimizzate via CV.
%  4) PLS-DA: modello discriminante globale, LV ottimizzate via CV.
%  ========================================================================

clear; close all; clc;
warning('off', 'all');  % Evita warning grafici PLS_Toolbox

%% ========================================================================
%  SEZIONE 0 — CONFIGURAZIONE PATHS E CARTELLA OUTPUT
%  ========================================================================
% Cartella dove si trova il .mat
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end
cd(scriptDir);

% Crea cartella plots se non esiste
plotDir = fullfile(scriptDir, 'plots');
if ~exist(plotDir, 'dir')
    mkdir(plotDir);
end

% Funzione helper per salvare i grafici in PNG
savePlot = @(fig, name) print(fig, fullfile(plotDir, name), '-dpng', '-r300');

fprintf('=== Cartella di lavoro: %s ===\n', scriptDir);
fprintf('=== Grafici salvati in: %s ===\n\n', plotDir);

%% ========================================================================
%  SEZIONE 1 — IMPORTAZIONE DATI
%  ========================================================================
fprintf('--- SEZIONE 1: Caricamento dati ---\n');

load('mosti.mat');

% Verifica variabili caricate
% mosti         -> matrice 98 x 6 (dati HPLC)
% nameobj_mosti -> etichette campioni (cell array o char)
% namevar_mosti -> etichette variabili
% classid_v     -> vettore classi varietali (numerico)

[nCamp, nVar] = size(mosti);
fprintf('Campioni: %d | Variabili: %d\n', nCamp, nVar);

% --- Conversione etichette in cell array se necessario ---
if ischar(nameobj_mosti)
    nameobj_mosti = cellstr(nameobj_mosti);
end
if ischar(namevar_mosti)
    namevar_mosti = cellstr(namevar_mosti);
end
% Rimuovi spazi superflui
nameobj_mosti = strtrim(nameobj_mosti);
namevar_mosti = strtrim(namevar_mosti);

% --- Estrazione etichette di classe testuale ---
% Le classi numeriche in classid_v corrispondono a:
% 1=Ancellotta, 2=Montepulciano, 3=Lambrusco Pugliese, 4=Sangiovese, 5=Nero d'Avola
nomiClassi = {'Ancellotta','Montepulciano','Lambrusco Pugn.','Sangiovese','Nero d''Avola'};
sigleClassi = {'A','M','LP','S','N'};
nClassi = length(nomiClassi);

% Estrazione annata dal nome campione (_00 -> 2000, _01 -> 2001)
annata = zeros(nCamp, 1);
for i = 1:nCamp
    nome = nameobj_mosti{i};
    if contains(nome, '_00') || endsWith(nome, '00')
        annata(i) = 2000;
    elseif contains(nome, '_01') || endsWith(nome, '01')
        annata(i) = 2001;
    else
        annata(i) = 0; % Non identificata
    end
end

fprintf('Classi varietali uniche: ');
fprintf('%s ', sigleClassi{:});
fprintf('\n');
for k = 1:nClassi
    n00 = sum(classid_v == k & annata == 2000);
    n01 = sum(classid_v == k & annata == 2001);
    fprintf('  %s: %d campioni (2000: %d, 2001: %d)\n', ...
        nomiClassi{k}, sum(classid_v == k), n00, n01);
end

%% ========================================================================
%  SEZIONE 1.5 — ANALISI ESPLORATIVA DEI DATI (EDA)
%  ========================================================================
fprintf('\n--- SEZIONE 1.5: Analisi Esplorativa dei Dati (EDA) ---\n');

% ---- 1.5.1 FREQUENCY HISTOGRAM ----
% Istogrammi di frequenza per ogni variabile
fprintf('  Generazione istogrammi di frequenza...\n');
fig_hist = figure('Visible','off','Position',[100 100 1200 800]);
for j = 1:nVar
    subplot(2, 3, j);
    histogram(mosti(:, j), 15, 'FaceColor', [0.3 0.6 0.8], 'EdgeColor', 'k');
    xlabel(namevar_mosti{j});
    ylabel('Frequenza');
    title(['Histogram - ' namevar_mosti{j}]);
    grid on;
end
sgtitle('Frequency Histograms - Tutte le Variabili');
savePlot(fig_hist, 'EDA_01_histograms');
fprintf('  Salvato: EDA_01_histograms.png\n');

% ---- 1.5.2 BOX PLOT ----
% Box plot per tutte le variabili (per confrontare distribuzioni)
fprintf('  Generazione box plot...\n');
fig_box1 = figure('Visible','off','Position',[100 100 1000 600]);
boxplot(mosti, 'Labels', namevar_mosti);
ylabel('Valore');
title('Box Plot - Confronto tra Variabili');
grid on;
xtickangle(45);
savePlot(fig_box1, 'EDA_02_boxplot_variabili');
fprintf('  Salvato: EDA_02_boxplot_variabili.png\n');

% Box plot per ogni classe varietale (media delle variabili)
fig_box2 = figure('Visible','off','Position',[100 100 1200 800]);
for j = 1:nVar
    subplot(2, 3, j);
    data_per_classe = [];
    gruppi = [];
    for k = 1:nClassi
        idx = classid_v == k;
        data_per_classe = [data_per_classe; mosti(idx, j)];
        gruppi = [gruppi; k * ones(sum(idx), 1)];
    end
    boxplot(data_per_classe, gruppi);
    set(gca, 'XTickLabel', sigleClassi);
    ylabel(namevar_mosti{j});
    title(['Box Plot - ' namevar_mosti{j}]);
    grid on;
end
sgtitle('Box Plot per Classe Varietale');
savePlot(fig_box2, 'EDA_03_boxplot_per_classe');
fprintf('  Salvato: EDA_03_boxplot_per_classe.png\n');

% ---- 1.5.3 SCATTER PLOT ----
% Scatter plot matrice (tutte le coppie di variabili)
fprintf('  Generazione scatter plot matrix...\n');
fig_scatter1 = figure('Visible','off','Position',[100 100 1200 1000]);
% Colori per le classi
colori_scatter = [0.894 0.102 0.110;   % rosso
                  0.216 0.494 0.722;   % blu
                  0.302 0.686 0.290;   % verde
                  1.000 0.498 0.000;   % arancio
                  0.596 0.306 0.639];  % viola

% Creiamo colori per ogni campione basati sulla classe
colori_campioni = colori_scatter(classid_v, :);

% Scatter plot matrix usando gplotmatrix
[~, ax] = gplotmatrix(mosti, [], classid_v, colori_scatter, '.o', 8, ...
                      'on', '', namevar_mosti, namevar_mosti);
sgtitle('Scatter Plot Matrix - Variabili per Classe');
savePlot(fig_scatter1, 'EDA_04_scatter_matrix');
fprintf('  Salvato: EDA_04_scatter_matrix.png\n');

% Scatter plot specifici per alcune coppie di variabili rilevanti
fig_scatter2 = figure('Visible','off','Position',[100 100 1200 400]);
% Selezioniamo 3 coppie di variabili interessanti
coppie = [1 2; 3 4; 2 5];  % Indici delle coppie di variabili
for p = 1:size(coppie, 1)
    subplot(1, 3, p);
    hold on;
    for k = 1:nClassi
        idx = classid_v == k;
        scatter(mosti(idx, coppie(p,1)), mosti(idx, coppie(p,2)), 60, ...
                colori_scatter(k,:), 'filled', 'MarkerEdgeColor', 'k');
    end
    xlabel(namevar_mosti{coppie(p,1)});
    ylabel(namevar_mosti{coppie(p,2)});
    title([namevar_mosti{coppie(p,1)} ' vs ' namevar_mosti{coppie(p,2)}]);
    legend(nomiClassi, 'Location', 'best', 'FontSize', 7);
    grid on;
    hold off;
end
sgtitle('Scatter Plots - Coppie di Variabili Selezionate');
savePlot(fig_scatter2, 'EDA_05_scatter_selected');
fprintf('  Salvato: EDA_05_scatter_selected.png\n');

% ---- 1.5.4 LINE PLOT ----
% Line plot: profilo medio per classe (tutte le variabili)
fprintf('  Generazione line plot...\n');
fig_line1 = figure('Visible','off','Position',[100 100 1000 600]);
hold on;
for k = 1:nClassi
    idx = classid_v == k;
    media_classe = mean(mosti(idx, :), 1);
    plot(1:nVar, media_classe, '-o', 'Color', colori_scatter(k,:), ...
         'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colori_scatter(k,:));
end
xlabel('Variabile');
ylabel('Valore Medio');
title('Line Plot - Profili Medi per Classe Varietale');
set(gca, 'XTick', 1:nVar, 'XTickLabel', namevar_mosti);
xtickangle(45);
legend(nomiClassi, 'Location', 'best');
grid on;
hold off;
savePlot(fig_line1, 'EDA_06_line_plot_medie');
fprintf('  Salvato: EDA_06_line_plot_medie.png\n');

% Line plot: profili individuali per alcune classi selezionate
fig_line2 = figure('Visible','off','Position',[100 100 1200 800]);
for k = 1:nClassi
    subplot(2, 3, k);
    idx = classid_v == k;
    campioni_classe = mosti(idx, :);
    hold on;
    % Tutti i campioni della classe in grigio trasparente
    for i = 1:size(campioni_classe, 1)
        plot(1:nVar, campioni_classe(i, :), '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
    end
    % Media della classe in evidenza
    media_classe = mean(campioni_classe, 1);
    plot(1:nVar, media_classe, '-o', 'Color', colori_scatter(k,:), ...
         'LineWidth', 3, 'MarkerSize', 8, 'MarkerFaceColor', colori_scatter(k,:));
    xlabel('Variabile');
    ylabel('Valore');
    title([nomiClassi{k} ' (n=' num2str(sum(idx)) ')']);
    set(gca, 'XTick', 1:nVar, 'XTickLabel', namevar_mosti);
    xtickangle(45);
    grid on;
    hold off;
end
sgtitle('Line Plot - Profili Individuali e Medi per Classe');
savePlot(fig_line2, 'EDA_07_line_plot_individuali');
fprintf('  Salvato: EDA_07_line_plot_individuali.png\n');

% Line plot: confronto tra annate per ogni classe
fig_line3 = figure('Visible','off','Position',[100 100 1200 800]);
for k = 1:nClassi
    subplot(2, 3, k);
    idx_00 = (classid_v == k) & (annata == 2000);
    idx_01 = (classid_v == k) & (annata == 2001);
    hold on;
    if sum(idx_00) > 0
        media_00 = mean(mosti(idx_00, :), 1);
        plot(1:nVar, media_00, '-o', 'Color', [0.2 0.6 0.2], ...
             'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0.2 0.6 0.2]);
    end
    if sum(idx_01) > 0
        media_01 = mean(mosti(idx_01, :), 1);
        plot(1:nVar, media_01, '-s', 'Color', [0.8 0.2 0.2], ...
             'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0.8 0.2 0.2]);
    end
    xlabel('Variabile');
    ylabel('Valore Medio');
    title(nomiClassi{k});
    set(gca, 'XTick', 1:nVar, 'XTickLabel', namevar_mosti);
    xtickangle(45);
    legend({'2000', '2001'}, 'Location', 'best');
    grid on;
    hold off;
end
sgtitle('Line Plot - Confronto Annate per Classe');
savePlot(fig_line3, 'EDA_08_line_plot_annate');
fprintf('  Salvato: EDA_08_line_plot_annate.png\n');

fprintf('\n=== Analisi esplorativa completata ===\n');
fprintf('  Generati 8 grafici nella cartella plots/\n\n');

%% ========================================================================
%  SEZIONE 2 — PREPROCESSING
%  ========================================================================
%  SCELTA: Autoscaling (mean-centering + divisione per deviazione standard)
%  MOTIVAZIONE: Le variabili sono aree percentuali HPLC. Anche se hanno
%  la stessa unità di misura (%), possono avere intervalli molto diversi
%  (es. MVD% domina). L'autoscaling dà peso uguale a tutte le variabili,
%  evitando che la PCA sia guidata solo dalle variabili con varianza
%  maggiore. È lo standard industriale per dati spettroscopici/cromatografici.
%  ========================================================================
fprintf('\n--- SEZIONE 2: Preprocessing (Autoscaling) ---\n');

X = mosti;  % matrice originale

% Calcolo media e deviazione standard
mu_X  = mean(X, 1);
std_X = std(X, 0, 1);

% Autoscaling manuale (per chiarezza didattica)
X_auto = (X - mu_X) ./ std_X;

fprintf('Preprocessing: Autoscaling applicato.\n');
fprintf('Medie variabili: '); fprintf('%.2f ', mu_X); fprintf('\n');
fprintf('Std variabili:   '); fprintf('%.2f ', std_X); fprintf('\n');

%% ========================================================================
%  SEZIONE 3 — PCA ESPLORATIVA
%  ========================================================================
fprintf('\n--- SEZIONE 3: PCA Esplorativa ---\n');

% --- Costruzione DataSet PLS_Toolbox ---
% Il DataSet è l'oggetto standard di PLS_Toolbox per gestire dati + metadati
% Usiamo i dati ORIGINALI e lasciamo che PLS_Toolbox applichi il preprocessing
Xds = dataset(X);  % dati originali, non X_auto
Xds.label{1} = nameobj_mosti;          % etichette campioni (mode 1)
Xds.label{2} = namevar_mosti;          % etichette variabili (mode 2)
% Nota: usiamo classid_v separatamente per identificare le classi nei plot

% --- PCA con PLS_Toolbox ---
% Usiamo fino a 6 PC (max = n. variabili) per analizzare la varianza
nPC_max = min(nCamp-1, nVar);  % massimo PC possibili = min(n-1, p)

% Sintassi PLS_Toolbox: pca(data, ncomp, options_struct)
options = pca('options');
options.preprocessing = {preprocess('default', 'autoscale')};  % autoscaling standard
options.display = 'off';
options.plots = 'none';

modelPCA = pca(Xds, nPC_max, options);

% Estrazione risultati
% In PLS_Toolbox, loads{1} e loads{2} possono essere dataset o matrici
% Estraiamo in modo sicuro
if isa(modelPCA.loads{1}, 'dataset')
    T = modelPCA.loads{1}.data;   % se è dataset, usiamo .data
    P = modelPCA.loads{2}.data;
else
    T = modelPCA.loads{1};        % se è già matrice, usiamo direttamente
    P = modelPCA.loads{2};
end

scores   = T;  % scores (campioni nello spazio PC) - per compatibilità
loadings = P;  % loadings (variabili nello spazio PC)

% --- Estrazione varianza spiegata e eigenvalues ---
% Numero effettivo di PC restituiti (può essere < nPC_max)
nPC_actual = size(T, 2);
nPC_max = nPC_actual;  % aggiorna per coerenza

% Calcolo eigenvalues e varianza spiegata dagli scores
eigVal = zeros(nPC_max, 1);
for i = 1:nPC_max
    eigVal(i) = var(T(:, i), 1) * nCamp;  % var con normalizzazione N
end
varExpl = (eigVal / sum(eigVal)) * 100;
varCumul = cumsum(varExpl);

fprintf('\n  PC   Eigenvalue   Var.Spiegata(%%)   Var.Cumulata(%%)\n');
fprintf('  ---  ----------   ----------------  ----------------\n');
for i = 1:nPC_max
    fprintf('  %2d   %8.4f     %8.2f%%          %8.2f%%\n', ...
        i, eigVal(i), varExpl(i), varCumul(i));
end

% Hotelling T2 e Q residuals 
% Calcolo manuale completo
% T2 = sum((t_i / sqrt(lambda_i))^2) per ogni campione
T2 = zeros(nCamp, 1);
for i = 1:nCamp
    T2(i) = sum((T(i,:).^2) ./ eigVal');
end

% Q residuals = SPE (Squared Prediction Error)
% X_ricostruito = T * P', residui = X - X_ricostruito
% Usiamo i dati autoscalati che abbiamo calcolato nella Sezione 2
X_hat_full = T * P';
E_pca = X_auto - X_hat_full;
Qres = sum(E_pca.^2, 2);

% Scegliere un numero ragionevole di PC per i plot
% In base alla varianza spiegata, di solito 2-3 PC bastano per >80%
nPC_plot = find(varCumul >= 80, 1, 'first');
if isempty(nPC_plot) || nPC_plot < 2
    nPC_plot = 2;
end
fprintf('\nPC scelte per visualizzazione: %d (varianza cumulata: %.1f%%)\n', ...
    nPC_plot, varCumul(nPC_plot));

%% ========================================================================
%  SEZIONE 3.1 — GRAFICI PCA
%  ========================================================================
fprintf('\n--- SEZIONE 3.1: Grafici PCA ---\n');

% Colori per le 5 classi (palette distinguibile)
colori = [0.894 0.102 0.110;   % rosso      - Ancellotta
          0.216 0.494 0.722;   % blu        - Montepulciano
          0.302 0.686 0.290;   % verde      - Lambrusco Pugn.
          1.000 0.498 0.000;   % arancio    - Sangiovese
          0.596 0.306 0.639];  % viola      - Nero d'Avola
markers = {'o','s','d','^','v'};

% Marcatori per annata
marker_00 = 'o';   % cerchio = 2000
marker_01 = 's';   % quadrato = 2001

% ---- 3.1.1 SCREE PLOT ----
fig1 = figure('Visible','off','Position',[100 100 700 450]);
yyaxis left
bar(1:nPC_max, varExpl, 0.5, 'FaceColor', [0.5 0.7 0.9]);
ylabel('Varianza spiegata (%)');
yyaxis right
plot(1:nPC_max, varCumul, 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
ylabel('Varianza cumulata (%)');
xlabel('Componente Principale');
title('Scree Plot — Varianza Spiegata per PC');
set(gca, 'XTick', 1:nPC_max);
legend({'Var. singola','Var. cumulata'}, 'Location', 'east');
grid on;
savePlot(fig1, 'PCA_01_scree_plot');
fprintf('  Salvato: PCA_01_scree_plot.png\n');

% ---- 3.1.2 SCORE PLOT PC1 vs PC2 (colorato per varietale) ----
fig2 = figure('Visible','off','Position',[100 100 800 600]);
hold on;
for k = 1:nClassi
    idx = classid_v == k;
    scatter(T(idx,1), T(idx,2), 80, colori(k,:), markers{k}, ...
        'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
end
xlabel(sprintf('PC1 (%.1f%%)', varExpl(1)));
ylabel(sprintf('PC2 (%.1f%%)', varExpl(2)));
title('Score Plot — PC1 vs PC2 (per Varietale)');
legend(nomiClassi, 'Location', 'best');
grid on; axis equal;
% Aggiungi linee di riferimento
xline(0, '--', 'Color', [0.5 0.5 0.5]);
yline(0, '--', 'Color', [0.5 0.5 0.5]);
% Aggiungi ellisse di confidenza 95% (Hotelling T2)
theta = linspace(0, 2*pi, 200);
% Limiti T2 95% per 2 PC
T2lim95 = ((2*(nCamp-1))/(nCamp-2)) * finv(0.95, 2, nCamp-2);
a1 = sqrt(eigVal(1) * T2lim95);
a2 = sqrt(eigVal(2) * T2lim95);
plot(a1*cos(theta), a2*sin(theta), 'k--', 'LineWidth', 1.5);
hold off;
savePlot(fig2, 'PCA_02_score_plot_PC1_PC2_varietale');
fprintf('  Salvato: PCA_02_score_plot_PC1_PC2_varietale.png\n');

% ---- 3.1.3 SCORE PLOT PC1 vs PC2 (colorato per annata) ----
fig3 = figure('Visible','off','Position',[100 100 800 600]);
hold on;
colAnn = [0.2 0.6 0.2; 0.8 0.2 0.2];  % verde=2000, rosso=2001
idx00 = annata == 2000;
idx01 = annata == 2001;
scatter(T(idx00,1), T(idx00,2), 80, colAnn(1,:), 'o', ...
    'filled', 'MarkerEdgeColor', 'k');
scatter(T(idx01,1), T(idx01,2), 80, colAnn(2,:), 's', ...
    'filled', 'MarkerEdgeColor', 'k');
xlabel(sprintf('PC1 (%.1f%%)', varExpl(1)));
ylabel(sprintf('PC2 (%.1f%%)', varExpl(2)));
title('Score Plot — PC1 vs PC2 (per Annata)');
legend({'2000','2001'}, 'Location', 'best');
grid on; axis equal;
xline(0, '--', 'Color', [0.5 0.5 0.5]);
yline(0, '--', 'Color', [0.5 0.5 0.5]);
hold off;
savePlot(fig3, 'PCA_03_score_plot_PC1_PC2_annata');
fprintf('  Salvato: PCA_03_score_plot_PC1_PC2_annata.png\n');

% ---- 3.1.4 SCORE PLOT PC1 vs PC2 (varietale + annata) ----
fig4 = figure('Visible','off','Position',[100 100 900 650]);
hold on;
legendEntries = {};
legendHandles = [];
for k = 1:nClassi
    for a = [2000 2001]
        idx = (classid_v == k) & (annata == a);
        if a == 2000
            mk = 'o';
            mkSize = 80;
        else
            mk = 's';
            mkSize = 80;
        end
        h = scatter(T(idx,1), T(idx,2), mkSize, colori(k,:), mk, ...
            'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
        if any(idx)
            legendHandles(end+1) = h;
            if a == 2000
                legendEntries{end+1} = [sigleClassi{k} ' 2000'];
            else
                legendEntries{end+1} = [sigleClassi{k} ' 2001'];
            end
        end
    end
end
xlabel(sprintf('PC1 (%.1f%%)', varExpl(1)));
ylabel(sprintf('PC2 (%.1f%%)', varExpl(2)));
title('Score Plot — PC1 vs PC2 (Varietale + Annata)');
legend(legendHandles, legendEntries, 'Location', 'bestoutside', 'FontSize', 7);
grid on; axis equal;
xline(0, '--', 'Color', [0.5 0.5 0.5]);
yline(0, '--', 'Color', [0.5 0.5 0.5]);
hold off;
savePlot(fig4, 'PCA_04_score_plot_varietale_annata');
fprintf('  Salvato: PCA_04_score_plot_varietale_annata.png\n');

% ---- 3.1.5 SCORE PLOT PC1 vs PC3 (se utile) ----
if nPC_max >= 3
    fig5 = figure('Visible','off','Position',[100 100 800 600]);
    hold on;
    for k = 1:nClassi
        idx = classid_v == k;
        scatter(T(idx,1), T(idx,3), 80, colori(k,:), markers{k}, ...
            'filled', 'MarkerEdgeColor', 'k');
    end
    xlabel(sprintf('PC1 (%.1f%%)', varExpl(1)));
    ylabel(sprintf('PC3 (%.1f%%)', varExpl(3)));
    title('Score Plot — PC1 vs PC3 (per Varietale)');
    legend(nomiClassi, 'Location', 'best');
    grid on;
    xline(0, '--', 'Color', [0.5 0.5 0.5]);
    yline(0, '--', 'Color', [0.5 0.5 0.5]);
    hold off;
    savePlot(fig5, 'PCA_05_score_plot_PC1_PC3');
    fprintf('  Salvato: PCA_05_score_plot_PC1_PC3.png\n');
end

% ---- 3.1.6 LOADING PLOT PC1 vs PC2 ----
fig6 = figure('Visible','off','Position',[100 100 700 550]);
hold on;
for j = 1:nVar
    plot([0 P(j,1)], [0 P(j,2)], 'b-', 'LineWidth', 1.5);
    text(P(j,1)*1.08, P(j,2)*1.08, namevar_mosti{j}, ...
        'FontSize', 10, 'FontWeight', 'bold', 'Color', 'b');
end
scatter(P(:,1), P(:,2), 60, 'b', 'filled');
xlabel(sprintf('PC1 (%.1f%%)', varExpl(1)));
ylabel(sprintf('PC2 (%.1f%%)', varExpl(2)));
title('Loading Plot — PC1 vs PC2');
grid on;
xline(0, '--', 'Color', [0.5 0.5 0.5]);
yline(0, '--', 'Color', [0.5 0.5 0.5]);
% Cerchio unitario di riferimento
theta = linspace(0, 2*pi, 200);
plot(cos(theta), sin(theta), 'k--', 'LineWidth', 0.8);
axis equal;
hold off;
savePlot(fig6, 'PCA_06_loading_plot');
fprintf('  Salvato: PCA_06_loading_plot.png\n');

% ---- 3.1.7 BIPLOT PC1 vs PC2 ----
fig7 = figure('Visible','off','Position',[100 100 900 700]);
hold on;
% Scores (normalizzati per visualizzazione)
maxS = max(abs(T(:,1:2)), [], 'all');
maxL = max(abs(P(:,1:2)), [], 'all');
scaleFactor = maxS / maxL * 0.7;
for k = 1:nClassi
    idx = classid_v == k;
    scatter(T(idx,1), T(idx,2), 60, colori(k,:), markers{k}, ...
        'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.3);
end
% Loadings scalati
for j = 1:nVar
    quiver(0, 0, P(j,1)*scaleFactor, P(j,2)*scaleFactor, 0, ...
        'Color', [0 0 0], 'LineWidth', 2, 'MaxHeadSize', 0.3);
    text(P(j,1)*scaleFactor*1.12, P(j,2)*scaleFactor*1.12, ...
        namevar_mosti{j}, 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
end
xlabel(sprintf('PC1 (%.1f%%)', varExpl(1)));
ylabel(sprintf('PC2 (%.1f%%)', varExpl(2)));
title('Biplot — PC1 vs PC2');
legend(nomiClassi, 'Location', 'best');
grid on;
xline(0, '--', 'Color', [0.5 0.5 0.5]);
yline(0, '--', 'Color', [0.5 0.5 0.5]);
hold off;
savePlot(fig7, 'PCA_07_biplot');
fprintf('  Salvato: PCA_07_biplot.png\n');

% ---- 3.1.8 HOTELLING T² ----
% T2 con limite al 95% e 99%
% Calcoliamo T2 sulle prime nPC_plot componenti
T2_vals = zeros(nCamp, 1);
for i = 1:nCamp
    T2_vals(i) = sum((T(i,1:nPC_plot).^2) ./ eigVal(1:nPC_plot)');
end
T2_lim95 = ((nPC_plot*(nCamp-1))/(nCamp-nPC_plot)) * finv(0.95, nPC_plot, nCamp-nPC_plot);
T2_lim99 = ((nPC_plot*(nCamp-1))/(nCamp-nPC_plot)) * finv(0.99, nPC_plot, nCamp-nPC_plot);

fig8 = figure('Visible','off','Position',[100 100 900 450]);
hold on;
for k = 1:nClassi
    idx = find(classid_v == k);
    scatter(idx, T2_vals(idx), 50, colori(k,:), markers{k}, ...
        'filled', 'MarkerEdgeColor', 'k');
end
yline(T2_lim95, 'r--', 'LineWidth', 1.5);
yline(T2_lim99, 'r:', 'LineWidth', 1.5);
text(nCamp+1, T2_lim95, '95%', 'Color', 'r', 'FontWeight', 'bold');
text(nCamp+1, T2_lim99, '99%', 'Color', 'r', 'FontWeight', 'bold');
xlabel('Campione');
ylabel('Hotelling T^2');
title(sprintf('Hotelling T^2 (%d PC)', nPC_plot));
legend(nomiClassi, 'Location', 'best');
grid on;
hold off;
savePlot(fig8, 'PCA_08_hotelling_T2');
fprintf('  Salvato: PCA_08_hotelling_T2.png\n');

% ---- 3.1.9 Q RESIDUALS ----
% Residui = X - T * P'  (solo prime nPC_plot PC)
X_hat_plot = T(:,1:nPC_plot) * P(:,1:nPC_plot)';
E = X_auto - X_hat_plot;
Q_vals = sum(E.^2, 2);

% Limite Q (approssimazione Jackson-Mudholkar)
eigRes = eigVal(nPC_plot+1:end);
theta1 = sum(eigRes);
theta2 = sum(eigRes.^2);
theta3 = sum(eigRes.^3);
if theta1 > 0
    h0 = 1 - (2*theta1*theta3)/(3*theta2^2);
    if h0 > 0
        ca = norminv(0.95);
        Q_lim95 = theta1 * (1 + (ca*sqrt(2*theta2*h0^2)/theta1) + ...
            (theta2*h0*(h0-1)/theta1^2))^(1/h0);
    else
        Q_lim95 = max(Q_vals) * 1.5;  % fallback
    end
else
    Q_lim95 = 0;
end

fig9 = figure('Visible','off','Position',[100 100 900 450]);
hold on;
for k = 1:nClassi
    idx = find(classid_v == k);
    scatter(idx, Q_vals(idx), 50, colori(k,:), markers{k}, ...
        'filled', 'MarkerEdgeColor', 'k');
end
yline(Q_lim95, 'r--', 'LineWidth', 1.5);
text(nCamp+1, Q_lim95, '95%', 'Color', 'r', 'FontWeight', 'bold');
xlabel('Campione');
ylabel('Q Residuals (SPE)');
title(sprintf('Q Residuals (%d PC)', nPC_plot));
legend(nomiClassi, 'Location', 'best');
grid on;
hold off;
savePlot(fig9, 'PCA_09_Q_residuals');
fprintf('  Salvato: PCA_09_Q_residuals.png\n');

% ---- 3.1.10 T² vs Q (Influence Plot) ----
fig10 = figure('Visible','off','Position',[100 100 800 600]);
hold on;
for k = 1:nClassi
    idx = classid_v == k;
    scatter(T2_vals(idx), Q_vals(idx), 70, colori(k,:), markers{k}, ...
        'filled', 'MarkerEdgeColor', 'k');
end
xline(T2_lim95, 'r--', 'LineWidth', 1.5);
yline(Q_lim95, 'r--', 'LineWidth', 1.5);
xlabel('Hotelling T^2');
ylabel('Q Residuals');
title('Influence Plot — T^2 vs Q');
legend(nomiClassi, 'Location', 'best');
grid on;
hold off;
savePlot(fig10, 'PCA_10_influence_plot');
fprintf('  Salvato: PCA_10_influence_plot.png\n');

% ---- 3.1.11 VIP-like (Importanza Variabili da PCA) ----
%  In PCA non esiste il VIP "classico" (che è di PLS), ma possiamo
%  calcolare un analogo: l'importanza di ciascuna variabile pesata
%  per la varianza spiegata di ogni PC.
%  VIP_PCA_j = sqrt(p * sum_a(w_ja^2 * lambda_a) / sum(lambda))
%  dove w_ja = loading della variabile j sulla PC a
%  Questa è una misura consolidata in chemometria esplorativa.

VIP_PCA = zeros(nVar, 1);
totVar = sum(eigVal(1:nPC_plot));
for j = 1:nVar
    VIP_PCA(j) = sqrt(nVar * sum(P(j,1:nPC_plot).^2 .* eigVal(1:nPC_plot)') / totVar);
end

fig11 = figure('Visible','off','Position',[100 100 700 450]);
bar(VIP_PCA, 0.6, 'FaceColor', [0.4 0.6 0.8], 'EdgeColor', 'k');
hold on;
yline(1, 'r--', 'LineWidth', 1.5);
yline(0.8, 'b:', 'LineWidth', 1);
set(gca, 'XTick', 1:nVar, 'XTickLabel', namevar_mosti);
xlabel('Variabili');
ylabel('VIP Score (PCA-based)');
title('Importanza Variabili (VIP-like da PCA)');
text(nVar, 1.02, 'soglia = 1', 'Color', 'r', 'HorizontalAlignment', 'right');
grid on;
hold off;
savePlot(fig11, 'PCA_11_VIP_importanza_variabili');
fprintf('  Salvato: PCA_11_VIP_importanza_variabili.png\n');

% ---- 3.1.12 VARIANZA SPIEGATA PER PC ----
fig12 = figure('Visible','off','Position',[100 100 700 400]);
bar(1:nPC_max, varExpl, 0.6, 'FaceColor', [0.3 0.5 0.7]);
xlabel('Componente Principale');
ylabel('Varianza Spiegata (%)');
title('Varianza Spiegata per Componente Principale');
set(gca, 'XTick', 1:nPC_max);
for i = 1:nPC_max
    text(i, varExpl(i)+0.5, sprintf('%.1f%%', varExpl(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end
grid on;
savePlot(fig12, 'PCA_12_varianza_spiegata');
fprintf('  Salvato: PCA_12_varianza_spiegata.png\n');

fprintf('\n=== INTERPRETAZIONE PCA ===\n');
fprintf('Se i cluster si separano per varietale nel score plot -> il profilo\n');
fprintf('antocianico è caratteristico del vitigno.\n');
fprintf('Se i campioni 2000/2001 dello stesso varietale restano vicini -> \n');
fprintf('l''annata produce variabilita'' minore del fattore varietale.\n');
fprintf('Le variabili con loading elevato su PC1/PC2 guidano la separazione.\n');
fprintf('VIP > 1 indica variabili particolarmente importanti.\n\n');

%% ========================================================================
%  SEZIONE 4 — TRAIN / TEST SPLIT STRATIFICATO
%  ========================================================================
%  SCELTA: 70% training / 30% test
%  MOTIVAZIONE: Con 98 campioni e 5 classi, il rapporto 70/30 è il miglior
%  compromesso tra avere abbastanza dati per costruire modelli robusti e
%  mantenere un test set significativo. Lo split è stratificato per classe
%  per garantire che ogni varietale sia rappresentato in entrambi i set.
%  Seed fisso = 42 per riproducibilità.
%  ========================================================================
fprintf('--- SEZIONE 4: Train/Test Split Stratificato ---\n');

rng(42);  % Seed per riproducibilità

trainIdx = [];
testIdx  = [];
trainRatio = 0.70;

for k = 1:nClassi
    idxClass = find(classid_v == k);
    nk = length(idxClass);
    nTrain_k = round(nk * trainRatio);
    
    % Permutazione casuale all'interno della classe
    perm = randperm(nk);
    trainIdx = [trainIdx; idxClass(perm(1:nTrain_k))];
    testIdx  = [testIdx;  idxClass(perm(nTrain_k+1:end))];
end

% Ordinamento per coerenza
trainIdx = sort(trainIdx);
testIdx  = sort(testIdx);

% Creazione set
X_train = X(trainIdx, :);
X_test  = X(testIdx, :);
y_train = classid_v(trainIdx);
y_test  = classid_v(testIdx);
names_train = nameobj_mosti(trainIdx);
names_test  = nameobj_mosti(testIdx);

fprintf('Training set: %d campioni\n', length(trainIdx));
fprintf('Test set:     %d campioni\n', length(testIdx));
for k = 1:nClassi
    fprintf('  %s — Train: %d, Test: %d\n', nomiClassi{k}, ...
        sum(y_train == k), sum(y_test == k));
end
fprintf('\n');

%% ========================================================================
%  SEZIONE 5 — SIMCA (Soft Independent Modelling of Class Analogy)
%  ========================================================================
%  SIMCA costruisce un modello PCA locale per ogni classe. Un nuovo campione
%  viene assegnato alla classe il cui modello lo "accetta" (basandosi su
%  distanza Q e T² dal modello locale).
%  PREPROCESSING: Autoscaling (coerente con la PCA esplorativa).
%  N. PC per classe: selezionato via cross-validation (leave-one-out).
%  ========================================================================
fprintf('--- SEZIONE 5: SIMCA ---\n');

% --- 5.1 Costruzione modelli locali PCA per classe (su training set) ---
% Autoscaling calcolato sul training set (da applicare poi al test)
mu_train = mean(X_train, 1);
std_train = std(X_train, 0, 1);

X_train_auto = (X_train - mu_train) ./ std_train;
X_test_auto  = (X_test  - mu_train) ./ std_train;  % NOTA: usiamo media/std del TRAIN

% Cross-validation per scegliere il numero di PC per ogni classe
% Usiamo leave-one-out (LOO) CV e scegliamo il n. PC che minimizza PRESS
fprintf('\n  Selezione PC per classe via LOO-CV:\n');

maxPC_simca = 5;  % max PC da testare per classe
nPC_simca = zeros(nClassi, 1);

for k = 1:nClassi
    idxK = find(y_train == k);
    Xk = X_train_auto(idxK, :);
    nk = size(Xk, 1);
    
    pressCV = zeros(1, maxPC_simca);
    for npc = 1:maxPC_simca
        press_sum = 0;
        for i = 1:nk
            % Leave-one-out
            Xk_cv = Xk([1:i-1, i+1:end], :);
            xtest_cv = Xk(i, :);
            
            % PCA su Xk_cv (mean-center locale)
            mu_cv = mean(Xk_cv, 1);
            Xk_cv_c = Xk_cv - mu_cv;
            xt_c = xtest_cv - mu_cv;
            
            [~, ~, V] = svd(Xk_cv_c, 'econ');
            P_cv = V(:, 1:min(npc, size(V,2)));
            
            % Ricostruzione e residuo
            xhat = xt_c * P_cv * P_cv';
            press_sum = press_sum + sum((xt_c - xhat).^2);
        end
        pressCV(npc) = press_sum;
    end
    
    % Scelta: primo minimo o primo "gomito"
    [~, bestPC] = min(pressCV);
    nPC_simca(k) = bestPC;
    fprintf('    Classe %d (%s): %d PC (PRESS = %.4f)\n', ...
        k, sigleClassi{k}, bestPC, pressCV(bestPC));
end

% --- 5.2 Costruzione modelli locali finali ---
modelli_simca = struct();
alpha_simca = 0.05;  % livello di significatività

for k = 1:nClassi
    idxK = find(y_train == k);
    Xk = X_train_auto(idxK, :);
    nk = size(Xk, 1);
    
    mu_k = mean(Xk, 1);
    Xk_c = Xk - mu_k;
    
    [U, S, V] = svd(Xk_c, 'econ');
    npc = nPC_simca(k);
    Pk = V(:, 1:npc);
    
    % Score e residui per i campioni della classe
    Tk = Xk_c * Pk;
    Xhat_k = Tk * Pk';
    Ek = Xk_c - Xhat_k;
    
    Qk = sum(Ek.^2, 2);
    eigK = diag(S).^2 / (nk - 1);
    T2k = zeros(nk, 1);
    for i = 1:nk
        T2k(i) = sum(Tk(i,:).^2 ./ eigK(1:npc)');
    end
    
    % Limiti statistici
    T2_lim_k = ((npc * (nk - 1)) / (nk - npc)) * finv(1 - alpha_simca, npc, nk - npc);
    Q_mean_k = mean(Qk);
    Q_std_k = std(Qk);
    
    % Limite Q (approssimazione chi2 o empirico)
    eigRes_k = eigK(npc+1:end);
    if ~isempty(eigRes_k) && sum(eigRes_k) > 0
        th1 = sum(eigRes_k);
        th2 = sum(eigRes_k.^2);
        th3 = sum(eigRes_k.^3);
        h0k = 1 - (2*th1*th3)/(3*th2^2);
        if h0k > 0
            cak = norminv(1 - alpha_simca);
            Q_lim_k = th1 * (1 + (cak*sqrt(2*th2*h0k^2)/th1) + ...
                (th2*h0k*(h0k-1)/th1^2))^(1/h0k);
        else
            Q_lim_k = Q_mean_k + 3 * Q_std_k;
        end
    else
        Q_lim_k = Q_mean_k + 3 * Q_std_k;
    end
    
    modelli_simca(k).mu = mu_k;
    modelli_simca(k).P = Pk;
    modelli_simca(k).npc = npc;
    modelli_simca(k).eigvals = eigK(1:npc);
    modelli_simca(k).T2_lim = T2_lim_k;
    modelli_simca(k).Q_lim = Q_lim_k;
    modelli_simca(k).Tk = Tk;
    modelli_simca(k).Qk = Qk;
    modelli_simca(k).T2k = T2k;
end

% --- 5.3 Predizione SIMCA ---
% Per ogni campione, calcola Q e T2 rispetto a OGNI modello locale.
% Assegna alla classe il cui modello ha la distanza combinata minore.
% Se nessun modello lo accetta (Q e T2 sopra i limiti per tutte le classi),
% il campione è "non assegnato".

function [y_pred, dist_Q, dist_T2] = predictSIMCA(X_auto, modelli, nClassi)
    nTest = size(X_auto, 1);
    dist_Q  = zeros(nTest, nClassi);
    dist_T2 = zeros(nTest, nClassi);
    
    for kk = 1:nClassi
        Pk = modelli(kk).P;
        mu_k = modelli(kk).mu;
        eigK = modelli(kk).eigvals;
        
        for i = 1:nTest
            xi = X_auto(i, :) - mu_k;
            ti = xi * Pk;
            xhat = ti * Pk';
            ei = xi - xhat;
            
            dist_Q(i, kk) = sum(ei.^2);
            dist_T2(i, kk) = sum(ti.^2 ./ eigK');
        end
        
        % Normalizza per i limiti
        dist_Q(:, kk)  = dist_Q(:, kk)  / modelli(kk).Q_lim;
        dist_T2(:, kk) = dist_T2(:, kk) / modelli(kk).T2_lim;
    end
    
    % Distanza combinata (somma normalizzata Q + T2)
    distTot = dist_Q + dist_T2;
    [~, y_pred] = min(distTot, [], 2);
end

% Predizioni su training e test
[y_pred_train_simca, distQ_train, distT2_train] = predictSIMCA(X_train_auto, modelli_simca, nClassi);
[y_pred_test_simca, distQ_test, distT2_test]    = predictSIMCA(X_test_auto,  modelli_simca, nClassi);

% --- 5.4 Metriche SIMCA ---
fprintf('\n  === SIMCA — Risultati Training ===\n');
CM_train_simca = confusionmat(y_train, y_pred_train_simca);
acc_train_simca = sum(diag(CM_train_simca)) / sum(CM_train_simca(:)) * 100;
fprintf('  Accuracy Training: %.1f%%\n', acc_train_simca);
fprintf('  Confusion Matrix (Train):\n');
disp(CM_train_simca);

fprintf('  === SIMCA — Risultati Test ===\n');
CM_test_simca = confusionmat(y_test, y_pred_test_simca);
acc_test_simca = sum(diag(CM_test_simca)) / sum(CM_test_simca(:)) * 100;
fprintf('  Accuracy Test: %.1f%%\n', acc_test_simca);
fprintf('  Confusion Matrix (Test):\n');
disp(CM_test_simca);

% Sensitivity e Specificity per classe
fprintf('  Classe      Sensitivity  Specificity\n');
fprintf('  --------    -----------  -----------\n');
for k = 1:nClassi
    TP = CM_test_simca(k, k);
    FN = sum(CM_test_simca(k, :)) - TP;
    FP = sum(CM_test_simca(:, k)) - TP;
    TN = sum(CM_test_simca(:)) - TP - FN - FP;
    sens = TP / (TP + FN) * 100;
    spec = TN / (TN + FP) * 100;
    fprintf('  %-12s %6.1f%%      %6.1f%%\n', sigleClassi{k}, sens, spec);
end

% --- 5.5 Grafici SIMCA ---
% Confusion Matrix Test
fig13 = figure('Visible','off','Position',[100 100 600 500]);
imagesc(CM_test_simca);
colormap(flipud(hot));
colorbar;
set(gca, 'XTick', 1:nClassi, 'XTickLabel', sigleClassi);
set(gca, 'YTick', 1:nClassi, 'YTickLabel', sigleClassi);
xlabel('Predetto');
ylabel('Reale');
title(sprintf('SIMCA — Confusion Matrix (Test, Acc=%.1f%%)', acc_test_simca));
% Aggiungi numeri nelle celle
for i = 1:nClassi
    for j = 1:nClassi
        text(j, i, num2str(CM_test_simca(i,j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 14, ...
            'FontWeight', 'bold', 'Color', 'b');
    end
end
savePlot(fig13, 'SIMCA_01_confusion_matrix_test');
fprintf('\n  Salvato: SIMCA_01_confusion_matrix_test.png\n');

% Score Plot del modello locale classe 1 (esempio)
fig14 = figure('Visible','off','Position',[100 100 900 600]);
nSubplots = min(nClassi, 6);
for k = 1:nSubplots
    subplot(2, 3, k);
    Tk = modelli_simca(k).Tk;
    if modelli_simca(k).npc >= 2
        scatter(Tk(:,1), Tk(:,2), 50, colori(k,:), 'filled', 'MarkerEdgeColor', 'k');
        xlabel('PC1'); ylabel('PC2');
    else
        scatter(Tk(:,1), zeros(size(Tk,1),1), 50, colori(k,:), 'filled', 'MarkerEdgeColor', 'k');
        xlabel('PC1'); ylabel('');
    end
    title(sprintf('SIMCA: %s (%d PC)', sigleClassi{k}, modelli_simca(k).npc));
    grid on;
end
sgtitle('SIMCA — Score Plots Modelli Locali (Training)');
savePlot(fig14, 'SIMCA_02_score_plots_locali');
fprintf('  Salvato: SIMCA_02_score_plots_locali.png\n');

% Distanze di Classe (Coomans Plot) - scelta 2 classi più importanti
% Coomans plot: distanza Q normalizzata classe i vs classe j
% Facciamo per tutte le coppie più significative
fig15 = figure('Visible','off','Position',[100 100 900 700]);
combIdx = 1;
coppie = [1 2; 1 3; 1 4; 2 5; 3 4];  % coppie di esempio
nCoppie = min(size(coppie, 1), 6);
for cc = 1:nCoppie
    c1 = coppie(cc, 1);
    c2 = coppie(cc, 2);
    subplot(2, 3, cc);
    hold on;
    for k = 1:nClassi
        idx = y_test == k;
        scatter(distQ_test(idx, c1), distQ_test(idx, c2), 50, ...
            colori(k,:), markers{k}, 'filled', 'MarkerEdgeColor', 'k');
    end
    xline(1, 'r--');
    yline(1, 'r--');
    xlabel(sprintf('Dist. Norm. %s', sigleClassi{c1}));
    ylabel(sprintf('Dist. Norm. %s', sigleClassi{c2}));
    title(sprintf('%s vs %s', sigleClassi{c1}, sigleClassi{c2}));
    grid on;
    hold off;
end
sgtitle('SIMCA — Coomans Plots (Distanze di Classe Normalizzate)');
legend(nomiClassi, 'Position', [0.7 0.15 0.25 0.2]);
savePlot(fig15, 'SIMCA_03_coomans_plots');
fprintf('  Salvato: SIMCA_03_coomans_plots.png\n');

% Predicted vs True (SIMCA)
fig16 = figure('Visible','off','Position',[100 100 700 500]);
hold on;
for k = 1:nClassi
    idx = y_test == k;
    scatter(y_test(idx) + randn(sum(idx),1)*0.05, ...
        y_pred_test_simca(idx) + randn(sum(idx),1)*0.05, ...
        60, colori(k,:), markers{k}, 'filled', 'MarkerEdgeColor', 'k');
end
plot([0.5 nClassi+0.5], [0.5 nClassi+0.5], 'k--', 'LineWidth', 1.5);
set(gca, 'XTick', 1:nClassi, 'XTickLabel', sigleClassi);
set(gca, 'YTick', 1:nClassi, 'YTickLabel', sigleClassi);
xlabel('Classe Reale');
ylabel('Classe Predetta');
title('SIMCA — Predicted vs True (Test)');
legend(nomiClassi, 'Location', 'best');
grid on;
hold off;
savePlot(fig16, 'SIMCA_04_predicted_vs_true');
fprintf('  Salvato: SIMCA_04_predicted_vs_true.png\n');

%% ========================================================================
%  SEZIONE 6 — PLS-DA (Partial Least Squares Discriminant Analysis)
%  ========================================================================
%  PLS-DA è il metodo standard industriale per classificazione in
%  chemometria. Tratta il problema come una regressione PLS dove la Y
%  è una matrice dummy (one-hot encoding delle classi).
%  PREPROCESSING: Autoscaling su X (già applicato).
%  Y: codificata come dummy matrix (0/1) — mean-centered internamente.
%  N. LV: selezionato via cross-validation (venetian blinds o LOO).
%  ========================================================================
fprintf('\n--- SEZIONE 6: PLS-DA ---\n');

% --- 6.1 Preparazione Y dummy ---
Y_train_dummy = zeros(length(y_train), nClassi);
Y_test_dummy  = zeros(length(y_test), nClassi);
for k = 1:nClassi
    Y_train_dummy(y_train == k, k) = 1;
    Y_test_dummy(y_test == k, k) = 1;
end

% --- 6.2 Creazione DataSet PLS_Toolbox ---
Xds_train = dataset(X_train);
Xds_train.label{1} = names_train;
Xds_train.label{2} = namevar_mosti;
% Nota: gestiamo le classi tramite Y_train_dummy per PLS-DA

Yds_train = dataset(Y_train_dummy);
Yds_train.label{1} = names_train;
Yds_train.label{2} = sigleClassi;

Xds_test = dataset(X_test);
Xds_test.label{1} = names_test;
Xds_test.label{2} = namevar_mosti;
% Nota: gestiamo le classi tramite Y_test_dummy per PLS-DA

Yds_test = dataset(Y_test_dummy);
Yds_test.label{1} = names_test;
Yds_test.label{2} = sigleClassi;

% --- 6.3 Cross-validation per selezione numero LV ---
%  SCELTA CV: Venetian Blinds (10 splits)
%  MOTIVAZIONE: Con 98 campioni, venetian blinds è più robusto di LOO
%  e meno costoso computazionalmente. 10 splits è standard.
%  Cerchiamo il numero di LV che minimizza RMSECV.

maxLV = min(10, min(size(X_train))-1);

% Sintassi corretta PLS_Toolbox: plsda(X, Y, ncomp, options)
optPLSDA = plsda('options');
optPLSDA.display = 'off';
optPLSDA.plots = 'none';
optPLSDA.preprocessing = {preprocess('default','autoscale') preprocess('default','autoscale')};
optPLSDA.crossvalidation = {'vet', 10};  % venetian blinds, 10 splits

fprintf('  Cross-validation: Venetian Blinds, 10 splits\n');
fprintf('  Max LV testati: %d\n', maxLV);

% Costruzione modello PLS-DA con CV - ATTENZIONE: maxLV come terzo argomento!
modelPLSDA = plsda(Xds_train, Yds_train, maxLV, optPLSDA);

% Estrazione RMSECV per selezione LV
try
    % Accesso sicuro ai risultati CV
    if isfield(modelPLSDA, 'detail') && isfield(modelPLSDA.detail, 'rmsecv')
        rmsecv_vals = modelPLSDA.detail.rmsecv;
    elseif isfield(modelPLSDA, 'rmsecv')
        rmsecv_vals = modelPLSDA.rmsecv;
    else
        error('RMSECV non trovato');
    end
    
    % rmsecv_vals potrebbe essere matrice (LV x nClassi)
    if size(rmsecv_vals, 2) > 1
        rmsecv_avg = mean(rmsecv_vals, 2);  % media su tutte le classi
    else
        rmsecv_avg = rmsecv_vals(:);  % forza colonna
    end
    
    % Verifica che rmsecv_avg non sia vuoto
    if isempty(rmsecv_avg) || all(isnan(rmsecv_avg))
        error('RMSECV vuoto o NaN');
    end
    
    [~, bestLV] = min(rmsecv_avg);
    bestLV = double(bestLV);  % assicura sia double scalare
    fprintf('  LV ottimali (min RMSECV medio): %d\n', bestLV);
    fprintf('  RMSECV per LV:\n');
    for lv = 1:length(rmsecv_avg)
        fprintf('    LV %d: %.4f', lv, rmsecv_avg(lv));
        if lv == bestLV
            fprintf(' <-- ottimale');
        end
        fprintf('\n');
    end
catch ME
    % Se la struttura è diversa, scelta ragionevole
    fprintf('  Avviso: impossibile estrarre RMSECV (%s)\n', ME.message);
    bestLV = min(3, maxLV);
    fprintf('  LV scelte (default): %d\n', bestLV);
end

% --- 6.4 Modello finale con numero ottimale di LV ---
optFinal = plsda('options');
optFinal.display = 'off';
optFinal.plots = 'none';
optFinal.preprocessing = {preprocess('default','autoscale') preprocess('default','autoscale')};

modelPLSDA_final = plsda(Xds_train, Yds_train, bestLV, optFinal);

% --- 6.5 Predizione su Training e Test ---
% Ispezionare struttura completa del modello per capire dove sono P e Q
fprintf('  === Ispezione struttura modello PLS-DA ===\n');
fprintf('  Campi principali del modello:\n');
campi = fieldnames(modelPLSDA_final);
for i = 1:length(campi)
    fprintf('    - %s\n', campi{i});
end

fprintf('\n  Campi in detail:\n');
if isfield(modelPLSDA_final, 'detail')
    campi_detail = fieldnames(modelPLSDA_final.detail);
    for i = 1:length(campi_detail)
        fprintf('    - detail.%s\n', campi_detail{i});
    end
end

fprintf('\n  Numero elementi in loads: %d\n', length(modelPLSDA_final.loads));
for idx = 1:length(modelPLSDA_final.loads)
    if isa(modelPLSDA_final.loads{idx}, 'dataset')
        fprintf('    loads{%d} = dataset %dx%d\n', idx, ...
            size(modelPLSDA_final.loads{idx}.data, 1), ...
            size(modelPLSDA_final.loads{idx}.data, 2));
    else
        fprintf('    loads{%d} = %dx%d\n', idx, ...
            size(modelPLSDA_final.loads{idx}, 1), ...
            size(modelPLSDA_final.loads{idx}, 2));
    end
end

% Strategia alternativa: usare i dati grezzi per calcolare le predizioni
% loads{1} = T (scores 69×3)
% loads{2} = W (weights 6×3)
% Devo calcolare P da X e T: X = T*P' => P' = T\X => P = (T\X)'

fprintf('\n  Calcolo manuale di P e Q dai dati...\n');

% Estrarre T (scores training)
if isa(modelPLSDA_final.loads{1}, 'dataset')
    T_train = modelPLSDA_final.loads{1}.data;
else
    T_train = modelPLSDA_final.loads{1};
end

% Estrarre W (weights)
if isa(modelPLSDA_final.loads{2}, 'dataset')
    W_plsda = modelPLSDA_final.loads{2}.data;
else
    W_plsda = modelPLSDA_final.loads{2};
end

% Calcolare P (X-loadings) da: X_auto = T*P' + E => P = (T\X_auto)'
P_plsda = (T_train \ X_train_auto)';
fprintf('    P calcolato: %dx%d\n', size(P_plsda,1), size(P_plsda,2));

% Calcolare Q (Y-loadings) da: Y_dummy = T*Q' + F => Q = (T\Y_dummy)'
Q_plsda = (T_train \ Y_train_dummy)';
fprintf('    Q calcolato: %dx%d\n', size(Q_plsda,1), size(Q_plsda,2));

% Predizione training: Y_pred_train = T_train * Q'
Y_pred_train = T_train * Q_plsda';

% Per test: T_test = X_test_auto * W * inv(P'*W)
R_plsda = W_plsda / (P_plsda' * W_plsda);
T_test = X_test_auto * R_plsda;

% Predizione test: Y_pred_test = T_test * Q'
Y_pred_test = T_test * Q_plsda';

% Predizione test: Y_pred_test = T_test * Q'
Y_pred_test = T_test * Q_plsda';

% Assegnazione classe: argmax delle colonne Y predetta
[~, y_pred_train_plsda] = max(Y_pred_train, [], 2);
[~, y_pred_test_plsda]  = max(Y_pred_test, [], 2);

% Forza vettori colonna per confusionmat
y_pred_train_plsda = y_pred_train_plsda(:);
y_pred_test_plsda = y_pred_test_plsda(:);

% Verifica dimensioni prima di confusionmat
fprintf('  Debug: size(y_train)=%dx%d, size(y_pred_train_plsda)=%dx%d\n', ...
    size(y_train,1), size(y_train,2), size(y_pred_train_plsda,1), size(y_pred_train_plsda,2));

% --- 6.6 Metriche PLS-DA ---
fprintf('\n  === PLS-DA — Risultati Training ===\n');
CM_train_plsda = confusionmat(y_train, y_pred_train_plsda);
acc_train_plsda = sum(diag(CM_train_plsda)) / sum(CM_train_plsda(:)) * 100;
fprintf('  Accuracy Training: %.1f%%\n', acc_train_plsda);
fprintf('  Confusion Matrix (Train):\n');
disp(CM_train_plsda);

fprintf('  === PLS-DA — Risultati Test ===\n');
CM_test_plsda = confusionmat(y_test, y_pred_test_plsda);
acc_test_plsda = sum(diag(CM_test_plsda)) / sum(CM_test_plsda(:)) * 100;
fprintf('  Accuracy Test: %.1f%%\n', acc_test_plsda);
fprintf('  Confusion Matrix (Test):\n');
disp(CM_test_plsda);

% RMSEC, RMSEP
RMSEC = sqrt(mean((Y_train_dummy(:) - Y_pred_train(:)).^2));
RMSEP = sqrt(mean((Y_test_dummy(:) - Y_pred_test(:)).^2));
fprintf('  RMSEC (calibration): %.4f\n', RMSEC);
fprintf('  RMSEP (prediction):  %.4f\n', RMSEP);

% R²
SS_res_train = sum((Y_train_dummy(:) - Y_pred_train(:)).^2);
SS_tot_train = sum((Y_train_dummy(:) - mean(Y_train_dummy(:))).^2);
R2_cal = 1 - SS_res_train / SS_tot_train;

SS_res_test = sum((Y_test_dummy(:) - Y_pred_test(:)).^2);
SS_tot_test = sum((Y_test_dummy(:) - mean(Y_test_dummy(:))).^2);
R2_pred = 1 - SS_res_test / SS_tot_test;

fprintf('  R^2 Calibrazione: %.4f\n', R2_cal);
fprintf('  R^2 Predizione:   %.4f\n', R2_pred);

% Sensitivity e Specificity per classe
fprintf('\n  Classe      Sensitivity  Specificity\n');
fprintf('  --------    -----------  -----------\n');
for k = 1:nClassi
    TP = CM_test_plsda(k, k);
    FN = sum(CM_test_plsda(k, :)) - TP;
    FP = sum(CM_test_plsda(:, k)) - TP;
    TN = sum(CM_test_plsda(:)) - TP - FN - FP;
    sens = TP / max((TP + FN), 1) * 100;
    spec = TN / max((TN + FP), 1) * 100;
    fprintf('  %-12s %6.1f%%      %6.1f%%\n', sigleClassi{k}, sens, spec);
end

% --- 6.7 VIP (Variable Importance in Projection) ---
% VIP è calcolato dai pesi W del PLS e dalla varianza Y spiegata
% VIP_j = sqrt(p * sum_a(w_ja^2 * SS_a) / sum(SS_a))
% È IL metodo standard per identificare variabili importanti in PLS-DA.

try
    % Estrazione pesi e scores dal modello in modo sicuro
    if isa(modelPLSDA_final.loads{3}, 'dataset')
        W = modelPLSDA_final.loads{3}.data;  % pesi W (nVar x nLV)
    else
        W = modelPLSDA_final.loads{3};
    end
    
    if isa(modelPLSDA_final.loads{1}, 'dataset')
        T_pls = modelPLSDA_final.loads{1}.data;  % scores T (nTrain x nLV)
    else
        T_pls = modelPLSDA_final.loads{1};
    end
    
    if isa(modelPLSDA_final.loads{4}, 'dataset')
        Q_pls = modelPLSDA_final.loads{4}.data;  % loadings Y (nClassi x nLV)
    else
        Q_pls = modelPLSDA_final.loads{4};
    end
    
    % Calcolo VIP
    nLV = bestLV;
    SS = zeros(nLV, 1);
    for a = 1:nLV
        SS(a) = (T_pls(:,a)' * T_pls(:,a)) * (Q_pls(:,a)' * Q_pls(:,a));
    end
    
    VIP = zeros(nVar, 1);
    for j = 1:nVar
        VIP(j) = sqrt(nVar * sum(W(j,:).^2 .* SS') / sum(SS));
    end
catch
    % Calcolo VIP alternativo se la struttura del modello è diversa
    fprintf('  Calcolo VIP con metodo alternativo...\n');
    
    % Autoscaling manuale
    Xtrain_as = (X_train - mean(X_train)) ./ std(X_train);
    Ytrain_as = (Y_train_dummy - mean(Y_train_dummy)) ./ std(Y_train_dummy);
    
    % PLS NIPALS manuale per estrarre W, T, P, Q
    nLV = bestLV;
    [nTr, p] = size(Xtrain_as);
    W_man = zeros(p, nLV);
    T_man = zeros(nTr, nLV);
    P_man = zeros(p, nLV);
    Q_man = zeros(nClassi, nLV);
    
    E = Xtrain_as;
    F = Ytrain_as;
    
    for a = 1:nLV
        [~, ~, V_pls] = svd(E' * F, 'econ');
        U_pls = F * (F' * E * V_pls(:,1)) / norm(F' * E * V_pls(:,1));
        w = E' * U_pls / (U_pls' * U_pls);
        w = w / norm(w);
        t = E * w;
        p_a = E' * t / (t' * t);
        q_a = F' * t / (t' * t);
        
        E = E - t * p_a';
        F = F - t * q_a';
        
        W_man(:, a) = w;
        T_man(:, a) = t;
        P_man(:, a) = p_a;
        Q_man(:, a) = q_a;
    end
    
    SS = zeros(nLV, 1);
    for a = 1:nLV
        SS(a) = (T_man(:,a)' * T_man(:,a)) * (Q_man(:,a)' * Q_man(:,a));
    end
    
    VIP = zeros(nVar, 1);
    for j = 1:nVar
        VIP(j) = sqrt(nVar * sum(W_man(j,:).^2 .* SS') / sum(SS));
    end
end

fprintf('\n  VIP Scores (PLS-DA):\n');
for j = 1:nVar
    imp = '';
    if VIP(j) >= 1
        imp = ' *** IMPORTANTE';
    elseif VIP(j) >= 0.8
        imp = ' *  rilevante';
    end
    fprintf('    %-12s: %.4f%s\n', namevar_mosti{j}, VIP(j), imp);
end

% --- 6.8 Grafici PLS-DA ---

% 6.8.1 Score Plot PLS-DA (LV1 vs LV2)
fig17 = figure('Visible','off','Position',[100 100 800 600]);
try
    if isa(modelPLSDA_final.loads{1}, 'dataset')
        T_plsda = modelPLSDA_final.loads{1}.data;
    else
        T_plsda = modelPLSDA_final.loads{1};
    end
catch
    T_plsda = T_man;
end
hold on;
for k = 1:nClassi
    idx = y_train == k;
    scatter(T_plsda(idx,1), T_plsda(idx,min(2,size(T_plsda,2))), ...
        80, colori(k,:), markers{k}, 'filled', 'MarkerEdgeColor', 'k');
end
xlabel('LV1'); ylabel('LV2');
title('PLS-DA — Score Plot (Training)');
legend(nomiClassi, 'Location', 'best');
grid on;
xline(0, '--', 'Color', [0.5 0.5 0.5]);
yline(0, '--', 'Color', [0.5 0.5 0.5]);
hold off;
savePlot(fig17, 'PLSDA_01_score_plot');
fprintf('  Salvato: PLSDA_01_score_plot.png\n');

% 6.8.2 Confusion Matrix PLS-DA (Test)
fig18 = figure('Visible','off','Position',[100 100 600 500]);
imagesc(CM_test_plsda);
colormap(flipud(hot));
colorbar;
set(gca, 'XTick', 1:nClassi, 'XTickLabel', sigleClassi);
set(gca, 'YTick', 1:nClassi, 'YTickLabel', sigleClassi);
xlabel('Predetto');
ylabel('Reale');
title(sprintf('PLS-DA — Confusion Matrix (Test, Acc=%.1f%%)', acc_test_plsda));
for i = 1:nClassi
    for j = 1:nClassi
        text(j, i, num2str(CM_test_plsda(i,j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 14, ...
            'FontWeight', 'bold', 'Color', 'b');
    end
end
savePlot(fig18, 'PLSDA_02_confusion_matrix_test');
fprintf('  Salvato: PLSDA_02_confusion_matrix_test.png\n');

% 6.8.3 VIP Plot
fig19 = figure('Visible','off','Position',[100 100 700 450]);
barh(VIP, 0.6, 'FaceColor', [0.4 0.6 0.8], 'EdgeColor', 'k');
hold on;
xline(1, 'r--', 'LineWidth', 1.5);
xline(0.8, 'b:', 'LineWidth', 1);
set(gca, 'YTick', 1:nVar, 'YTickLabel', namevar_mosti);
ylabel('Variabili');
xlabel('VIP Score');
title('PLS-DA — Variable Importance in Projection (VIP)');
text(1.02, 0.5, 'soglia = 1', 'Color', 'r');
grid on;
hold off;
savePlot(fig19, 'PLSDA_03_VIP');
fprintf('  Salvato: PLSDA_03_VIP.png\n');

% 6.8.4 Predicted vs True (PLS-DA)
fig20 = figure('Visible','off','Position',[100 100 700 500]);
hold on;
for k = 1:nClassi
    idx = y_test == k;
    scatter(y_test(idx) + randn(sum(idx),1)*0.05, ...
        y_pred_test_plsda(idx) + randn(sum(idx),1)*0.05, ...
        60, colori(k,:), markers{k}, 'filled', 'MarkerEdgeColor', 'k');
end
plot([0.5 nClassi+0.5], [0.5 nClassi+0.5], 'k--', 'LineWidth', 1.5);
set(gca, 'XTick', 1:nClassi, 'XTickLabel', sigleClassi);
set(gca, 'YTick', 1:nClassi, 'YTickLabel', sigleClassi);
xlabel('Classe Reale');
ylabel('Classe Predetta');
title('PLS-DA — Predicted vs True (Test)');
legend(nomiClassi, 'Location', 'best');
grid on;
hold off;
savePlot(fig20, 'PLSDA_04_predicted_vs_true');
fprintf('  Salvato: PLSDA_04_predicted_vs_true.png\n');

% 6.8.5 Y Predicted values (discrimination plot)
fig21 = figure('Visible','off','Position',[100 100 1000 700]);
for k = 1:nClassi
    subplot(2, 3, k);
    hold on;
    for kk = 1:nClassi
        idx = y_test == kk;
        scatter(find(idx), Y_pred_test(idx, k), 40, colori(kk,:), ...
            markers{kk}, 'filled', 'MarkerEdgeColor', 'k');
    end
    yline(0.5, 'r--', 'LineWidth', 1.5);
    xlabel('Campione');
    ylabel(sprintf('Y pred (%s)', sigleClassi{k}));
    title(sprintf('Discriminazione %s', sigleClassi{k}));
    grid on;
    hold off;
end
sgtitle('PLS-DA — Valori Y Predetti per Classe (Test)');
savePlot(fig21, 'PLSDA_05_discrimination_plot');
fprintf('  Salvato: PLSDA_05_discrimination_plot.png\n');

% 6.8.6 RMSECV plot (se disponibile)
try
    fig22 = figure('Visible','off','Position',[100 100 700 450]);
    plot(1:length(rmsecv_avg), rmsecv_avg, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
    hold on;
    plot(bestLV, rmsecv_avg(bestLV), 'rp', 'MarkerSize', 15, 'MarkerFaceColor', 'r');
    xlabel('Numero di Variabili Latenti (LV)');
    ylabel('RMSECV (medio)');
    title('PLS-DA — Selezione LV via Cross-Validation');
    grid on;
    text(bestLV, rmsecv_avg(bestLV)*1.05, sprintf('LV=%d', bestLV), ...
        'Color', 'r', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    hold off;
    savePlot(fig22, 'PLSDA_06_RMSECV_vs_LV');
    fprintf('  Salvato: PLSDA_06_RMSECV_vs_LV.png\n');
catch
    fprintf('  (RMSECV plot non disponibile)\n');
end

% 6.8.7 Leverage vs Residuals (PLS-DA)
fig23 = figure('Visible','off','Position',[100 100 800 600]);
hold on;
% Leverage = diag(T * (T'T)^-1 * T')
if size(T_plsda, 2) >= 1
    Hat = T_plsda * pinv(T_plsda' * T_plsda) * T_plsda';
    lev = diag(Hat);
    % Residui Y
    Y_train_pred_check = Y_pred_train;
    resY = sum((Y_train_dummy - Y_train_pred_check).^2, 2);
    
    for k = 1:nClassi
        idx = y_train == k;
        scatter(lev(idx), resY(idx), 60, colori(k,:), markers{k}, ...
            'filled', 'MarkerEdgeColor', 'k');
    end
    lev_lim = 2 * bestLV / length(y_train);  % limite leverage
    xline(lev_lim, 'r--', 'LineWidth', 1.5);
    xlabel('Leverage');
    ylabel('Residui Y (SSE)');
    title('PLS-DA — Leverage vs Residuals (Training)');
    legend(nomiClassi, 'Location', 'best');
    grid on;
end
hold off;
savePlot(fig23, 'PLSDA_07_leverage_residuals');
fprintf('  Salvato: PLSDA_07_leverage_residuals.png\n');

%% ========================================================================
%  SEZIONE 7 — RIEPILOGO FINALE
%  ========================================================================
fprintf('\n');
fprintf('================================================================\n');
fprintf('  RIEPILOGO RISULTATI CLASSIFICAZIONE\n');
fprintf('================================================================\n');
fprintf('  Modello      Acc.Train    Acc.Test    RMSEP\n');
fprintf('  ---------    ---------    --------    -----\n');
fprintf('  SIMCA        %5.1f%%       %5.1f%%       n/a\n', acc_train_simca, acc_test_simca);
fprintf('  PLS-DA       %5.1f%%       %5.1f%%       %.4f\n', acc_train_plsda, acc_test_plsda, RMSEP);
fprintf('================================================================\n');
fprintf('\n');
fprintf('  PLS-DA - R^2 Cal: %.4f | R^2 Pred: %.4f\n', R2_cal, R2_pred);
fprintf('  PLS-DA - RMSEC:   %.4f | RMSEP:    %.4f\n', RMSEC, RMSEP);
fprintf('  PLS-DA - N. LV:   %d\n', bestLV);
fprintf('\n');
fprintf('  Variabili piu'' importanti (VIP > 1):\n');
for j = 1:nVar
    if VIP(j) >= 1
        fprintf('    -> %s (VIP = %.3f)\n', namevar_mosti{j}, VIP(j));
    end
end
fprintf('\n');
fprintf('  Tutti i grafici salvati in: %s\n', plotDir);
fprintf('================================================================\n');

fprintf('\nAnalisi completata con successo.\n');

%% ========================================================================
%  FINE SCRIPT
%  ========================================================================
