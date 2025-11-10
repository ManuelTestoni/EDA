%% ANALISI COMPLETA CLUSTERING - OLIVE OIL DATASET
% Questo script genera TUTTI i grafici richiesti in una sola esecuzione
% per confrontare i diversi metodi di clustering

clear; close all; clc;

fprintf('=== INIZIO ANALISI COMPLETA OLIVE OIL ===\n\n');

%% CARICAMENTO DATI
olivedata = readmatrix('oliveoil.csv');
data = normalize(olivedata);  % autoscaling
[nSamples, nVars] = size(data);
fprintf('Dataset: %d campioni x %d variabili\n\n', nSamples, nVars);

%% CALCOLO PCA (usato in tutti i plot)
[u, s, v] = svds(data, 2);
scores = u * s;

%% =========================================================================
%% PARTE 1: CLUSTERING GERARCHICO - TEST DIVERSE COMBINAZIONI
%% =========================================================================
fprintf('--- PARTE 1: CLUSTERING GERARCHICO ---\n');

% Liste di combinazioni da testare
linkage_methods = {'ward', 'single', 'complete', 'average'};
distance_methods = {'euclidean', 'cityblock', 'correlation'};

% Numero di cluster da cercare
nClusters = 3;

fprintf('Testando combinazioni di linkage e distanza:\n');
best_silhouette = -Inf;
best_linkage = '';
best_distance = '';
best_T = [];
best_Z = [];

figNum = 1;

for i = 1:length(linkage_methods)
    for j = 1:length(distance_methods)
        link_method = linkage_methods{i};
        dist_method = distance_methods{j};
        
        % Ward funziona solo con euclidean
        if strcmp(link_method, 'ward') && ~strcmp(dist_method, 'euclidean')
            continue;
        end
        
        try
            % Calcola linkage
            Z = linkage(data, link_method, dist_method);
            
            % Ottieni cluster
            T = cluster(Z, 'maxclust', nClusters);
            
            % Calcola silhouette
            sil_vals = silhouette(data, T);
            mean_sil = mean(sil_vals);
            
            fprintf('  %s + %s: silhouette = %.4f\n', link_method, dist_method, mean_sil);
            
            % GRAFICO: Dendrogramma per questa combinazione
            figure(figNum);
            dendrogram(Z, 0);
            title(sprintf('Dendrogram: %s linkage + %s distance (Sil=%.3f)', ...
                link_method, dist_method, mean_sil));
            xlabel('Sample index');
            ylabel('Distance');
            figNum = figNum + 1;
            
            % Salva se è il migliore
            if mean_sil > best_silhouette
                best_silhouette = mean_sil;
                best_linkage = link_method;
                best_distance = dist_method;
                best_T = T;
                best_Z = Z;
            end
            
        catch ME
            fprintf('  %s + %s: ERRORE - %s\n', link_method, dist_method, ME.message);
        end
    end
end

fprintf('\n*** MIGLIORE COMBINAZIONE: %s + %s (silhouette = %.4f) ***\n\n', ...
    best_linkage, best_distance, best_silhouette);

%% =========================================================================
%% PARTE 2: FIGURE RICHIESTE CON IL CLUSTERING MIGLIORE
%% =========================================================================
fprintf('--- PARTE 2: GRAFICI RICHIESTI (MIGLIORE CLUSTERING) ---\n');

% Calcola cutoff per avere esattamente nClusters
cutoff_idx = size(best_Z, 1) - nClusters + 2;
cutoff = best_Z(cutoff_idx, 3);

fprintf('FIGURA 1: Samples dendrogram con cutoff = %.4f\n', cutoff);
figure(figNum); figNum = figNum + 1;
[H, T_order, perm] = dendrogram(best_Z, 0, 'ColorThreshold', cutoff);
title(sprintf('Samples Dendrogram - %s + %s (cutoff=%.3f)', ...
    best_linkage, best_distance, cutoff));
xlabel('Sample index');
ylabel('Linkage distance');
grid on;

fprintf('FIGURA 2: PC1 vs PC2 colorati per cluster\n');
figure(figNum); figNum = figNum + 1;
gscatter(scores(:,1), scores(:,2), best_T);
title(sprintf('PC1 vs PC2 colored by clusters (cutoff=%.3f, %s+%s)', ...
    cutoff, best_linkage, best_distance));
xlabel('PC1');
ylabel('PC2');
legend('Location', 'best');
grid on;

fprintf('FIGURA 3: Clustering variabili + imagesc ordinata\n');
% Clustering delle variabili (trasposta) con correlazione
datav = data';
Zv = linkage(datav, 'average', 'correlation');

% Ottieni ordine dal dendrogramma delle variabili
figure('Visible', 'off');
[~, ~, permv] = dendrogram(Zv, 0, 'Orientation', 'left');
close(gcf);
var_order = permv;

% Figura combinata 2x2 come richiesto dalla professoressa
figure(figNum); figNum = figNum + 1;
set(gcf, 'Position', [100, 100, 1200, 800]);

% Subplot 1: vuoto (o titolo)
subplot(2, 2, 1);
axis off;
text(0.5, 0.5, sprintf('Olive Oil Clustering\n%s + %s\nCutoff=%.3f\nSilhouette=%.3f', ...
    best_linkage, best_distance, cutoff, best_silhouette), ...
    'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');

% Subplot 2: Dendrogramma campioni (samples) - ORIZZONTALE IN ALTO
subplot(2, 2, 2);
dendrogram(best_Z, 0, 'ColorThreshold', cutoff);
title('Sample Dendrogram');
ylabel('Distance');
xlabel('');
set(gca, 'XTickLabel', []);

% Subplot 3: Dendrogramma variabili (variables) - VERTICALE A SINISTRA
subplot(2, 2, 3);
dendrogram(Zv, 0, 'Orientation', 'left');
title('Variable Dendrogram (correlation)');
xlabel('Distance');
set(gca, 'YTickLabel', []);

% Subplot 4: imagesc dei dati ordinati secondo entrambi i dendrogrammi
subplot(2, 2, 4);
imagesc(data(perm, var_order)');
colormap('jet');
colorbar;
title('Standardized data (ordered by dendrograms)');
xlabel('Samples (ordered)');
ylabel('Variables (ordered)');
axis tight;

%% =========================================================================
%% PARTE 3: K-MEANS
%% =========================================================================
fprintf('\n--- PARTE 3: K-MEANS ---\n');

fprintf('Eseguendo k-means con k=%d...\n', nClusters);
[idx_kmeans, C] = kmeans(data, nClusters, 'Replicates', 10);
sil_kmeans = mean(silhouette(data, idx_kmeans));
fprintf('K-means silhouette = %.4f\n', sil_kmeans);

fprintf('FIGURA: PC1 vs PC2 per k-means\n');
figure(figNum); figNum = figNum + 1;
gscatter(scores(:,1), scores(:,2), idx_kmeans);
title(sprintf('PC1 vs PC2 - K-means (k=%d, Sil=%.3f)', nClusters, sil_kmeans));
xlabel('PC1');
ylabel('PC2');
legend('Location', 'best');
grid on;

fprintf('FIGURA: Silhouette plot k-means\n');
figure(figNum); figNum = figNum + 1;
silhouette(data, idx_kmeans);
title(sprintf('Silhouette K-means (mean=%.3f)', sil_kmeans));

%% =========================================================================
%% PARTE 4: DBSCAN (se disponibile PLS Toolbox)
%% =========================================================================
fprintf('\n--- PARTE 4: DBSCAN ---\n');

cls_dbscan = [];
epsdist = NaN;
dbscan_available = false;

try
    minpts = 5;
    fprintf('Tentativo DBSCAN con minpts=%d...\n', minpts);
    [cls_dbscan, epsdist] = dbscan(data, minpts);
    dbscan_available = true;
    
    n_clusters_dbscan = length(unique(cls_dbscan(cls_dbscan>0)));
    n_noise = sum(cls_dbscan == 0);
    fprintf('DBSCAN: eps automatico = %.4f, %d cluster + %d noise points\n', ...
        epsdist, n_clusters_dbscan, n_noise);
    
    fprintf('FIGURA: PC1 vs PC2 per DBSCAN\n');
    figure(figNum); figNum = figNum + 1;
    gscatter(scores(:,1), scores(:,2), cls_dbscan);
    title(sprintf('PC1 vs PC2 - DBSCAN (minpts=%d, eps=%.3f, %d clusters)', ...
        minpts, epsdist, n_clusters_dbscan));
    xlabel('PC1');
    ylabel('PC2');
    legend('Location', 'best');
    grid on;
    
    % FIGURA: Silhouette per DBSCAN (solo punti non-noise)
    if n_clusters_dbscan > 1
        fprintf('FIGURA: Silhouette DBSCAN\n');
        figure(figNum); figNum = figNum + 1;
        valid_idx = cls_dbscan > 0;
        if sum(valid_idx) > 0
            sil_dbscan = silhouette(data(valid_idx, :), cls_dbscan(valid_idx));
            title(sprintf('Silhouette DBSCAN (mean=%.3f, excluding noise)', mean(sil_dbscan)));
        end
    end
    
catch ME
    fprintf('DBSCAN non disponibile o errore: %s\n', ME.message);
    fprintf('(Probabilmente PLS Toolbox non installato)\n');
end

%% =========================================================================
%% PARTE 5: OPTICS
%% =========================================================================
fprintf('\n--- PARTE 5: OPTICS ---\n');

k_optics = round(nSamples / 25);
fprintf('Eseguendo OPTICS con k=%d...\n', k_optics);
[RD, CD, order_optics] = optics(data, k_optics);

fprintf('FIGURA: Reachability plot OPTICS\n');
figure(figNum); figNum = figNum + 1;
bar(RD);
title(sprintf('OPTICS Reachability Plot (k=%d)', k_optics));
xlabel('Sample order');
ylabel('Reachability distance');
grid on;

%% =========================================================================
%% PARTE 6: CONFRONTO FINALE TRA METODI
%% =========================================================================
fprintf('\n--- PARTE 6: CONFRONTO GERARCHICO vs K-MEANS ---\n');

% FIGURA: Confronto diretto Hierarchical vs K-means
fprintf('FIGURA: Confronto Hierarchical vs K-means (side-by-side)\n');
figure(figNum); figNum = figNum + 1;
set(gcf, 'Position', [100, 100, 1400, 500]);

subplot(1, 2, 1);
gscatter(scores(:,1), scores(:,2), best_T);
title(sprintf('Hierarchical (%s+%s)\nSilhouette=%.3f', ...
    best_linkage, best_distance, best_silhouette));
xlabel('PC1'); ylabel('PC2');
legend('Location', 'best');
grid on;

subplot(1, 2, 2);
gscatter(scores(:,1), scores(:,2), idx_kmeans);
title(sprintf('K-means (k=%d)\nSilhouette=%.3f', nClusters, sil_kmeans));
xlabel('PC1'); ylabel('PC2');
legend('Location', 'best');
grid on;

% FIGURA: Confronto silhouette
fprintf('FIGURA: Confronto silhouette tra metodi\n');
figure(figNum); figNum = figNum + 1;

if dbscan_available && ~isempty(cls_dbscan)
    set(gcf, 'Position', [100, 100, 1400, 500]);
    
    subplot(1, 3, 1);
    silhouette(data, best_T);
    title(sprintf('Hierarchical\n%s+%s\nMean=%.3f', best_linkage, best_distance, best_silhouette));
    
    subplot(1, 3, 2);
    silhouette(data, idx_kmeans);
    title(sprintf('K-means\nk=%d\nMean=%.3f', nClusters, sil_kmeans));
    
    subplot(1, 3, 3);
    valid_idx = cls_dbscan > 0;
    if sum(valid_idx) > 0
        sil_db = silhouette(data(valid_idx, :), cls_dbscan(valid_idx));
        title(sprintf('DBSCAN\n(no noise)\nMean=%.3f', mean(sil_db)));
    end
else
    set(gcf, 'Position', [100, 100, 1000, 500]);
    
    subplot(1, 2, 1);
    silhouette(data, best_T);
    title(sprintf('Hierarchical\n%s+%s\nMean=%.3f', best_linkage, best_distance, best_silhouette));
    
    subplot(1, 2, 2);
    silhouette(data, idx_kmeans);
    title(sprintf('K-means\nk=%d\nMean=%.3f', nClusters, sil_kmeans));
end

%% =========================================================================
%% RIEPILOGO FINALE
%% =========================================================================
fprintf('\n========================================\n');
fprintf('=== RIEPILOGO RISULTATI FINALI ===\n');
fprintf('========================================\n');
fprintf('Dataset: %d campioni, %d variabili\n', nSamples, nVars);
fprintf('\n1. CLUSTERING GERARCHICO (MIGLIORE):\n');
fprintf('   Linkage: %s\n', best_linkage);
fprintf('   Distanza: %s\n', best_distance);
fprintf('   Cutoff: %.4f\n', cutoff);
fprintf('   Silhouette medio: %.4f\n', best_silhouette);
fprintf('   Numero cluster: %d\n', nClusters);

fprintf('\n2. K-MEANS:\n');
fprintf('   k: %d\n', nClusters);
fprintf('   Silhouette medio: %.4f\n', sil_kmeans);
fprintf('   Differenza vs Hierarchical: %.4f\n', abs(sil_kmeans - best_silhouette));

if dbscan_available && ~isempty(cls_dbscan)
    valid_idx = cls_dbscan > 0;
    if sum(valid_idx) > 0
        sil_db = mean(silhouette(data(valid_idx, :), cls_dbscan(valid_idx)));
        fprintf('\n3. DBSCAN:\n');
        fprintf('   minpts: %d\n', minpts);
        fprintf('   eps: %.4f\n', epsdist);
        fprintf('   Cluster trovati: %d\n', length(unique(cls_dbscan(cls_dbscan>0))));
        fprintf('   Noise points: %d\n', sum(cls_dbscan == 0));
        fprintf('   Silhouette medio (no noise): %.4f\n', sil_db);
    end
end

fprintf('\n4. OPTICS:\n');
fprintf('   k parameter: %d\n', k_optics);
fprintf('   (Vedi reachability plot per identificare cluster)\n');

fprintf('\n========================================\n');
fprintf('CONCLUSIONE CONFRONTO:\n');
if sil_kmeans > best_silhouette
    fprintf('>>> K-means ha performance MIGLIORE (Δsil = +%.4f)\n', sil_kmeans - best_silhouette);
else
    fprintf('>>> Hierarchical ha performance MIGLIORE (Δsil = +%.4f)\n', best_silhouette - sil_kmeans);
end
fprintf('========================================\n');

fprintf('\nTotale figure generate: %d\n', figNum - 1);
fprintf('\n=== ANALISI COMPLETA TERMINATA ===\n');
