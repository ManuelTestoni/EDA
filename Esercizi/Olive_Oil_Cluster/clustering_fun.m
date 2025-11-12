
% data set da analizzare:

%1) Adattato per Olive Oil dataset (originariamente per Fisher Iris)

load oliveoil.mat
% oliveoil.mat contiene la variabile 'olivdata' con i dati numerici
fprintf('Caricato oliveoil.mat - variabile: olivdata\n');

%% cluster nel PLS Toolbox:  prova e confronta i risultati di hierarchical agglomerative con diversi criteri di linkage 

%% If PLS Toolbox is first in setpath
% per avere informazioni :
% otherwise see PDF file in team CLusterPLSToolbox.pdf for documentation

%% Hierarchical Agglomerative

%gcluster  % se volete usare l'interfaccia grafica

% %% Per usare  command line
% options=cluster('options');
% 
% % In otpions ci sono i set di default per le diverse opzioni , che poi potete cambiare 
% %  sotto la lista di possibili opzioni da mettere in options variable che è structure array with fields:
% 
% % options.plots: 'final' % govern level of plotting
% 
% options.algorithm = 'ward'; % defaut is ward can be changed in:
% % 'knn' (single linkage) 'fn' (complete linkage) 'med'(centroidi pesati per numero oggetti)  'avgpair' (average) 'cnt' (centroid)
% 
% %  options.preprocessing: {[]} % quale preprocessing applicare ai dati il default è
% %  nessuno ma è consigliato fare lo stesso preprocessing che usereste in
% %  PCA
% % per riempire preprocessing usate:
% p=preprocess; % si apre un interfaccia grafica dalla quale scegliete il preprocessing desiderato
% options.preprocessing=p;
% 
% %  options.pca: 'off'  % se fare PCA prima di cluster analysis (default è off
% %  quindi no. Se i dati come variabili sono molto numerosi, esempio segnali, conviene farlo
% %  options.ncomp: []   % con quante PCs
% %  options.mahalanobis: 'off'  % se autoscalare le PCs
% 
% %  options.slack: 0         % see documentation
% %  options.maxlabels: 200  
% 
% options.distance= 'euclidean' ; % tipo di distanza può essere anche 'manhattan'
% 
% %% Una volta preparata la variabile options, per lanciare l'algoritmo di cluster
% 
% [results,fig,distances] = cluster(meas,options);
% 
% % se si vuole colorare per categorie trasformare in dataset la variabile
% % con i dati
% data=dataset(meas);
% % inserire indice di classe
% data.class{1}=species;
% % ripetere il comando
% [results,fig,distances] = cluster(data,options);
% % Per come interpretare i risultati leggere nella documentazione
% 


%% CLustering con il Machine Learning Toolbox
% IMPORTANTE: Rimuovi PLS Toolbox dal path per evitare conflitti
warning('off', 'all'); % Disabilita warning temporaneamente

% Rimuovi PLS Toolbox dal path se presente (evita conflitto con cluster())
try
    evrimovepath('bottom'); % Sposta PLS Toolbox in fondo al path
catch
    % PLS Toolbox non presente, ok
end

% Usa i dati già caricati da oliveoil.mat
% NOTA: olivdata è già una matrice numerica, non serve readtable
data = normalize(olivdata); % autoscaling (zscore normalization)
[nSamples, nVars] = size(data);
fprintf('Dataset normalizzato: %d campioni x %d variabili\n', nSamples, nVars);

%% =========================================================================
%% CLUSTERING ITERATIVO: Testa tutte le combinazioni linkage/distanza
%% =========================================================================

fprintf('\n=== CLUSTERING GERARCHICO ITERATIVO ===\n');

% Liste di metodi da testare
linkage_methods = {'ward', 'single', 'complete', 'average', 'centroid'};
distance_methods = {'euclidean', 'cityblock', 'correlation', 'cosine'};

% Numero di cluster desiderato
nClusters = 3;

% Struttura per salvare risultati
results = struct();
result_idx = 1;

fprintf('Testando combinazioni linkage × distanza:\n');
fprintf('%-12s %-12s %-12s %s\n', 'Linkage', 'Distanza', 'Silhouette', 'Note');
fprintf('%s\n', repmat('-', 60, 1));

figNum = 1;

for i = 1:length(linkage_methods)
    for j = 1:length(distance_methods)
        link_method = linkage_methods{i};
        dist_method = distance_methods{j};
        
        % Ward funziona solo con euclidean
        if strcmp(link_method, 'ward') && ~strcmp(dist_method, 'euclidean')
            fprintf('%-12s %-12s %-12s %s\n', link_method, dist_method, 'N/A', 'Ward richiede euclidean');
            continue;
        end
        
        try
            % Calcola linkage
            Z = linkage(data, link_method, dist_method);
            
            % Estrai cluster
            T = cluster(Z, 'maxclust', nClusters);
            
            % Calcola silhouette
            sil_vals = silhouette(data, T);
            mean_sil = mean(sil_vals);
            
            % Salva risultati
            results(result_idx).linkage = link_method;
            results(result_idx).distance = dist_method;
            results(result_idx).silhouette = mean_sil;
            results(result_idx).Z = Z;
            results(result_idx).T = T;
            results(result_idx).sil_vals = sil_vals;
            
            fprintf('%-12s %-12s %-12.4f %s\n', link_method, dist_method, mean_sil, '✓');
            
            % Genera dendrogramma per questa combinazione
            figure(figNum);
            dendrogram(Z, 0, 'ColorThreshold', 'default');
            title(sprintf('Dendrogram: %s + %s (Silhouette=%.3f)', ...
                link_method, dist_method, mean_sil));
            xlabel('Sample index');
            ylabel('Distance');
            grid on;
            figNum = figNum + 1;
            
            result_idx = result_idx + 1;
            
        catch ME
            fprintf('%-12s %-12s %-12s %s\n', link_method, dist_method, 'ERRORE', ME.message);
        end
    end
end

% Ordina per silhouette (migliore → peggiore)
[~, sort_idx] = sort([results.silhouette], 'descend');
results = results(sort_idx);

fprintf('\n%s\n', repmat('=', 60, 1));
fprintf('RANKING PER SILHOUETTE SCORE:\n');
fprintf('%s\n', repmat('=', 60, 1));
for i = 1:min(5, length(results))
    fprintf('%d. %s + %s: %.4f\n', i, ...
        results(i).linkage, results(i).distance, results(i).silhouette);
end

%% =========================================================================
%% ANALISI DETTAGLIATA DEI 2 MIGLIORI METODI
%% =========================================================================

fprintf('\n=== ANALISI DETTAGLIATA TOP 2 METODI ===\n');

for top_idx = 1:min(2, length(results))
    fprintf('\n--- METODO #%d: %s + %s (Silhouette=%.4f) ---\n', ...
        top_idx, results(top_idx).linkage, results(top_idx).distance, ...
        results(top_idx).silhouette);
    
    % Recupera dati
    Z = results(top_idx).Z;
    T = results(top_idx).T;
    link_method = results(top_idx).linkage;
    dist_method = results(top_idx).distance;
    mean_sil = results(top_idx).silhouette;
    
    % Calcola cutoff per nClusters
    cutoff_idx = size(Z, 1) - nClusters + 2;
    cutoff = Z(cutoff_idx, 3);
    
    % FIGURA 1: Dendrogramma con cutoff evidenziato
    figure(figNum); figNum = figNum + 1;
    dendrogram(Z, 0, 'ColorThreshold', cutoff);
    title(sprintf('Metodo #%d: %s + %s (cutoff=%.3f, Sil=%.3f)', ...
        top_idx, link_method, dist_method, cutoff, mean_sil));
    xlabel('Sample index');
    ylabel('Distance');
    grid on;


    % FIGURA 2: Scatter PC1 vs PC2 colorato per cluster
    [u s v]=svds(data,2);
    scores=u*s;
    var_explained = diag(s).^2 / sum(diag(s).^2) * 100;
    
    figure(figNum); figNum = figNum + 1;
    gscatter(scores(:,1),scores(:,2),T);
    title(sprintf('Metodo #%d: %s + %s - PC1 vs PC2 (Sil=%.3f)', ...
        top_idx, link_method, dist_method, mean_sil));
    xlabel(sprintf('PC1 (%.1f%% variance)', var_explained(1)));
    ylabel(sprintf('PC2 (%.1f%% variance)', var_explained(2)));
    legend('Location', 'best');
    grid on;
    
    % FIGURA 3: Silhouette plot
    figure(figNum); figNum = figNum + 1;
    silhouette(data, T);
    title(sprintf('Metodo #%d: Silhouette plot (mean=%.3f)', top_idx, mean_sil));
end

%% =========================================================================
%% CLUSTERING VARIABILI (per il miglior metodo)
%% =========================================================================

fprintf('\n=== CLUSTERING VARIABILI (metodo migliore) ===\n');


% Usa il MIGLIOR metodo per clustering variabili
best_result = results(1);
Z_best = best_result.Z;
T_best = best_result.T;
link_best = best_result.linkage;
dist_best = best_result.distance;
sil_best = best_result.silhouette;

% Calcola cutoff per il miglior metodo
cutoff_best = Z_best(size(Z_best,1) - nClusters + 2, 3);

% Clustering delle variabili con correlation
datav=data'; % matrice trasposta (ora le variabili sono le righe)
Zv=linkage(datav,'average','correlation');
Tv = cluster(Zv,'maxclust',2);

figure(figNum); figNum = figNum + 1;
dendrogram(Zv,0,'ColorThreshold', 'default');
title('Dendrogramma Variabili (correlation distance)');
xlabel('Variable index');
ylabel('Correlation distance');
grid on;

% ottieni l'ordine delle variabili dal dendrogramma
av=gca;
labelv=str2num(av.XTickLabel);
ordvar=labelv(end:-1:1);

%% FIGURA COMBINATA 2x2 (miglior metodo)
fprintf('Generando figura combinata 2x2...\n');

% Ottieni ordine campioni dal dendrogramma migliore
figure('Visible', 'off');
dendrogram(Z_best, 0);
ao = gca;
label_o = str2num(ao.XTickLabel);
close(gcf);

% Figura finale 2x2
figure(figNum); figNum = figNum + 1;
set(gcf, 'Position', [100, 100, 1200, 800]);

subplot(2,2,1);
axis off;
text(0.5, 0.5, sprintf('Olive Oil Clustering\nMIGLIORE: %s + %s\nSilhouette=%.3f\n%d samples x %d vars', ...
    link_best, dist_best, sil_best, nSamples, nVars), ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');

subplot(2,2,2); 
dendrogram(Z_best,0,'ColorThreshold',cutoff_best);
title(sprintf('Sample Dendrogram (%s+%s)', link_best, dist_best));
ylabel('Distance');

subplot(2,2,3);
dendrogram(Zv,0,'orientation','left');
title('Variable Dendrogram (correlation)');
xlabel('Distance');

subplot(2,2,4);
imagesc(data(label_o,ordvar)'); 
colormap('jet');
colorbar;
title('Heatmap (ordered by dendrograms)');
xlabel('Samples (ordered)');
ylabel('Variables (ordered)');
ai=gca;
ai.YTickLabel=ordvar;

fprintf('\n=== ANALISI GERARCHICA COMPLETATA ===\n');
fprintf('Totale figure generate: %d\n', figNum-1);
fprintf('Miglior metodo: %s + %s (Silhouette=%.4f)\n\n', link_best, dist_best, sil_best);