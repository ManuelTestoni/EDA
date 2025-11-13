% del Machine learning toolbox
%% per usare k means, seguite istruzioni in
 % potete seguire esempio in clusterdemo.m
% Adattato per Olive Oil dataset
load oliveoil.mat
data=normalize(olivdata); % autoscaling
[nSamples, nVars] = size(data);
fprintf('Dataset: %d campioni x %d variabili\n', nSamples, nVars);

% NOTA: Il dataset Olive Oil ha 8 categorie naturali (varietà di olio)
% Puoi testare con k=3 (macro-gruppi) o k=8 (categorie reali)
k_clusters = 8;  % Cambia questo valore: 3, 5, 8, etc.
fprintf('K-means con k=%d cluster\n', k_clusters);

[idx,C] = kmeans(data, k_clusters, 'Replicates', 10);  % cerco k cluster
% in idx c'è l'indice di cluster
% in C cisono le coordinate dei centroidi dei cluster rispetto alle
% variabili originali

%% per i plot procedere con PCA 
% fai PCA del data set con 2 PCs
[u s v]=svds(data,2);
scores=u*s;

% fai uno scatter plot scores PC1 PC2 e usa come colore indice di cluster
figure;
gscatter(scores(:,1),scores(:,2),idx);
title(sprintf('K-means Clustering - PC1 vs PC2 (k=%d)', k_clusters));
xlabel('PC1'); ylabel('PC2');
if k_clusters <= 10
    legend('Location', 'best');  % Solo se non troppi cluster
end
grid on;

% NOTA: Olive Oil dataset non ha etichette di categoria (a differenza di Iris)
% quindi non possiamo fare confronto con "classi vere"

% indice silhouette
figure; 
silhouette(data,idx);
title(sprintf('Silhouette K-means (mean=%.3f)', mean(silhouette(data,idx))));


%% per dbscan usate PLS toolbox

clear classes
evrimovepath('top')
% del PLS Toolbox
% help dbscan 
load oliveoil.mat
minpts = 5 ; % minpoints da decidere dall'utente (consigliato 3-5)
data=normalize(olivdata); % autoscaling

fprintf('\n=== DBSCAN ===\n');
fprintf('minpts: %d\n', minpts);

try
    [cls,epsdist] = dbscan(data, minpts);  
    % in cls c'è l'indice di cluster (0 = noise)
    % in epsdist c'è il raggio eps stimato come migliore dall'algoritmo
    
    n_clusters = length(unique(cls(cls>0)));
    n_noise = sum(cls == 0);
    fprintf('DBSCAN automatico: eps=%.4f, %d clusters, %d noise points\n', epsdist, n_clusters, n_noise);
    
    % Se trova 1 solo cluster, prova valori eps più piccoli manualmente
    if n_clusters == 1
        fprintf('\n⚠️  DBSCAN ha trovato solo 1 cluster!\n');
        fprintf('Provo con eps più piccolo per trovare più cluster...\n\n');
        
        % Prova diversi valori di eps
        eps_values = [epsdist*0.3, epsdist*0.5, epsdist*0.7];
        
        for eps_test = eps_values
            cls_test = dbscan(data, minpts, eps_test);
            n_clusters_test = length(unique(cls_test(cls_test>0)));
            n_noise_test = sum(cls_test == 0);
            fprintf('  eps=%.4f: %d clusters, %d noise points\n', eps_test, n_clusters_test, n_noise_test);
            
            % Usa il primo che trova 2+ cluster
            if n_clusters_test > 1 && n_clusters_test < 10
                cls = cls_test;
                epsdist = eps_test;
                n_clusters = n_clusters_test;
                n_noise = n_noise_test;
                fprintf('✓ Usato eps=%.4f per clustering\n\n', eps_test);
                break;
            end
        end
    end
    
    %% per i plot procedere con PCA 
    % fai PCA del data set con 2 PCs
    [u s v]=svds(data,2);
    scores=u*s;
    
    % fai uno scatter plot scores PC1 PC2 e usa come colore indice di cluster
    figure;
    gscatter(scores(:,1),scores(:,2),cls);
    title(sprintf('DBSCAN Clustering - PC1 vs PC2 (eps=%.3f, %d clusters)', epsdist, n_clusters));
    xlabel('PC1'); ylabel('PC2');
    legend('Location', 'best');
    grid on;
    
    % Grafico diagnostico: k-distance per scegliere eps ottimale
    fprintf('\nCalcolo k-distance plot per diagnostica...\n');
    D = pdist(data, 'euclidean');
    D_matrix = squareform(D);
    k_dist = zeros(size(data,1), 1);
    for i = 1:size(data,1)
        sorted_dist = sort(D_matrix(i,:));
        k_dist(i) = sorted_dist(minpts+1); % +1 perché esclude se stesso
    end
    k_dist_sorted = sort(k_dist, 'descend');
    
    figure;
    plot(k_dist_sorted, 'b-', 'LineWidth', 1.5);
    hold on;
    yline(epsdist, 'r--', sprintf('eps usato: %.3f', epsdist), 'LineWidth', 2);
    title(sprintf('K-distance Plot (k=%d) - Diagnostica DBSCAN', minpts));
    xlabel('Punti ordinati per distanza');
    ylabel(sprintf('%d-esima distanza più vicina', minpts));
    grid on;
    legend('K-distance', 'Eps usato', 'Location', 'best');
    fprintf('Suggerimento: Il "gomito" nel grafico indica eps ottimale\n');
    
    % Silhouette (solo per punti non-noise)
    if n_clusters > 1
        valid_idx = cls > 0;
        if sum(valid_idx) > 0
            figure;
            silhouette(data(valid_idx, :), cls(valid_idx));
            title(sprintf('Silhouette DBSCAN (mean=%.3f, excluding noise)', mean(silhouette(data(valid_idx,:), cls(valid_idx)))));
        end
    else
        fprintf('⚠️  Solo 1 cluster trovato - Silhouette non calcolabile\n');
    end
    
catch ME
    fprintf('ERRORE DBSCAN: %s\n', ME.message);
    fprintf('(Probabilmente PLS Toolbox non disponibile)\n');
end

%% optics (vedi file optics.m nella cartella)
load oliveoil.mat
data=normalize(olivdata);
[nSamples, ~] = size(data);

% k parameter per OPTICS (consigliato: numero_campioni/20 o /25)
% Valori tipici: 10-30 per dataset di dimensioni medie
k_optics = round(nSamples / 25);
fprintf('\n=== OPTICS ===\n');
fprintf('Dataset: %d campioni\n', nSamples);
fprintf('k parameter: %d (= %d/25)\n', k_optics, nSamples);
fprintf('Nota: k più basso → cluster più piccoli, k più alto → cluster più grandi\n\n');

[RD, CD, order] = optics(data, k_optics);

% Reachability plot (mostra struttura clustering)
figure;
bar(RD);
title(sprintf('OPTICS Reachability Plot (k=%d)', k_optics));
xlabel('Sample order');
ylabel('Reachability distance');
grid on;

% Scatter plot PC1 vs PC2 colorato per reachability distance
[u s v]=svds(data,2);
scores=u*s;

figure;
scatter(scores(:,1), scores(:,2), 50, RD, 'filled');
colorbar;
colormap('jet');
title(sprintf('OPTICS - PC1 vs PC2 colored by reachability (k=%d)', k_optics));
xlabel('PC1');
ylabel('PC2');
grid on;

% NOTA: Olive Oil non ha "classi vere" quindi non possiamo colorare per categoria
% Se avessi category, useresti:
% cat=categorical(category);
% cat=double(cat);
% figure;for i=1:nSamples;hold on;bar(i,RD(i),color(cat(order(i))));end