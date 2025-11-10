% del Machine learning toolbox
%% per usare k means, seguite istruzioni in
% doc kmeans 
% potete seguire esempio in clusterdemo.m
% Per olive oil
olivedata = readmatrix('oliveoil.csv');
data=normalize(olivedata); % autoscaling
[idx,C] = kmeans(data,3);  % cerco 3 cluster
% in idx c'è l'indice di cluster
% in C cisono le coordinate dei centroidi dei cluster rispetto alle
% variabili originali
%% per i plot procedere con PCA 
% fai PCA del data set con 2 PCs
[u s v]=svds(data,2);
scores=u*s;
% fai uno scatter plot scores PC1 PC2e usa come colore indice di cluster
figure;gscatter(scores(:,1),scores(:,2),idx);
title('PC1 vs PC2 colored by k-means clusters');
xlabel('PC1'); ylabel('PC2');
% indice silhouette
figure; silhouette(data,idx)


%% per dbscan usate PLS toolbox

clear classes
evrimovepath('top')
% help dbscan
% del PLS Toolbox
% help dbscan 
minpts = 3 ; % minpoints da decidere dall'utente è consigliato minimo 3
[cls,epsdist] = dbscan(data, minpts);  
% in cls c'è l'indice di cluster
% in epsdist c'è il raggio eps stimato come migliore dall'algoritmo
% se volete provare raggi diversi:
eps=0.8; % mettete il valore che volete provare es:1
cls1 = dbscan(data, minpts,eps);  
%% per i plot procedere con PCA 
% fai PCA del data set con 2 PCs
[u s v]=svds(data,2);
scores=u*s;
% fai uno scatter plot scores PC1 PC2e usa come colore indice di cluster
figure;gscatter(scores(:,1),scores(:,2),cls);
title('PC1 vs PC2 colored by DBSCAN clusters');
xlabel('PC1'); ylabel('PC2');

%% optics in teams file
olivedata = readmatrix('oliveoil.csv');
data=normalize(olivedata);
[RD CD order]=optics(data,572/25);
% Plot reachability
figure;bar(RD)
title('OPTICS Reachability Plot');
xlabel('Sample order');
ylabel('Reachability distance');