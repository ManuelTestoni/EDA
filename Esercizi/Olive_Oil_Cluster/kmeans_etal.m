% del Machine learning toolbox
%% per usare k means, seguite istruzioni in
doc kmeans 
% potete seguire esempio in clusterdemo.m
% Per fisheriris
data=normalize(meas); % autoscaling
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
% confronta qs plot con uno colorato con l'indice delle classi vere in
% category
figure;gscatter(scores(:,1),scores(:,2),species);
% indice silhouette
figure; silhouette(data,idx)


%% per dbscan usate PLS toolbox

clear classes
evrimovepath('top')
help dbscan
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
% confronta qs plot con uno colorato con l'indice delle classi vere in
% category
figure;gscatter(scores(:,1),scores(:,2),species);

%% optics in teams file
load fisheriris
category=species;
[RD CD order]=optics(data,150/25);
cat=categorical(category);
cat=double(cat);
if unique(cat)<=7
color=['b';'r';'g';'c';'m';'y';'k'] % colori fino a 7 classi
else
color =colormap('jet');
color=color(1:25:end,:);
end
figure;bar(RD)
% colorata per le classi "vere"
figure;for i=1:150;hold on;bar(i,RD(i),color(cat(order(i))));end