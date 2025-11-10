% data set da analizzare:

%1) Olive oil dataset

% Carica dati olive oil
olivedata = readmatrix('oliveoil.csv');
data = normalize(olivedata);


%% cluster nel PLS Toolbox:  prova e confronta i risultati di hierarchical agglomerative con diversi criteri di linkage 

%% If PLS Toolbox is first in setpath
% per avere informazioni :
% doc cluster
% otherwise see PDF file in team CLusterPLSToolbox.pdf for documentation

%% Hierarchical Agglomerative

% gcluster  % se volete usare l'interfaccia grafica

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
% mettere PLS Toolbox last in setpath
clear classes
evrimovepath('bottom');
% now the Statistical and Machine Leanring Toolbox can be used
% Per avere informazioni
% doc clusterdata

% esempio con data set olive oil autoscalato T contiene l'indice di cluster
% trovato 
olivedata = readmatrix('oliveoil.csv');
data=normalize(olivedata);
T=clusterdata(data,'linkage','ward','distance','euclidean','maxclust',3);
% maxclust indica il numero massimo di clusters che si vogliono definire,
% in questo modo il threshold è scelto automaticamente per avere quel
% numero di clusters. Altrimenti si può usare un valore di threshold
% desiderato scelto in base al dendrogramma
% to obtain the dendogram
Z=linkage(data,'ward');
dendrogram(Z,0) % mette tutti i campioni
% dendrogram(Z,100)  %mette solo 100 nodi
cutoff=9.7; % scegliere in base al dendrogramma
dendrogram(Z,0,'ColorThreshold',cutoff);
T=clusterdata(data,'criterion','distance','linkage','ward','distance','euclidean','Cutoff',cutoff);


%% to see the effect of clustering
% fai PCA del data set con 2 PCs
[u s v]=svds(data,2);
scores=u*s;
% fai uno scatter plot scores PC1 PC2e usa come colore indice di cluster
figure;gscatter(scores(:,1),scores(:,2),T);
title('PC1 vs PC2 colored by cluster assignment');
xlabel('PC1'); ylabel('PC2');


%% to do clustering on variables
olivedata = readmatrix('oliveoil.csv');
data=normalize(olivedata);
datav=data'; % matrice trasposta
Tv=clusterdata(datav,'linkage','single','distance','correlation','maxclust',2);
Zv=linkage(datav,'single','correlation');
cutoff=0.1;
figure; dendrogram(Zv,0,'ColorThreshold', cutoff); %variabili
% ottieni l'ordine delle vfigure; 
av=gca;
labelv=str2num(av.XTickLabel);
ordvar=labelv(end:-1:1);
% fai il plot con cluster campioni, cluster variabili e matrice ordinata
figure; subplot(2,2,2); dendrogram(Z,0,'ColorThreshold',cutoff);
ao=gca;
label_o=str2num(ao.XTickLabel);
hold on;subplot(2,2,3);dendrogram(Zv,0,'orientation','left');
hold on;subplot(2,2,4);imagesc(data(label_o,ordvar)'); 
ai=gca;
ai.YTickLabel=ordvar;
colormap('jet');
% per mettere i numeri dei campioni nell'ordine del cluster
% (se sono molti non si leggono)
ai.XTickMode='manual';
ai.XTick=[1:size(data,1)];
ai.XTickLabel=label_o;
ai.XTickLabelRotation=90;








