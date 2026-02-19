function [modsimca, pred]=torunSIMCA(X,icl,maxnpc,Xpred,icltest)
% if using aknowledge:  prof. Federico Marini Università Roma La Sapienza
% Prof. Marina Cocchi University of Modena and Reggio Emilia
% da scrivere in command line
%[modsimca, pred]=torunSIMCA(X,icl,maxnpc,Xpred,icltest)

%% Input
% se X non è un dataset con indice di classe già inserito
% dare anche icl= indice numerico di classe del training, se no icl=[]
% optional input: Xpred (set da predirre)
%                       icltest (se l'indice di classe è noto anche per il test
%                                  e Xpred non è un dataset con l'indice di
%                                  calsse già inserito, altrimenti icltest=[])
% maxnpc = numero massimo di componenti da considerare in CV
%                  deve essere < del valore più basso tra numero di oggetti
%                  della classe meno numerosa e numero di colonne della
%                  matrice

%% OUTPUT
% modsimca
% è uno structure array con diversi fields:
% modsimca.finalmodel.SIMCA.sensspec 
%   array nclassi x nclassi che contiene
%   sensibilità per ciascuna classe nella diagonale
%   specificità negli elementi extradiagonali
% modsimca.finalmodel.SIMCA.ncomp
%    numero di componenti usate nei modelli finali PCA per ciascuna classe
% modsimca.finalmodel.SIMCA.predclass
%    classe a cui è assegnato ciascun campione dal modello
% modsimca.finalmodel.PCmodels
%    un cell array con tatne celle quante classi , contiene i modelli PCA
%    finali di ciascuna classe
%    esempio per la classe 1
% modsimca.finalmodel.PCmodels{1}
%       prepr: {1×2 cell}
%          res: [150×4 double]
%       scores: [150×2 double]
%        loads: [4×2 double]
%         eigs: [4×1 double]
%     variance: {[4×1 double]  [4×1 double]}
%     critvals: [6.5144 2.9460]
%          tsq: [150×1 double]
%         qres: [1×150 double]

% richiede PLS_toolbox e Machine Learning installati
% evrimovepath('top')  % to get PLS toolbox primo nel path
%% crea structure array con le options di default (modificabili)
opt=simcafCV_2020('options');
% metti in opt le options desiderate

% esempio per CV venetian blind con 5 splits
opt.cv.cvtype='syst123';    %  altrimenti 'full' [leave one out], 'syst111' [contigui], 'random' [random]
opt.cv.cvsegments=5; %numero di split

%esempio per autoscaling preprocess
opt.preproc='auto'; 
%per pareto scaling (sqrt(dev.st) come scaling )
% opt.preproc='autop';
% per mean center
%opt.preproc='mean'
% per non usare nessun preprocessing
%opt.preproc='none'

%% Prepara X come dataset se non lo è,  fa CV e salva modello finale
if isdataset(X)
    ncl=max(X.class{1});
else
    X=dataset(X);
    ncl=max(icl);
    X.class{1}=icl;
end
icl=X.class{1};
% lancia crossvalidazione 
% maxnpc il numero massimo di PC che vuoi considerare
opt.plots='on';
modsimca=simcaf_mainCV2020(X,maxnpc,opt);  % se X è un dataset structure del PLStoolbox
% la routine ti genera il grafico per sciegliere il numero di PC
% e ti salva in modsimca, i modelli per i cicli di CV ed il modello finale
% con il nuemero PC scelto in
% modsimca.finalmodel
%% chiude figure generate
% close all
%% genera tabella con valori sens spec per il training
sensspec=modsimca.finalmodel.SIMCA.sensspec;
Xtr=X;
ncl=max(Xtr.class{1});
for i=1:ncl
    nc(i)=nnz(find(Xtr.class{1}==i));
    sens_tr(i)=sensspec(i,i)/nc(i);
end
for i=1:ncl
    j=setdiff([1:ncl],i);
    for k=1:ncl-1
        spec_tr(i,j(k))=sensspec(i,j(k))/nc(j(k));
    end
end
nomi=cellstr(strcat('Class ',int2str(unique(Xtr.class{1})')));
Tsens_train=array2table(sens_tr);
Tsens_train.Properties.VariableNames=nomi;
Tsens_train.Properties.RowNames={'Sensitivity'};
Tspec_train=array2table(spec_tr);
Tspec_train.Properties.VariableNames=nomi;
for i=1:ncl
    Tspec_train.Properties.RowNames{i}=strcat('Specificity of C',int2str(i),'vs.');
end
save  sensspecTrain Tsens_train Tspec_train
%% se volete calcolare discriminant power delle variabili e fare il plot
% se nel plot il 95 percentile non si vede vuol dire che coincide con il
% 99 percentile
dpow=calcdispow_tocorrect(modsimca, Xtr.class{1}, Xtr.label{2});
save dpow dpow

%% se volete fare il plot dei loadings 
figure;
ncomp=modsimca.finalmodel.SIMCA.ncomp;
cc=0;
for j=1:ncl
  for nc=1:ncomp(j)
      cc=cc+1;
      subplot(ncl,max(ncomp),cc);plot(modsimca.finalmodel.PCmodels{1,j}.loads(:,nc),'-o');
      ylabel(strcat('Loadings C',int2str(j),' PC ',int2str(nc)));
      a=gca;
      a.XTickMode='Manual';
      a.XTick=[1:size(Xtr,2)];
      a.XTickLabel=Xtr.label{2};      
  end
  cc=max(cc,max(ncomp)*j);
end
name='loadings_plot';
hgsave(name);
close all
%% Predizioni su  test set esterno

if nargin >3 & ~isempty(Xpred)
    if ~isdataset(Xpred)
        Xpred=dataset(Xpred);
        if ~isempty(icltest)
            Xpred.class{1}=icltest;
        end
    end
    icltest=Xpred.class{1};
    
    pred=simcaf_pred2020(Xpred,modsimca.finalmodel);
    close all
    %% 
    % per fare plot calibration and test sets insieme, avendo salvate nel folder dove si è
    % le figure con il grafico in per il calibration set
    if ncl <=7
       markers=['b';'g';'r';'c';'m';'y';'k'];
    else
      markers=[0 0 1;0 1 0;1 0 0;1 0 1;0 1 1; 0 0 0; 0.6 0 0.8; 1 0.5 0; 0.1 0.4 1; 0.7 0.2 0.2;0.9 0.8 0.2];
    end
    for ic=1:ncl
        lim=pred.PCmodels{ic}.critvals;
        t2lim=lim(1);
        qlim=lim(2);
        tsq=pred.PCmodels{ic}.tsq;
        q=pred.PCmodels{ic}.qres;
        tsq_tr=modsimca.finalmodel.PCmodels{ic}.tsq;
        q_tr=modsimca.finalmodel.PCmodels{ic}.qres;
        figure; 
        gscatter(tsq_tr./t2lim,q_tr./qlim,icl,markers,'o');
        figname=strcat('T2rQrplot_C',int2str(ic),'.fig');
        %openfig(figname);
        % rimetto scala lineare
        a=0:0.0001:sqrt(2);
        hold on; plot(a,sqrt(2-a.^2),'-r');
        axis ([0 5 0 5]); 
        xlabel('score distance'); ylabel('orthogonal distance');
        title(['simca model for the category: ', num2str(ic)]);
        ax=gca;
        ax.XAxis.Scale = 'linear';
        ax.YAxis.Scale = 'linear';
        
        %
        hold on
        gscatter(tsq./t2lim,q./qlim,icltest,markers,'d')  % per validazione stessa classe
        xlabel('score distance'); ylabel('orthogonal distance');
%         % scala logaritma per vedere anche i punti più distanti
        ax=gca;
        ax.XAxis.Scale = 'log';
        ax.YAxis.Scale = 'log';
        
        axis ([0.01 10^2 0.01 10^2]);
%         %
        leg=legend;
        legs=leg.String;
        legsn=strcat(legs,' (diamond=test circle=train)');
        leg.String=legsn;
        hgsave(strcat(figname(1,1:end-4),'_trainetest'));
        close
    end
    %% genera tabella con valori sens spec per il test
    sensspec=pred.SIMCA.sensspec;
    Xts=Xpred; % 
    ncl=max(Xts.class{1});
    for i=1:ncl
        nc(i)=nnz(find(Xts.class{1}==i));
        sens_ts(i)=sensspec(i,i)/nc(i);
    end
    for i=1:ncl
        j=setdiff([1:ncl],i);
        for k=1:ncl-1
            spec_ts(i,j(k))=sensspec(i,j(k))/nc(j(k));
        end
    end
    nomi=cellstr(strcat('Class ',int2str(unique(Xtr.class{1})')));
    Tsens_test=array2table(sens_ts);
    Tsens_test.Properties.VariableNames=nomi;
    Tsens_test.Properties.RowNames={'Sensitivity'};
    Tspec_test=array2table(spec_ts);
    Tspec_test.Properties.VariableNames=nomi;
    for i=1:ncl
        Tspec_test.Properties.RowNames{i}=strcat('Specificity of C',int2str(i),'vs.');
    end
    save sensspecTest Tspec_test Tsens_test
else
    pred=[];
end
