% prima di eseguire il codice sostituire a X il nome del dataset usato
% e del modello nel caso non si chiami modsimca
X=oliv_tr; % cambiare nome se cambia data set
 Nsamples=size(X,1);
ncl=max(X.class{1});
model=modoliv; % cambiare nome se cambia il nom edle modello
%% mostra risultati in CV
sesp=cat(3,model.CrossVal.sensspec{:});
        for ic=1:ncl
            cnam{ic}=strcat('Class',int2str(ic));
            nc=nnz(find(X.class{1}==ic));
            sens(ic,:)=squeeze(sesp(ic,ic,:))./nc;
            spec(ic,:)=squeeze(sum(sesp(ic,setdiff(1:ncl,ic),:),2))./(Nsamples-nc);
        end
 for i=1:size(sens,2);namPC{i}=strcat('PC ',int2str(i));end
Tsens=array2table(sens);
Tsens.Properties.RowNames=cnam;
Tsens.Properties.VariableNames=namPC;
Tspec=array2table(spec);
Tspec.Properties.RowNames=cnam;
Tspec.Properties.VariableNames=namPC';
display('Sensitivity')
display(Tsens)
display('Specificity')
display(Tspec)
save SensSpecTables Tsens Tspec
%
%% per il test
sensspec=predoliv.SIMCA.sensspec; % cambia nome se cambia nome il modello di predizione
Xts=oliv_ts; % cambia nome se cambia il nome del test set
ncl=max(Xts.class{1});
for i=1:ncl
    nc(i)=nnz(find(Xts.class{1}==i));
    sens_ts(i)=sensspec(i,i)/nc(i);
end
for i=1:ncl
    j=setdiff([1:ncl],i)
    for k=1:ncl-1
        spec_ts(i,j(k))=sensspec(i,j(k))/nc(j(k));
    end
end
save sensspecTest sens_ts spec_ts
%%
%% per il training
sensspec=modoliv.finalmodel.SIMCA.sensspec; % cambia nome se il modello si  chiama in modo diverso
Xtr=oliv_tr;
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
save  sensspecTrain sens_tr spec_tr


