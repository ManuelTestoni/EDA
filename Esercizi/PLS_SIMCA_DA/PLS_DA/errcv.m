%classification error in crossvalidation
function [err_output, pred]= errcv(cvpred,Y,model,test,Xtest,Ytest)
% calculate correct classification rate and calssification error for NPLSDA
% model
%% input
% if only error in CV are required , i.e. to choose the best number of LV
% give in input: [] for model and 0 for test
% do not give Xtest, Ytest so:
% err_output= errcv(cvpred,Y,[],0);

% if you want for each LV also the error for the training then:
% err_output= errcv(cvpred,Y,1,0);
% 

% if you want to save the error for the test set  then:
% err_output= errcv(cvpred,Y,model,1,Xtest,Ytest);
% model is the NPLSDA model calulated with the number of CV you choose
% Xtest is the array with the test set, Ytest contain the classes
%  for the test set

if ~isempty(model)
    train=1;
else
    train=0;
end
    
[no nc]=size(Y);
nlv=size(cvpred.cvpred,3);
for i=1:nc
    ii =find(Y(:,i)==1);
    indc(i)=length(ii);
    ic(ii)=i;
end
for i=1:nlv
    for k=1:no
        [a iac(k,i)]=max(squeeze(cvpred.cvpred(k,:,i)));
        if train==1
            [b iatrain(k,i)]=max(squeeze(cvpred.cpred(k,:,i)));
        end
    end
end
diff=(iac-ic'*ones(1,nlv));
if train==1; difftrain=(iatrain-ic'*ones(1,nlv)); end
for i=1:nlv
    for j=1:nc
        err(j,i) = nnz(diff(find(Y(:,j)==1),i));
        corr(j,i) = 100*((indc(j)-err(j,i))./indc(j));
        if train==1
           errtrain(j,i) = nnz(difftrain(find(Y(:,j)==1),i));
          corrtrain(j,i) = 100*((indc(j)-errtrain(j,i))./indc(j));
        end
    end
end
err_output.errcv=err;
err_output.corrcv=corr;
if train==1
    err_output.errtrain=errtrain;
    err_output.corrtrain=corrtrain;
end
% riscrivo per il modello finale in modo da avere solo l'errore per le LVs scelte
if ismodel(model)
    nlv=size(model.loads{1},2);
    err_output.errtrain=errtrain(:,nlv);
    err_output.corrtrain=corrtrain(:,nlv);    
end

if test==1
    nlv=size(model.loads{1},2);
    for i=nlv:nlv
        pred=plsda(Xtest,Ytest,i,model);
        for k=1:size(Xtest,1)
            [a ipc(k,i)]=max(squeeze(pred.pred{2}(k,:)));
        end
    end
    err_output.predtest=ipc;
    if ~isempty(Ytest)
        for i=1:nc
            ii =find(Ytest(:,i)==1);
            indct(i)=length(ii);
            ict(ii)=i;
        end
        diffp=(ipc-ict'*ones(1,nlv));
        for i=1:nlv
            for j=1:nc
                errt(j,i) = nnz(diffp(find(Ytest(:,j)==1),i));
                corrt(j,i) = 100*((indct(j)-errt(j,i))./indct(j));
            end
        end
        % tengo solo la predizioen per le componenti scelte
        errt=errt(:,nlv);
        corrt=corrt(:,nlv);
        % save
        err_output.errtest=errt;
        err_output.corrtest=corrt;
    end
end

%%% Creo il plot riassuntivo degli errori e giuste assegnazioni
for il=1:nc
    label{il}=strcat('Class', int2str(il));
end
if test==0
figure;
subplot(2,2,1)
plot(corr')
title('Plot of %Correct CV Classifications')
xlabel('Latent Variables');
legend(label)
subplot(2,2,3)
plot(err')
title('Plot of CV Misclassified')
xlabel('Latent Variables');
legend(label)
subplot(2,2,2)
plot(sum(corr)./nc)
title('Sum of Correct CV Classifications, Highest is the best')
xlabel('Latent Variables');
subplot(2,2,4)
plot(sum(err))
title('Sum of Misclassified CV, Lowest is the best')
xlabel('Latent Variables');
hgsave( 'CVerr');
end
%%%%Faccio i calcoli degli errori anche sul training set
%%%% Faccio la figura anche per il training se richiesta

clear a ipc diffp
if test==0 & train==1
    
    figure;
    subplot(2,2,1)
    plot(corrtrain')
    title('Plot of %Correct Classifications training set')
    xlabel('Latent Variables');
    legend(label)
    subplot(2,2,3)
    plot(errtrain')
    title('Plot of Misclassified training set')
    xlabel('Latent Variables');
    legend(label)
    subplot(2,2,2)
    plot(sum(corrtrain)./nc)
    title('Sum of Correct Classifications training set, Highest is the best')
    xlabel('Latent Variables');
    subplot(2,2,4)
    plot(sum(errtrain))
    title('Sum of Misclassified training test, Lowest is the best')
    xlabel('Latent Variables');
    hgsave( 'TRAINerr');
    
end
%close all


