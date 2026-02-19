function moddis=cvPLSDA(Xtrain, classind, Xtest, classtest)
%% to run PLS DA outside PLS toolbox
%% syntax
% moddis=cvPLSDA(Xtrain, classind,Xtest,Ytest)

%% INPUT
% Xtrain: a matrix with your data (double)
% classind: a vector with class index (as many rows as samples a number for
% each class)
% Xtest: a matrix with test set if you have it
% classtest: a vector with class index for test set 
%
% optional input: Xtest, classtest
% if you have already dataset, eg called X, you can obtain Xtrain and
% classind from the following:
% Xtrain=X.data;
% classind=X.class{1};


evrimovepath('top');
%% 1st step prepare a file with class information coded 1 0 for each class
% from classind which contain the numerical class index
[n m]=size(Xtrain);
nc=unique(classind);
Ytrain=zeros(n,max(nc));
for ic=1:max(nc)
    Ytrain(classind==ic,ic)=1;
end
if nargin > 3
    [nt, ~]=size(Xtest);
    nct=unique(classtest);
    Ytest=zeros(nt,max(nct));
    for ict=1:max(nct)
       Ytest(classtest==ict,ict)=1;
    end
else
    Ytest=[];
end
% prepare option file for crossval
opt=crossval('options');
% set the preprocessing
disp('choose preprocessing for X-block')
px=preprocess;
py=[];
opt.preprocessing={px py};
opt.display='off';
maxlv=min(size(Xtrain));
resultscv = crossval(Xtrain,Ytrain,'plsda',{'vet',6,1},maxlv, opt);  % here 6 split were chosen and up to  maxLV were computed

% run errcv to obtain plot of classification error and missclassified in CV

err_output= errcv(resultscv,Ytrain,1,0);
moddis.errcvloop=err_output;
% you'll get figure with error in cv as function of number of LV  (also you
% get same figure for training set)
% so you can see were you have minima (if there is no much difference in missclassified 
% choose a lower n° of LV also if it is not an absolute minimum) 

%%  calculate the PLSDA model with the nLV you choose 
i=input('how many LVs?');
nLV=i;
close all
% fix preprocessing to be the same you used in cv
optPLS=plsda('options');
optPLS.preprocessing={px py};
optPLS.display='off';
optPLS.plots='none';
% calculate PLSDA model
modPLSDA=plsda(Xtrain,Ytrain,nLV,optPLS);
% run errcv to calculate errtrain and corrtrain for final model
err_final= errcv(resultscv,Ytrain,modPLSDA,0);
moddis.modelPLSDA=modPLSDA;
moddis.errfinal=err_final;

% if a test set is given in input also test will be predicted at this step
if nargin>2
    if ~isempty(Ytest)
        [err_final, pred]= errcv(resultscv,Ytrain,modPLSDA,1, Xtest, Ytest);
        moddis.errfinal=err_final;
        moddis.predPLSDA=pred;
    else
        [err_final, pred]= errcv(resultscv,Ytrain,modPLSDA,1, Xtest, []);
        moddis.errfinal=err_final;
        moddis.predPLSDA=pred;
    end
end
%% plots
figure;
for i=1:max(nc)
    if ~isempty(Ytest)
        subplot(2,1,1)
        hold on;
        h=plot([1:n],modPLSDA.pred{1,2}(:,i),'o-');
        numc=find(Ytrain(:,i)==1);
        a=h.Color;       
        hold on;plot(numc,modPLSDA.pred{1,2}(numc,i),'o','MarkerFaceColor',a,'MarkerEdgeColor',a)
        title('Training set: filled symbols belong to the class');
        xlabel('samples n°');
        ylabel('Estimated Y');
        subplot(2,1,2)
        hold on;
        h=plot([1:nt],pred.pred{1,2}(:,i),'o-');
        numct=find(Ytest(:,i)==1);
        a=h.Color;
        hold on;plot(numct,pred.pred{1,2}(numct,i),'o','MarkerFaceColor',a,'MarkerEdgeColor',a);
        title('Test set: filled symbols belong to the class');
        xlabel('samples n°');
        ylabel('Predicted Y')
    else
        hold on;
        h=plot([1:n],modPLSDA.pred{1,2}(:,i),'o-');
        numc=find(Ytrain(:,i)==1);
        a=h.Color;       
        hold on;plot(numc,modPLSDA.pred{1,2}(numc,i),'o','MarkerFaceColor',a,'MarkerEdgeColor',a)
        title('Training set: filled symbols belong to the class');
        xlabel('samples n°');
        ylabel('Estimated Y');
    end
end




