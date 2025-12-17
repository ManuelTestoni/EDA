function pred=simcaf_pred2020(Xpred,simcamod)
% Predicts the class belonging of unknown samples based on a computed
% SIMCA model

ncl=simcamod.SIMCA.nclass;
sctype=simcamod.SIMCA.opts.preproc;
ncomp=simcamod.SIMCA.ncomp;
cl_pr=0;
nclp=0;
if isdataset(Xpred)
    Xp=Xpred.data;
    if ~isempty(Xpred.class{1})
        clp=Xpred.class{1};
        nclp=nnz(find(unique(clp)));
        sensspec=zeros(ncl,ncl);
        cl_pr=1;
        markers1=['b';'g';'r';'c';'m';'y';'k'];
        markers2=['o';'x';'+';'*';'s';'d';'v';'^';'<';'>';'p';'h';'.';];

        mark=char(zeros(ceil(ncl/length(markers1)),2));
  
        for i=1:ceil(ncl/length(markers1))
            for j=1:length(markers1)
                mark((i-1)*length(markers2)+j,:)=[markers2(i),markers1(j)];
            end
        end
        model.marks=mark;

    end
else
    Xp=Xpred;
end

nspred=size(Xp,1);

for i=1:ncl
    P=simcamod.PCmodels{i}.loads;
    lambda=diag(simcamod.PCmodels{i}.eigs);
    % class-based preprocessing
    if strcmp(sctype,'scale')
        stdx=simcamod.PCmodels{i}.prepr{2};
        Xp1=Xp./(ones(nspred,1)*stdx);
    elseif strcmp(sctype,'auto')
        stdx=simcamod.PCmodels{i}.prepr{2};
        mx=simcamod.PCmodels{i}.prepr{1};
        Xp1=(Xp-ones(nspred,1)*mx)./(ones(nspred,1)*stdx);
    elseif strcmp(sctype,'autop')
        stdx=simcamod.PCmodels{i}.prepr{2};
        mx=simcamod.PCmodels{i}.prepr{1};
        Xp1=(Xp-ones(nspred,1)*mx)./(ones(nspred,1)*sqrt(stdx));
    elseif strcmp(sctype,'mean')
        mx=simcamod.PCmodels{i}.prepr{1};
        Xp1=Xp-ones(nspred,1)*mx;
    elseif strcmp(sctype,'scalep')
        stdx=simcamod.PCmodels{i}.prepr{2};
        Xp1=Xp./(ones(nspred,1)*sqrt(stdx));
    elseif strcmp(sctype,'none')
        Xp1=Xp;
    end
    
    Tp=missmult(Xp1,P);
    tsq=diag(Tp*inv(lambda(1:ncomp(i),1:ncomp(i)))*Tp');
    q=misssum((Xp1-Tp*P').^2,2);
    lim=simcamod.SIMCA.opts.lim;
    switch lim
        case 'jm'
            t2lim=simcamod.PCmodels{i}.critvals(1);
            qlim=simcamod.PCmodels{i}.critvals(2);
        case 'prct'
            t2lim=prctlim(tsq,simcamod.SIMCA.opts.cl);
            qlim=prctlim(q',simcamod.SIMCA.opts.cl);
    end
    cl_acc=zeros(nspred,1);
    cr_i=zeros(nspred,1);
    switch simcamod.SIMCA.opts.crit
        case 'combined'
            for j=1:nspred
                cr_i(j)=sqrt((tsq(j)./t2lim).^2+(q(j)./qlim).^2);
                if cr_i(j)<=sqrt(2)
                    cl_acc(j)=1;
                    if cl_pr==1 && clp(j)==i
                        sensspec(i,clp(j))=sensspec(i,clp(j))+1;
                    end
                else
                    if cl_pr==1 && clp(j)~=i
                        if ncl>1
                            sensspec(i,clp(j))=sensspec(i,clp(j))+1;
                        end
                    end
                end
            end
                if cl_pr==1 && strcmp(simcamod.SIMCA.opts.plots,'on')
                    figure
                    hold on
                    for k=1:max(ncl,nclp)
                        plot(tsq(find(clp==k))./t2lim,q(find(clp==k))./qlim,mark(k,:))
                    end
                    a=0:0.0001:sqrt(2);
                    plot(a,sqrt(2-a.^2),'-r');
                    axis ([0 5 0 5]);
                    xlabel('leverage'); ylabel('orthogonal distance');
                    title(['simca prediction for the category: ',num2str(i)]);hold off
                    name=strcat('T2rQrplot_C', num2str(i),'_test');
                    hgsave(name);
                    close
                end
            
            
        case 'square'
            for j=1:nspred
                cr_i(j)=sqrt((tsq(j)./t2lim)*(q(j)./qlim));
                if (tsq(j)<t2lim)&& (q(j)<qlim)
                    cl_acc(j)=1;
                    if cl_pr==1 && clp(j)==i
                        sensspec(i,clp(j))=sensspec(i,clp(j))+1;
                    end
                else
                    if cl_pr==1 && clp(j)~=i
                        sensspec(i,clp(j))=sensspec(i,clp(j))+1;
                    end
                end
            end

            if cl_pr==1 && strcmp(simcamod.SIMCA.opts.plots,'on')
                figure
                hold on
                
                for k=1:ncl
                    plot((tsq(find(clp==k))./t2lim),(q(find(clp==k))./qlim),mark(k,:))
                end
                axis([0 5 0 5]);
                hline(1,'-r');
                vline(1,'-r');
                xlabel('leverage'); ylabel('orthogonal distance');title(['simca model for the category: ',num2str(i)]);hold off
            name=strcat('T2rQrplot_C', num2str(i),'_test');
                    hgsave(name);
                    close
            end
    end
    
    pred.PCmodels{i}.scores=Tp;
    pred.PCmodels{i}.loads=P;
    pred.PCmodels{i}.critvals=[t2lim, qlim];
    pred.PCmodels{i}.tsq=tsq;
    pred.PCmodels{i}.qres=q;
    pred.SIMCA.accept{i}=cl_acc;
    pred.SIMCA.crit(:,i)=cr_i;
end

if cl_pr==1
    pred.SIMCA.sensspec=sensspec;
end

pred.SIMCA.ncomp=ncomp;
pred.SIMCA.nclass=ncl;

for j=1:nspred
    [critmin,final_cl]=min(pred.SIMCA.crit,[],2);
end
pred.SIMCA.predclass=final_cl;
pred.SIMCA.critmin=critmin;

function rlimit = prctlim(x,alpha)
% calculate the 95 percentile of X
% outlying points, i.e. deviating for more than 1.5 times the interquartile range
% are excluded from limit computation
%
%% x is assume dto be made of only positive values

% rlimit=rlim(x,alpha)
%%% to fix bug disocvered May 2010
%% n=size(x,1); has been substituted by
n=length(x);
ind=ones(n,1);
p=100*alpha;
iqrange=iqr(x);
outlier_ind=find(x > iqrange*1.5+prctile(x,75));
ind(outlier_ind)=0;
xok_ind=find(ind);
rlimit=prctile(x(xok_ind),p);

    