function model=simcaf_mod2020(X,ncomp,options)
% Main function for the SIMCA routine
% Federico Marini 2008

if ~isdataset(X) || isempty(X.class{1})
    error('Input should be in the form of a dataset with a valid class field')
end
if length(ncomp)==1
    ncomp=ncomp*ones(max(X.class{1}),1);
elseif length(ncomp)~=max(X.class{1})
        error('The dimension of ncomp doesn''t match with the number of classes')
end

cll=X.class{1};
nclass=max(cll);
Xd=X.data;
[nsamp,nvar]=size(Xd);


markers1=['b';'g';'r';'c';'m';'y';'k'];
markers2=['o';'x';'+';'*';'s';'d';'v';'^';'<';'>';'p';'h';'.'];

mark=char(zeros(ceil(nclass/length(markers1)),2));
  
for i=1:ceil(nclass/length(markers1))
    for j=1:length(markers1)
        mark((i-1)*length(markers2)+j,:)=[markers2(i),markers1(j)];
    end
end
model.marks=mark;


sensspec=zeros(nclass,nclass);

sctype=options.preproc;

for i=1:nclass
    cl_ind=find(cll==i);
    Xd1=Xd(cl_ind,:);
    Xd2=Xd;
    
    % class-based preprocessing
    if strcmp(sctype,'scale')
        stdx=missstd(Xd1);     
        model.PCmodels{i}.prepr={[] stdx};
        Xd1=Xd1./(ones(size(Xd1,1),1)*stdx);
        Xd2=Xd2./(ones(nsamp,1)*stdx);
    elseif strcmp(sctype,'auto')
        stdx=missstd(Xd1);     
        mx=missmean(Xd1);
        model.PCmodels{i}.prepr={mx stdx};
        Xd1=(Xd1-ones(size(Xd1,1),1)*mx)./(ones(size(Xd1,1),1)*stdx);
        Xd2=(Xd2-ones(nsamp,1)*mx)./(ones(nsamp,1)*stdx);
    elseif strcmp(sctype,'autop')
        stdx=missstd(Xd1);     
        mx=missmean(Xd1);
        model.PCmodels{i}.prepr={mx stdx};
        Xd1=(Xd1-ones(size(Xd1,1),1)*mx)./(ones(size(Xd1,1),1)*sqrt(stdx));
        Xd2=(Xd2-ones(nsamp,1)*mx)./(ones(nsamp,1)*sqrt(stdx));    
    elseif strcmp(sctype,'scalep')
        stdx=missstd(Xd1);     
        model.PCmodels{i}.prepr={[] stdx};
        Xd1=Xd1./(ones(size(Xd1,1),1)*sqrt(stdx));
        Xd2=Xd2./(ones(nsamp,1)*sqrt(stdx));
    elseif strcmp(sctype,'mean')
        mx=missmean(Xd1);
        model.PCmodels{i}.prepr={mx []};
        Xd1=Xd1-ones(size(Xd1,1),1)*mx;
        Xd2=Xd2-ones(nsamp,1)*mx;
    elseif strcmp(sctype,'none')
        %Just to complete the options
        model.PCmodels{i}.prepr={[] []};
    end
   
    if nsamp>=nvar
        
        [u,s,v]=misssvd(Xd1,'econ');
        ncomp(i)=min(size(u,1),ncomp(i));
        T=u(:,1:ncomp(i))*s(1:ncomp(i),1:ncomp(i));
        P=v(:,1:ncomp(i));
    else
        [u,s,v]=misssvd(Xd1','econ');
        ncomp(i)=min(size(u,2),ncomp(i));
        T=v(:,1:ncomp(i))*s(1:ncomp(i),1:ncomp(i));
        P=u(:,1:ncomp(i));
    end
    
    lambda=(s.^2)./(size(Xd1,1)-1);
    if size(lambda) ==1
        tot_var=lambda(1,1);
        ev=(100./tot_var).*lambda;
    else
    tot_var=diag(lambda);   
    ev=(100./tot_var).*diag(lambda);
    end
    cv=zeros(length(ev),1);
    for ll=1:length(ev)
        cv(ll)=sum(ev(1:ll));
    end
    eigs=diag(lambda);
    
    %computation of t square and q on the dataset based on the class model
    T1=missmult(Xd2,P);
    tsq=diag(T1*inv(lambda(1:ncomp(i),1:ncomp(i)))*T1');
    q=misssum((Xd2-T1*P').^2,2);

    % computation of confidence limits
    
    switch options.lim
        case 'jm'
            t2lim = ncomp(i)*(size(Xd1,1)-1)/(size(Xd1,1)-ncomp(i))*ftest(1-options.cl,ncomp(i),size(Xd1,1)-ncomp(i));
            
            % computation of q2 limit J-M method
            theta1 = sum(eigs(ncomp(i)+1:length(eigs)));
            theta2 = sum(eigs(ncomp(i)+1:length(eigs)).^2);
            theta3 = sum(eigs(ncomp(i)+1:length(eigs)).^3);
            if theta1==0;
                qlim = 0;
            else
                h0     = 1-2*theta1*theta3/3/(theta2.^2);
                if h0<0.001
                    h0 = 0.001;
                end
                ca    = sqrt(2)*erfinv(2*options.cl-1);
                h1    = ca*sqrt(2*theta2*h0.^2)/theta1;
                h2    = theta2*h0*(h0-1)/(theta1.^2);
                qlim = theta1*(1+h1+h2).^(1/h0);
            end
            
        case 'prct'
            % computation of limits based on percentile (added July 2020 MC)
            t2lim=prctlim(tsq(cl_ind),options.cl);
            qlim=prctlim(q(cl_ind)',options.cl);
            %
    end
    cl_acc=zeros(nsamp,1);
    cr_i=zeros(nsamp,1);
    switch options.crit
        case 'combined'
            for j=1:nsamp
                cr_i(j)=sqrt((tsq(j)./t2lim).^2+(q(j)./qlim).^2);
                if cr_i(j)<=sqrt(2)
                    cl_acc(j)=1;
                    if cll(j)==i
                        sensspec(i,cll(j))=sensspec(i,cll(j))+1;
                    end
                else
                    if cll(j)~=i
                        sensspec(i,cll(j))=sensspec(i,cll(j))+1;
                    end
                end
            end
            if strcmp(options.plots,'on')
                figure
                hold on
                for k=1:nclass
                    plot(tsq(find(cll==k))./t2lim,q(find(cll==k))./qlim,mark(k,:))
                end
                a=0:0.0001:sqrt(2);
                plot(a,sqrt(2-a.^2),'-r');
                axis ([0 5 0 5]); 
                xlabel('score distance'); ylabel('orthogonal distance');
                title(['simca model for the category: ', num2str(i), 'nLV = ', num2str(ncomp(i))]);hold off
                name=strcat('T2rQrplot_C', num2str(i));
                hgsave(name);
                close
            end
              
            
        case 'square'
            for j=1:nsamp
                cr_i(j)=sqrt((tsq(j)./t2lim)*(q(j)./qlim));
                if (tsq(j)<t2lim)&& (q(j)<qlim)
                    cl_acc(j)=1;
                    if cll(j)==i
                        sensspec(i,cll(j))=sensspec(i,cll(j))+1;
                    end
                else
                    if cll(j)~=i
                        sensspec(i,cll(j))=sensspec(i,cll(j))+1;
                    end
                end
            end

            if strcmp(options.plots,'on')
                figure
                hold on
                
                for k=1:nclass
                    plot((tsq(find(cll==k))./t2lim),(q(find(cll==k))./qlim),mark(k,:))
                end
                axis([0 5 0 5]);
                hline(1,'-r');
                vline(1,'-r');
                xlabel('leverage'); ylabel('orthogonal distance');title(['simca model for the category: ',num2str(i)]);hold off
                name=strcat('T2rQrplot_C', num2str(i));
                hgsave(name);
                close
            end
    end
    model.PCmodels{i}.res=(Xd2-T1*P');
    model.PCmodels{i}.scores=T1;
    model.PCmodels{i}.loads=P;
    model.PCmodels{i}.eigs=eigs;
    model.PCmodels{i}.variance={ev,cv};
    model.PCmodels{i}.critvals=[t2lim, qlim];
    model.PCmodels{i}.tsq=tsq;
    model.PCmodels{i}.qres=q;
    model.SIMCA.accept{i}=cl_acc;
    model.SIMCA.crit(:,i)=cr_i;
end

model.SIMCA.sensspec=sensspec;
model.SIMCA.opts=options;
model.SIMCA.ncomp=ncomp;
model.SIMCA.nclass=nclass;

for j=1:nsamp
    [critmin,final_cl]=min(model.SIMCA.crit,[],2);
end
model.SIMCA.predclass=final_cl;
model.SIMCA.critmin=critmin;

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



            