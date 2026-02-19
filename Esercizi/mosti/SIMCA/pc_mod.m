function model=pc_mod(Xd1,Xd2, ncomp,options)
% Main function for pca routine
% 


[nsamp,nvar]=size(Xd2);
sctype=options.preproc;




%  preprocessing
if strcmp(sctype,'scale')
    stdx=missstd(Xd1);
    model.prepr={[] stdx};
    Xd1=Xd1./(ones(size(Xd1,1),1)*stdx);
    Xd2=Xd2./(ones(nsamp,1)*stdx);
elseif strcmp(sctype,'auto')
    stdx=missstd(Xd1);
    mx=missmean(Xd1);
    model.prepr={mx stdx};
    Xd1=(Xd1-ones(size(Xd1,1),1)*mx)./(ones(size(Xd1,1),1)*stdx);
    Xd2=(Xd2-ones(nsamp,1)*mx)./(ones(nsamp,1)*stdx);
elseif strcmp(sctype,'autop')
    stdx=missstd(Xd1);
    mx=missmean(Xd1);
    model.prepr={mx stdx};
    Xd1=(Xd1-ones(size(Xd1,1),1)*mx)./(ones(size(Xd1,1),1)*sqrt(stdx));
    Xd2=(Xd2-ones(nsamp,1)*mx)./(ones(nsamp,1)*sqrt(stdx));
elseif strcmp(sctype,'scalep')
    stdx=missstd(Xd1);
    model.prepr={[] stdx};
    Xd1=Xd1./(ones(size(Xd1,1),1)*sqrt(stdx));
    Xd2=Xd2./(ones(nsamp,1)*sqrt(stdx));
elseif strcmp(sctype,'mean')
    mx=missmean(Xd1);
    model.prepr={mx []};
    Xd1=Xd1-ones(size(Xd1,1),1)*mx;
    Xd2=Xd2-ones(nsamp,1)*mx;
elseif strcmp(sctype,'none')
    %Just to complete the options
    model.prepr={[] []};
end
   
if nsamp>=nvar
    [u,s,v]=misssvd(Xd1,'econ');
    T1=u(:,1:ncomp)*s(1:ncomp,1:ncomp);
    P=v(:,1:ncomp);
else
    [u,s,v]=misssvd(Xd1','econ');
    T1=v(:,1:ncomp)*s(1:ncomp,1:ncomp);
    P=u(:,1:ncomp);
end
    
lambda=(s.^2)./(size(Xd1,1)-1);
tot_var=trace(lambda);
ev=(100./tot_var)*diag(lambda);
cv=zeros(length(ev),1);
for ll=1:length(ev)
    cv(ll)=sum(ev(1:ll));
end
eigs=diag(lambda);

% computation of confidence limits
t2lim = ncomp*(size(Xd1,1)-1)/(size(Xd1,1)-ncomp)*ftest(1-options.cl,ncomp,size(Xd1,1)-ncomp);

% computation of q2 limit J-M method
theta1 = sum(eigs(ncomp+1:length(eigs)));
theta2 = sum(eigs(ncomp+1:length(eigs)).^2);
theta3 = sum(eigs(ncomp+1:length(eigs)).^3);
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

%computation of t square and q
% cal
tsq1=diag(T1*inv(lambda(1:ncomp,1:ncomp))*T1');
q1=misssum((Xd1-T1*P').^2,2);
%test
T2=missmult(Xd2,P);
tsq2=diag(T2*inv(lambda(1:ncomp,1:ncomp))*T2');
q2=misssum((Xd2-T2*P').^2,2);
    
model.tr.res=(Xd1-T1*P');
model.tr.scores=T1;
model.tr.loads=P;
model.tr.eigs=eigs;
model.tr.variance={ev,cv};
model.tr.critvals=[t2lim, qlim];
model.tr.tsq=tsq1;
model.tr.qres=q1;

model.ts.res=(Xd2-T2*P');
model.ts.scores=T2;
model.ts.tsq=tsq2;
model.ts.qres=q2;






            