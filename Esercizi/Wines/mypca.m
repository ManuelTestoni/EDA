function model=mypca(Xnam, opt) 
%  Xnam nome del file excel compresa .xls
% opt una struttura con opzioni (vedi sotto)
%% import data matrix 
% expect Xnam is the name of an excel file if you have a different format
% modify the import instructions

% Se Xnam non viene fornito, usa il file di default
if nargin < 1 || isempty(Xnam)
    Xnam = 'wines_condescrvar.xls';
end

dati=importdata(Xnam); % importa il file 
f=fieldnames(dati.data); % nome della variabile
datix=getfield(dati.data,f{1}); % estrae la parte numerica
txt=getfield(dati.textdata,f{1}); % estrae la parte di testo
vlab=txt(1,2:14)';
vlab=deblank(vlab);
category=cell2str(txt(2:end,1));

%
X=datix; % nome della matrice dati
%%
if nargin < 2
    opt.maxPC=rank(X);
    opt.cl=0.95; % confidence level
    opt.t2lim='Fdistrig';
    % opt.t2lim: reference dstribution to calculate T2 Distance limit.
    % Alternatives:
    % 'Fdistrig' use F distribution with A, n-A DoF and scaling factor: A*(n^2 -1)/(n*(n-A))
    % 'perc' uses 95% percentile;
    %  'chi2' uses chisquare distribution
    opt.qlim='jm';
    % opt.qlim: reference dstribution to calculate Orthogonal Distance (Q) limit
    % 'jm'  chisquare jackson mudholkar approx.
    % 'chi2box'  chisquare Box method
    opt.pre='auto';
    %  preprocessing options
    % 'none'  no preprocessing
    % 'mean'  mean centering
    % 'auto'  autoscaling
    % 'pareto'  pareto scaling
    % 'blocksc'  block scaling
    opt.indblk=[];
    % in case you select blocksc then furnish opt and put indblk in opt.indblk
    % indblk is a vector with block label (from 1 to
    % number of block , es: [ 1 1 2 1 1 3 3] in case of 7 variables and 3 blocks,with: 4 var in block
    % 1; 1 in block 2 and 2 in block 3;
end
pstr=struct('type', [], 'settings', []);
pstr.type=opt.pre;
pstr.settings=opt.indblk;


%% 
[Xc, ppars]=mypca_prep(X, 'model', pstr);

[ns,nv]=size(Xc); 
nF=opt.maxPC;
nF=min(nF, rank(Xc)); 

%Computes nF factors PCA model
if ns>nv
    [~,S,P]=svd(Xc'*Xc, 'econ');
    P=P(:,1:nF);
    T=Xc*P;    
    S=diag(S)./(ns-1);
else
    [u,S,P]=svd(Xc,'econ');
    P=P(:,1:nF);
    T=u(:,1:nF)*S(1:nF,1:nF);
    S=diag(S.^2)./(ns-1);
    
end

model.scores=T; 
model.loadings=P; 

lam=S; % All eigenvalues; 
model.eigs=S(1:nF); %Only significant eigenvalues
model.nPC=nF; 
%variance explained by each PC
ev=100*lam/sum(lam);
%cumulative variance explained by the first k Pcs
cv=cumsum(ev);

model.expl_var=ev(1:nF);
model.cum_var=cv(1:nF);
model.pretX.ptype=opt.pre; 
model.pretX.details=pstr; 
model.pretX.preppars=ppars; 
model.detail.eigs=lam; 
model.detail.expl_var=ev; 
model.detail.cum_var=cv; 


%% plotting
% scree plot
figure; plot(model.expl_var,'o-')

%% plot scores and loadings
ihih=0
while ihih==0
    figure;
    xPC=input('which PCs on x ?');
    yPC=input('which PCs on y ?');
    gscatter(T(:,xPC),T(:,yPC),category)
    xlab=strcat('PC',int2str(xPC),' %V (',num2str(model.expl_var(xPC)),')');
    xlabel(xlab)
    ylab=strcat('PC',int2str(yPC),' %V (',num2str(model.expl_var(yPC)),')');
    ylabel(ylab)
    % plot loadings
    figure;
    scatter(P(:,xPC),P(:, yPC),'*')
    hold on; textscatter(P(:,xPC),P(:,yPC),vlab)
    for i=1:size(X,2)
        line([0 P(i,xPC)],[0 P(i,yPC)]);
    end
    xline(0);
    yline(0);
    title('Loadings plot')
    % add x , y labels
    xlab=strcat('PC',int2str(xPC),' %V (',num2str(model.expl_var(xPC)),')');
    xlabel(xlab)
    ylab=strcat('PC',int2str(yPC),' %V (',num2str(model.expl_var(yPC)),')');
    ylabel(ylab)
    tt=input('more plot? (y/n)','s')
    if tt=='n'
        ihih=1;
    end
end
%% plot T2 vs Q
selPC=input('how many PCs you want to select?')

% calculate t2 , q and t2cont, qcont with the selected PCs
model=qtlim(Xc,model,opt,nF,ns,nv,selPC);
t2=model.t2;
q=model.q;
t2lim=model.t2lim;
qlim=model.qlim;
figure;
gscatter(t2,q,category);
xlabel(strcat('T2 with PC= ', int2str(selPC)))
yline(qlim)
xline(t2lim)
% to add samples number
for i=1:ns
    text(t2(i),q(i),int2str(i))
end
%% contribution plot
ncont=input('contribution plot for which samples');
qort=input('T2 or Q contribution ?','s')
figure;
switch qort
    case 'T2'
        t2con=model.t2con;
        bar(t2con(ncont,:)')
        legend([int2str(ncont')])
       a =gca;
       a.XTickLabel=vlab;
        
    case 'Q'
        qcon=model.qcon;
        bar(qcon(ncont,:)')
        legend([int2str(ncont')])
        a =gca;
       a.XTickLabel=vlab;
end

end
%******************
function [model]=qtlim(Xc,model,opt,nF,ns,nv,selPC)
% Calculation of Q contribution and of Q statistics
T=model.scores(:,1:selPC);
P=model.loadings(:,1:selPC);
if ns>nv
    qcon=Xc*(eye(nv)-P*P');
    q=sum(qcon.^2,2);
    qcon=sign(qcon).*qcon.^2;
else
    qcon=Xc-T*P'; 
    q=sum(qcon.^2,2);
    qcon=sign(qcon).*qcon.^2;
    
end

% Calculation of T2 and its contribution
t2con = T/(diag(sqrt(model.eigs(1:selPC))))*P';
t2=sum(t2con.^2,2);
t2con=sign(t2con).*t2con.^2;

model.t2=t2; 
model.t2con=t2con; 
model.q=q; 
model.qcon=qcon; 

% calculate Q lim and T2 lim
t2dof=[];
t2scfact=[];

qdof=[];
qscfact=[];

switch opt.t2lim
    case 'perc'
        [t2sort, ~]=sort(model.t2, 'ascend');
        qa=opt.cl*ns; ka=floor(qa);
        
        if ka~=0
            t2lim=t2sort(ka)+(qa-ka)*(t2sort(ka+1)-t2sort(ka));
        else
            t2lim=(qa-ka)*(t2sort(ka+1));
        end
        
    case 'Fdist'
        
        t2lim=selPC*(ns-1)*finv(opt.cl,selPC,ns-selPC)./(ns-selPC);
        t2dof=[selPC,ns-selPC];
        t2scfact=selPC*(ns-1)/(ns-selPC);
        
        
        
    case 'Fdistrig'
        t2lim=selPC*((ns^2)-1)*finv(opt.cl,selPC,ns-selPC)./(ns*(ns-selPC));
        t2dof=[selPC,ns-selPC];
        t2scfact=selPC*((ns^2)-1)/(ns*(ns-selPC));
        
    case 'chi2'
        t2lim=chi2inv(opt.cl, selPC);
        t2dof=selPC;
        t2scfact=1;

end
switch opt.qlim
    
    case 'perc'
        [qsort, ~]=sort(model.q, 'ascend');
        qa=opt.cl*ns; ka=floor(qa);
        
        if ka~=0
            qlim=qsort(ka)+(qa-ka)*(qsort(ka+1)-qsort(ka));
        else
            qlim=(qa-ka)*(qsort(ka+1));
        end
        
    case 'jm'
        
        theta1 = sum(model.detail.eigs(selPC+1:end));
        theta2 = sum(model.detail.eigs(selPC+1:end).^2);
        theta3 = sum(model.detail.eigs(selPC+1:end).^3);
        if theta1==0
            qlim = 0;
        else
            h0= 1-((2*theta1*theta3)/(3*(theta2.^2)));
            if h0<0.001
                h0 = 0.001;
            end
            ca    = sqrt(2)*erfinv(2*opt.cl-1);
            h1    = ca*sqrt(2*theta2*h0.^2)/theta1;
            h2    = theta2*h0*(h0-1)/(theta1.^2);
            qlim = theta1*(1+h1+h2).^(1/h0);
        end
        
    case 'chi2box'
        
        theta1 = sum(model.detail.eigs(selPC+1:end));
        theta2 = sum(model.detail.eigs(selPC+1:end).^2);
        
        
        g=theta2/theta1;
        Ng=(theta1^2)/theta2;
        qlim=g*chi2inv(opt.cl, Ng);
        qdof=Ng;
        qscfact=g;
end
model.t2lim=t2lim;
model.t2dof=t2dof;
model.t2scfact=t2scfact;

model.qlim=qlim;
model.qdof=qdof;
model.qscfact=qscfact;
end