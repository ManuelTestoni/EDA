function model=simcaf_mainCV2020(X,ncomp,options,icl)
% Main function for the SIMCA routine
% Federico Marini 2008
% 2014 modified Marina Cocchi to calculate sens/spec CV


% Starts crossvalidation
cvtype=options.cv.cvtype;
segments=options.cv.cvsegments;

Nsamples=size(X,1);
if isdataset(X)
    ncl=max(X.class{1});
else
    X=dataset(X);
    ncl=max(icl);
    X.class{1}=icl;
end
    

pred_cl=zeros(Nsamples,1);

switch cvtype
    case {'full', 'syst111', 'syst123', 'random', 'manual'}
        options.plots='off';
        if iscell(segments)
            manualseg=segments; % A cell array
            segments=max(size(segments)); % Now a scalar
        end
        
                
        if strcmp(cvtype,'full')
            cvtype='syst123';
            segments=Nsamples;
        end
        
        if strcmp(cvtype,'random')
            ix=randperm(Nsamples);
        end
        
        no_sampl=fix(Nsamples/segments);
        left_over_samples=mod(Nsamples,segments);
     
        count=1;
        
        for nlv=1:ncomp
            ssmatrix=zeros(ncl,ncl);
            for ii=1:segments
                options.plots='off';
                %s = sprintf('Cross validation segment number %g',ii); disp(s)
                if strcmp(cvtype,'syst111')
                    if left_over_samples==0
                        p_cvs=((ii-1)*no_sampl+1+(count-1):ii*no_sampl+(count-1))';
                    else
                        p_cvs=((ii-1)*no_sampl+1+(count-1):ii*no_sampl+count)';
                        count=count+1;
                        left_over_samples=left_over_samples-1;
                    end
                elseif strcmp(cvtype,'syst123')
                    p_cvs=(ii:segments:Nsamples)';
                elseif strcmp(cvtype,'random')
                    p_cvs=(ii:segments:Nsamples)';
                    p_cvs=ix(p_cvs)';
                elseif strcmp(cvtype,'manual')
                    if max(size(manualseg))~=segments
                        disp('The number of segments does not correspond to the segments in manualseg')
                        break
                    end
                    nn=0;
                    for jj=1:segments
                        nn=nn + max(size(manualseg{jj}));
                    end
                    if Nsamples~=nn
                        disp('The number of samples in X does not correspond to the number of samples in manualseg')
                        break
                    end
                    p_cvs=manualseg{ii};
                end
                tot=(1:Nsamples)';
                tot(p_cvs)=[];
                m_cvs = tot;
                model.CrossVal.Segments{ii}=p_cvs;
                Xseg=X(m_cvs,:);
                Xtseg=X(p_cvs,:);
                try
                    rr=rank(Xseg);
                catch
                    rr=rank(Xseg.data);
                end
                if nlv>rr
                    nlv=rr;
                    ncomp=nlv;
                end
                modcv=simcaf_mod2020(Xseg,nlv,options);
                predcv=simcaf_pred2020(Xtseg,modcv);
                
                model.CrossVal.CVModels{nlv,ii}=modcv;
                model.CrossVal.CVPreds{nlv,ii}=predcv;
                ssmatrix=ssmatrix+predcv.SIMCA.sensspec;
                pred_cl(p_cvs)=predcv.SIMCA.predclass;
            end
            options.plots='off';
            full_mod=simcaf_mod2020(X,nlv,options);
            model.Model{nlv}=full_mod;
            model.CrossVal.sensspec{nlv}=ssmatrix;
            model.CrossVal.predclass{nlv}=pred_cl;
        end
        figure;
        sesp=cat(3,model.CrossVal.sensspec{:});
        for ic=1:ncl
            nc=nnz(find(X.class{1}==ic));
            sens(ic,:)=squeeze(sesp(ic,ic,:))./nc;
            spec(ic,:)=squeeze(sum(sesp(ic,setdiff(1:ncl,ic),:),2))./(Nsamples-nc);
        end
        subplot(3,1,1);plot([1:ncomp],sens',':o');
        title('Sensitivity');
        subplot(3,1,2);plot([1:ncomp],spec',':o');
        title('Specificiity');
        eff=sqrt(sens.*spec);
        subplot(3,1,3);plot([1:ncomp],eff',':o');
        title('Efficiency');
        saveas(gcf,'SensSpecEffCV','fig');
        nlvf=input('Choose the number of latent variables for each class, esempio per 3 classi: [1 4 5]');
        options.plots='on';
        fin_mod=simcaf_mod2020(X,nlvf,options);
        model.finalmodel=fin_mod;



    case {'none'} % No validation
        net.CrossVal=[];
    otherwise
        disp('Not accepted validation method')
end



