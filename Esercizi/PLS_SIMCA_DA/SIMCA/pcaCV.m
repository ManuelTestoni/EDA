function model=pcaCV(X,ncomp,options)
%  function for pcaCV


% Starts crossvalidation
cvtype=options.cv.cvtype;
segments=options.cv.cvsegments;
Nsamples=size(X,1);

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
        
        
        %ssmatrix=zeros(ncl,ncl);
        scoresCV=[];
            qresCV=[];
            tsqCV=[];
            contCV=[];
        for ii=1:segments
            options.plots='off';
            s = sprintf('Cross validation segment number %g',ii); disp(s)
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
            
            modcv=pc_mod(Xseg,Xtseg,ncomp,options);
            scoresCV=[scoresCV; modcv.ts.scores];
            qresCV=[qresCV; modcv.ts.qres'];
            tsqCV=[tsqCV; modcv.ts.tsq];
            
            cont = [];
            for i=1:size(Xtseg,1)
                for j=1:size(Xtseg,2)
                    cont(i,j) = modcv.ts.scores(i,:)*diag(modcv.tr.eigs(1:ncomp,:).^(-1/2))*Xtseg(i,j)*modcv.tr.loads(j,:)';
                end
            end
            contCV=[contCV;cont];
        end
        model.CV.scoresCV=scoresCV;
        model.CV.qresCV=qresCV;
        model.CV.tsqCV=tsqCV;
        model.CV.contCV=contCV;
end



