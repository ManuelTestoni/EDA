function [Xp, model]= mypca_prep(X, pstrat, prstr)
%  Function for preprocessing data
%  The function mypca_prep  preprocess data (column wise), 
%  according to the pretreatments reported in the preprocessing structure prstr:
%  Preprocessing parameters can be based on the actual data:
%  pstrat='model' 
% or on a previous training dataset %
% pstrat='apply'
% or can undo preprocessing based on a previous training
% pstrat='undo'

% Check if prstr is provided, if not set it to empty
if nargin < 3
    prstr = struct('type', 'none', 'settings', []);
end

switch pstrat
    case 'model'
        [Xp,model]= prepmod(X,prstr);
        
    case 'apply'
        
        [Xp,model]= prepapply(X,prstr);
        
    case 'undo'
        
        Xp= prepundo(X,prstr);
        model=prstr;
end

%
    function [Xp,model]=prepmod(X,prstr)

    model=prstr;
    Xp=X;
    [ns,nv]=size(Xp);

    switch prstr.type
        case 'none'
            Xp=Xp;
            model.parameters=[];

        case 'mean'
            mx=mean(Xp);
            model.parameters=mx;
            Xp=Xp-repmat(mx,ns,1);

        case 'auto'
            mx=mean(Xp);
            sx=std(Xp);
            if isempty(prstr.settings) || ~strcmp(prstr.settings{1}, 'eps')
                Xp=Xp-repmat(mx,ns,1);
                Xp=Xp(:,sx~=0)./repmat(sx(sx~=0),ns,1);
            elseif ischar(prstr.settings{1}) && strcmp(prstr.settings{1}, 'eps')
                sx(sx==0)=eps;
                Xp=(Xp-repmat(mx,ns,1))./repmat(sx,ns,1);
            elseif ~ischar(prstr.settings{1})
                sx(sx==0)=prstr.settings{1};
                Xp=(Xp-repmat(mx,ns,1))./repmat(sx,ns,1);
            end

            model.parameters={mx sx};

        case 'pareto'

            mx=mean(Xp);
            sx=sqrt(std(Xp));

            if isempty(prstr.settings) || ~strcmp(prstr.settings{1}, 'eps')
                Xp=Xp-repmat(mx,ns,1);
                Xp=Xp(:,sx~=0)./repmat(sx(sx~=0),ns,1);
            elseif ischar(prstr.settings{1}) && strcmp(prstr.settings{1}, 'eps')
                sx(sx==0)=eps;
                Xp=(Xp-repmat(mx,ns,1))./repmat(sx,ns,1);
            elseif ~ischar(prstr.settings{1})
                sx(sx==0)=prstr.settings{1};
                Xp=(Xp-repmat(mx,ns,1))./repmat(sx,ns,1);
            end

            model.parameters={mx sx};

        case 'scalenc'

            sx=std(Xp);
            if isempty(prstr.settings) || ~strcmp(prstr.settings, 'eps')
                Xp=Xp(:,sx~=0)./repmat(sx(sx~=0),ns,1);
                Xp(:,sx==0)=0;
            elseif ischar(prstr.settings) && strcmp(prstr.settings, 'eps')
                sx(sx==0)=eps;
                Xp=Xp./repmat(sx,ns,1);
            elseif ~ischar(prstr.settings)
                sx(sx==0)=prstr.settings;
                Xp=Xp./repmat(sx,ns,1);
            end

            model.parameters=sx;

        case 'paretonc'

            sx=sqrt(std(Xp));
            if isempty(prstr.settings) || ~strcmp(prstr.settings, 'eps')
                Xp=Xp(:,sx~=0)./repmat(sx(sx~=0),ns,1);
                Xp(:,sx==0)=0;
            elseif ischar(prstr.settings{1}) && strcmp(prstr.settings, 'eps')
                sx(sx==0)=eps;
                Xp=Xp./repmat(sx,ns,1);
            elseif ~ischar(prstr.settings)
                sx(sx==0)=prstr.settings;
                Xp=Xp./repmat(sx,ns,1);
            end

            model.parameters=sx;


        case 'blocksc'
            ind_blk=prstr.settings;
            nblock=length(unique(ind_blk));
            mx=mean(Xp);
            Xp=Xp-repmat(mx,ns,1);
            SStot=misssumsq(Xp);
            for iblk=1:nblock
                indb=find(ind_blk==iblk);
                SSblock(iblk)=misssumsq(Xp(:,indb));
                sx(1,indb)=1./(sqrt(SStot./(SSblock(iblk).*nblock)));
            end
            Xp=Xp./repmat(sx,ns,1);
            model.parameters={mx sx};
    end

%
        function [SS]=misssumsq(X)

            %Calulates the sum-of-squares of the matrix X.
            %X may hold missing elements (NaN's)
            X=reshape(X,size(X,1),prod(size(X))/size(X,1));
            [inan jnan]=find(isnan(X));
            innan=size(inan,1);
            for i=1:innan,
                X(inan(i),jnan(i))=0;
            end;
            SS=sum(sum( (X).^2 ));
        end
    %

function [Xp,model]=prepapply(X,prstr)

model=prstr;
Xp=X;
[ns,nv]=size(Xp); 


    switch prstr.type
        case 'none'
            Xp=Xp; 
            
        case 'mean'
            mx=prstr.parameters; 
            Xp=Xp-repmat(mx,ns,1);
            
        case 'auto'
            mx=model.parameters{1}; 
            sx=model.parameters{2}; 
            if isempty(prstr.settings)
                Xp=Xp-repmat(mx,ns,1);
                Xp=Xp(:,sx~=0)./repmat(sx(sx~=0),ns,1);
            else
                Xp=(Xp-repmat(mx,ns,1))./repmat(sx,ns,1);
            end
                      
        case 'blocksc'
            mx=model.parameters{1};
            sx=model.parameters{2};
            Xp=Xp-repmat(mx,ns,1);
            Xp=Xp(:,sx~=0)./repmat(sx(sx~=0),ns,1);

        case 'pareto'
            
            mx=model.parameters{1}; 
            sx=model.parameters{2}; 
            if isempty(prstr.settings)
                Xp=Xp-repmat(mx,ns,1);
                Xp=Xp(:,sx~=0)./repmat(sx(sx~=0),ns,1);
            else
                Xp=(Xp-repmat(mx,ns,1))./repmat(sx,ns,1);
            end
            
        case 'scalenc'
            
            sx=model.parameters; 
            if isempty(prstr.settings)
                Xp=Xp(:,sx~=0)./repmat(sx(sx~=0),ns,1);
                Xp(:,sx==0)=0;
            else
                Xp=Xp./repmat(sx,ns,1);
            end
            
            case 'paretonc'
            
            sx=model.parameters; 
            if isempty(prstr.settings)
                Xp=Xp(:,sx~=0)./repmat(sx(sx~=0),ns,1);
                Xp(:,sx==0)=0;
            else
                Xp=Xp./repmat(sx,ns,1);
            end            
    end
    
end

function Xp=Rcprepundo(X,prstr)
nprep=length(prstr);
Xp=X;
[ns,nv]=size(Xp); 


    switch prstr.type
        case 'none'
            Xp=Xp; 
            
        case 'mean'
            mx=prstr.parameters; 
            Xp=Xp+repmat(mx,ns,1);
            
        case 'auto'
            mx=prstr.parameters{1}; 
            sx=prstr.parameters{2}; 
            Xp=(Xp.*repmat(sx,ns,1))+repmat(mx,ns,1);
         
        case 'blocksc'
            mx=prstr.parameters{1}; 
            sx=prstr.parameters{2}; 
            Xp=(Xp.*repmat(sx,ns,1))+repmat(mx,ns,1);

        case 'pareto'
            
            mx=prstr.parameters{1}; 
            sx=prstr.parameters{2}; 
            Xp=(Xp.*repmat(sx,ns,1))+repmat(mx,ns,1);
            
        case 'scalenc'
            
            sx=prstr.parameters; 
            Xp=Xp.*repmat(sx,ns,1);
            
           
            case 'paretonc'
            
            sx=prstr.parameters; 
            Xp=Xp.*repmat(sx,ns,1);
            
            
        
    end
    
end
    end
end
