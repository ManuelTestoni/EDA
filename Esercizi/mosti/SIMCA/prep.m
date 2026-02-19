function [Xd1, pret]=prep(Xd1, sctype)
mx=[];
stdx=[];
if strcmp(sctype,'scale')
        stdx=missstd(Xd1);        
        Xd1=Xd1./(ones(size(Xd1,1),1)*stdx);
    elseif strcmp(sctype,'auto')
        stdx=missstd(Xd1);     
        mx=missmean(Xd1);
        Xd1=(Xd1-ones(size(Xd1,1),1)*mx)./(ones(size(Xd1,1),1)*stdx);
    elseif strcmp(sctype,'autop')
        stdx=missstd(Xd1);     
        mx=missmean(Xd1);
        Xd1=(Xd1-ones(size(Xd1,1),1)*mx)./(ones(size(Xd1,1),1)*sqrt(stdx));
    elseif strcmp(sctype,'scalep')
        stdx=missstd(Xd1);     
        Xd1=Xd1./(ones(size(Xd1,1),1)*sqrt(stdx));
    elseif strcmp(sctype,'mean')
        mx=missmean(Xd1);
        Xd1=Xd1-ones(size(Xd1,1),1)*mx;
end
pret = {mx stdx};