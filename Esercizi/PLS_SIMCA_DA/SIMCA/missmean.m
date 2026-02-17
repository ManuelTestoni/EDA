function [mm]=missmean(X,def)
%[mm]=missmean(X,def)
%
% Calculates the mean of each column of X, ignoring NaN values.
% If def==2, operates along rows instead.

if nargin > 1
    if def==2
        X=X';
    end
end

missidx = isnan(X);
i = find(missidx);
X(i) = 0;

if min(size(X))==1
   n_real=length(X)-sum(missidx);
else
   n_real=size(X,1)-sum(missidx);
end

i=find(n_real==0);
if isempty(i)
   mm=sum(X)./n_real;
else
   n_real(i)=1;
   mm=sum(X)./n_real;
   mm(i)=i + NaN;
end
