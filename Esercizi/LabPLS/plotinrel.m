function plotinrel(model,varargin)
% I/O: plotinrel(model)
if ~isempty(varargin)
    sel = 1;
else
    sel = 0;
end

T=model.loads{1,1};
U=model.loads{1,2};

A=size(T,2);

figure;
for i=1:A
    if A == 2
        subplot(1,2,i);
    else
        subplot(ceil(sqrt(A)),ceil(sqrt(A)),i);
    end
    plot(T(:,i),U(:,i),'.k','MarkerSize',8);
    axis tight;
    c=axis;
    hold on;plot([c(1) c(2)],[0 0],'-g');
    hold on;plot([0 0],[c(3) c(4)],'-g');
    xlabel(['T ',int2str(i)]);
    ylabel(['U ',int2str(i)]);
    title(['Inner Relation for LV ',int2str(i)]);
end

if sel == 1
    figure;
    plotgui(U);
end

% disp('Disegnare i loadings come frecce');
% disp('for i=1:nsamps');
% disp('line([0 model.loads{2,1}(i,1)],[0 model.loads{2,1}(i,2)],[0 model.loads{2,1}(i,3)]);');
% disp('end');