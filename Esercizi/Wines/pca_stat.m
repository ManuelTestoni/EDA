%% using PCA of the stastics and machine learning toolbox
% import data matrix
dati=importdata('wines_condescrvar.xls'); % importa il file 
datix=dati.data.WINES; % estrae la parte numerica
txt=dati.textdata.WINES; % estrae la parte di testo
%vlab=cell2str(txt(1,2:14)');
vlab=txt(1,2:14)';
vlab=deblank(vlab);
category=cell2str(txt(2:end,1));

%
X=datix; % nome della matrice dati
npc=1; % inserisci numero componenti
pret='mn';% inserisci as per autoscaling; mn per mean centering
if pret=='mn'
% calcola PCA con mean centering usando SVD
Xcentered = X - mean(X);
[U, S, V] = svd(Xcentered, 'econ');
% Prima calcola TUTTE le componenti per lo scree plot
all_eigenvalues = diag(S).^2 / (size(X,1)-1);
all_varexpl = 100 * all_eigenvalues / sum(all_eigenvalues);
cum_varexpl = cumsum(all_varexpl);

% SCREE PLOT e info per selezionare le PCs
figure('Name', 'PC Selection Tools', 'Position', [100 100 1200 400]);
subplot(1,3,1)
plot(1:length(all_varexpl), all_varexpl, 'o-', 'LineWidth', 2, 'MarkerSize', 8)
xlabel('Principal Component')
ylabel('Variance Explained (%)')
title('Scree Plot')
grid on

subplot(1,3,2)
bar(all_varexpl)
xlabel('Principal Component')
ylabel('Variance Explained (%)')
title('Variance per PC')
grid on

subplot(1,3,3)
plot(1:length(cum_varexpl), cum_varexpl, 's-', 'LineWidth', 2, 'MarkerSize', 8)
xlabel('Principal Component')
ylabel('Cumulative Variance Explained (%)')
title('Cumulative Variance')
grid on
yline(70, '--g', '70%', 'LineWidth', 1.5)
yline(80, '--b', '80%', 'LineWidth', 1.5)
yline(90, '--r', '90%', 'LineWidth', 1.5)

% Stampa info nella console
fprintf('\n========== PC SELECTION INFORMATION ==========\n')
fprintf('Total variables: %d\n', size(X,2))
fprintf('Total samples: %d\n', size(X,1))
fprintf('\nVariance explained by each PC:\n')
for i = 1:min(5, length(all_varexpl))
    fprintf('  PC%d: %.2f%% (Cumulative: %.2f%%)\n', i, all_varexpl(i), cum_varexpl(i))
end
fprintf('\nRecommended: Use PCs that explain at least 70-80%% of total variance\n')
fprintf('==============================================\n\n')

% Ora estrai solo le prime npc componenti richieste
loading = V(:, 1:npc);
scores = U(:, 1:npc) * S(1:npc, 1:npc);
eigenvalues = all_eigenvalues(1:npc);
varexpl = all_varexpl(1:npc);
% Calcola residui
Xrec = scores * loading';
residuals = Xcentered - Xrec;
recX = Xrec + mean(X);
% since all principal components are used as default to compute T2 even when fewer
%   components are requested by setting npc=.... To obtain T2 compatible
%   with the number of pc selcted use the command mahal 
T2ok=mahal(scores,scores);
T2 = sum((scores ./ sqrt(eigenvalues)').^2, 2);
elseif pret=='as'
% calcola PCA con autoscaling usando SVD
Xmean = mean(X);
Xstd = std(X);
Xscaled = (X - Xmean) ./ Xstd;
[U, S, V] = svd(Xscaled, 'econ');
% Prima calcola TUTTE le componenti per lo scree plot
all_eigenvalues = diag(S).^2 / (size(X,1)-1);
all_varexpl = 100 * all_eigenvalues / sum(all_eigenvalues);
cum_varexpl = cumsum(all_varexpl);

% SCREE PLOT e info per selezionare le PCs
figure('Name', 'PC Selection Tools', 'Position', [100 100 1200 400]);
subplot(1,3,1)
plot(1:length(all_varexpl), all_varexpl, 'o-', 'LineWidth', 2, 'MarkerSize', 8)
xlabel('Principal Component')
ylabel('Variance Explained (%)')
title('Scree Plot')
grid on
% Aggiungi linea a eigenvalue = 1 (se scaled)
yline(100/size(X,2), '--r', 'Kaiser Criterion (eigenvalue=1)', 'LineWidth', 1.5)

subplot(1,3,2)
bar(all_varexpl)
xlabel('Principal Component')
ylabel('Variance Explained (%)')
title('Variance per PC')
grid on

subplot(1,3,3)
plot(1:length(cum_varexpl), cum_varexpl, 's-', 'LineWidth', 2, 'MarkerSize', 8)
xlabel('Principal Component')
ylabel('Cumulative Variance Explained (%)')
title('Cumulative Variance')
grid on
yline(70, '--g', '70%', 'LineWidth', 1.5)
yline(80, '--b', '80%', 'LineWidth', 1.5)
yline(90, '--r', '90%', 'LineWidth', 1.5)

% Stampa info nella console
fprintf('\n========== PC SELECTION INFORMATION ==========\n')
fprintf('Total variables: %d\n', size(X,2))
fprintf('Total samples: %d\n', size(X,1))
fprintf('\nVariance explained by each PC:\n')
for i = 1:min(5, length(all_varexpl))
    fprintf('  PC%d: %.2f%% (Cumulative: %.2f%%)\n', i, all_varexpl(i), cum_varexpl(i))
end
fprintf('\nKaiser criterion (eigenvalue > 1): Select PCs with variance > %.2f%%\n', 100/size(X,2))
fprintf('Recommended: Use PCs that explain at least 70-80%% of total variance\n')
fprintf('==============================================\n\n')

% Ora estrai solo le prime npc componenti richieste
loading = V(:, 1:npc);
scores = U(:, 1:npc) * S(1:npc, 1:npc);
eigenvalues = all_eigenvalues(1:npc);
varexpl = all_varexpl(1:npc);
% Calcola residui
Xrec = scores * loading';
residuals = Xscaled - Xrec;
recX = Xrec .* Xstd + Xmean;
% Aggiusta i loadings per riportarli alla scala originale
loading = loading ./ Xstd';
T2ok=mahal(scores,scores);
T2 = sum((scores ./ sqrt(eigenvalues)').^2, 2);
end
Q=sum(residuals.^2,2);
% qlim with box method 95% confidence
% box
m=mean(sum(residuals.^2,2));
s=var(sum(residuals.^2,2));
c=chi2inv(0.95,2*m.^2/s);
qlim_box=c*s/(2*m); 
%T2lim 95% confidence
T2limit=((npc*(size(T2,1).^2-1))/(size(T2,1)*(size(T2,1)-npc)))*finv(0.95,npc,size(T2,1)-npc);
% plot loadings
xa=1; %PC1 su x cambia il numero se vuoi componenti diverse
ya=2; % PC2 su y cambia il numero se vuoi componenti diverse
figure;
scatter(loading(:,xa),loading(:, ya),'*')
hold on; textscatter(loading(:,xa),loading(:, ya),vlab)
for i=1:size(X,2)
    line([0 loading(i,xa)],[0 loading(i,ya)]);
end
xline(0);
yline(0);
%biplot(loading(:,[xa ya]),'VarLabels',vlab);
title('Loadings plot')
% add x , y labels
xlab=strcat('PC',int2str(xa),' %V (',num2str(varexpl(xa)),')');
xlabel(xlab)
ylab=strcat('PC',int2str(ya),' %V (',num2str(varexpl(ya)),')');
ylabel(ylab)
% to have a biplot
figure
biplot(loading(:,[xa ya]),'Scores',scores(:,[xa ya]),'VarLabels',vlab);
%plot scores, use gscatter to color per category
figure;
gscatter(scores(:,xa), scores(:, ya),category);
title('Scores plot')
% add grid
xline(0);
yline(0);
% add x , y labels
xlab=strcat('PC',int2str(xa),' %V (',num2str(varexpl(xa)),')');
xlabel(xlab)
ylab=strcat('PC',int2str(ya),' %V (',num2str(varexpl(ya)),')');
ylabel(ylab)
%plot T2 vs Q
figure;
gscatter(T2ok,Q,category);
xlabel(strcat('T2 with PC= ', int2str(npc)))
yline(qlim_box)
xline(T2limit)
