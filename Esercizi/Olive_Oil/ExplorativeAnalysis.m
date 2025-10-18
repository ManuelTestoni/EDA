%% import excel by command lines
[olive, text]=xlsread('olive_oil'); 
obj_lab=text(2:383,1);
var_lab=text(1,2:end);
for i=1:7
    olive_lab{i}=deblank(var_lab{i});
end

%%Qui abbiamo le nostre variabili
olive_lab;
%%Qui abbiamo invece i valori della prima variabile "Categorie"
obj_lab;
%%Qui abbiamo invece i valori dei grassi nell olio.
olive;

% ===============================
% CONFIGURAZIONE DI BASE
% ===============================
output_folder = 'plots';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% ===============================
% Parallel Coordinates
% ===============================
figure;
parallelcoords(olive, 'group', obj_lab, 'standardize', 'on');
ax = gca;
ax.XTick = 1:length(olive_lab);     % posizioni dei tick
ax.XTickLabel = olive_lab;          % nomi degli acidi grassi
ax.XTickLabelRotation = 45;         % (opzionale) ruota per leggibilità
title('Parallel Coordinates - Dati standardizzati per categoria');
exportgraphics(gcf, fullfile(output_folder, 'parallel_coordinates.png'), 'Resolution', 300);
close(gcf);

% ===============================
% Gplotmatrix
% ===============================
figure;
gplotmatrix(olive, [], obj_lab, [], [], [], [], 'stairs', olive_lab, olive_lab);
sgtitle('Matrice di dispersione (Gplotmatrix) - Relazioni tra acidi grassi');
exportgraphics(gcf, fullfile(output_folder, 'gplotmatrix.png'), 'Resolution', 300);
close(gcf);

% ===============================
% Boxplot per ogni variabile
% ===============================
[n, m] = size(olive);
figure;
for i = 1:m
    msp = ceil(sqrt(m));
    subplot(msp, msp, i);
    boxplot(olive(:, i), obj_lab);
    title(olive_lab{i});
end
sgtitle('Boxplot per variabile e categoria');
exportgraphics(gcf, fullfile(output_folder, 'boxplot_variabili.png'), 'Resolution', 300);
close(gcf);

% ===============================
% Scatterhist (esempio su prime 2 variabili)
% ===============================
figure;
scatterhist(olive(:,1), olive(:,2), 'Group', obj_lab, 'Kernel', 'overlay');
xlabel(olive_lab{1});
ylabel(olive_lab{2});
title(['Scatterhist tra ', olive_lab{1}, ' e ', olive_lab{2}]);
exportgraphics(gcf, fullfile(output_folder, 'scatterhist.png'), 'Resolution', 300);
close(gcf);

% ===============================
% Frequency Histogram (tutte le variabili)
% ===============================
[n, m] = size(olive);
figure;
for i = 1:m
    nbin = ceil(sqrt(n));
    msp = ceil(sqrt(m));
    subplot(msp, msp, i);
    histogram(olive(:, i), nbin);
    title(olive_lab{i});
end
sgtitle('Istogrammi di frequenza per ciascun acido grasso');
exportgraphics(gcf, fullfile(output_folder, 'istogrammi_frequenza.png'), 'Resolution', 300);
close(gcf);

% ===============================
% Overlapping Histograms (stratificati per regione)
% ===============================
figure;
cat = unique(obj_lab, 'stable');
num_cat = numel(cat);
colors = lines(num_cat);  % genera colori distinti per ogni regione

for i = 1:m
    nbin = 2 * ceil(sqrt(n));
    msp = ceil(sqrt(m));
    subplot(msp, msp, i);
    hold on;
    
    % Per ogni regione, plotta l'istogramma
    for j = 1:num_cat
        data_region = olive(strcmp(obj_lab, cat{j}), i);
        histogram(data_region, nbin, 'FaceColor', colors(j, :), ...
                  'EdgeColor', 'black', 'FaceAlpha', 0.6, ...
                  'DisplayName', cat{j});
    end
    
    hold off;
    title(olive_lab{i});
    xlabel('Valore');
    ylabel('Frequenza');
    
    % Aggiungi legenda solo al primo subplot
    if i == 1
        legend('Location', 'best');
    end
end

sgtitle('Overlapping histograms - distribuzioni sovrapposte per regione');
exportgraphics(gcf, fullfile(output_folder, 'overlapping_histograms.png'), 'Resolution', 300);
close(gcf);

% ===============================
% Heatmap - Dati grezzi
% ===============================
figure;
imagesc(olive);
colormap('jet');
colorbar;

% Asse X = variabili (acidi grassi)
ax = gca;
set(ax, 'XTick', 1:m, 'XTickLabel', olive_lab, ...
        'XTickLabelRotation', 45);

% Asse Y = regioni (categorie)
cat = unique(obj_lab, 'stable');
ic = zeros(size(cat));
for i = 1:numel(cat)
    ic(i) = nnz(strcmp(obj_lab, cat{i}));
end
set(ax, 'YTick', cumsum(ic) - ic/2); % posizione media per ogni gruppo
set(ax, 'YTickLabel', cat);

title('Heatmap - Dati grezzi (regioni vs acidi grassi)');
xlabel('Acidi grassi');
ylabel('Regioni');
exportgraphics(gcf, fullfile(output_folder, 'heatmap_grezzi_regioni.png'), 'Resolution', 300);
close(gcf);

% ===============================
% Heatmap - Dati standardizzati
% ===============================
figure;
imagesc(normalize(olive, 'zscore'));
colormap('jet');
colorbar;

% Asse X = variabili (acidi grassi)
ax = gca;
set(ax, 'XTick', 1:m, 'XTickLabel', olive_lab, ...
        'XTickLabelRotation', 45);

% Asse Y = regioni (categorie)
cat = unique(obj_lab, 'stable');
ic = zeros(size(cat));
for i = 1:numel(cat)
    ic(i) = nnz(strcmp(obj_lab, cat{i}));
end
set(ax, 'YTick', cumsum(ic) - ic/2);
set(ax, 'YTickLabel', cat);

title('Heatmap - Dati standardizzati (regioni vs acidi grassi)');
xlabel('Acidi grassi');
ylabel('Regioni');
exportgraphics(gcf, fullfile(output_folder, 'heatmap_standardizzata_regioni.png'), 'Resolution', 300);
close(gcf);


% ===============================
% Correlation Matrix (al quadrato)
% ===============================
figure;
corr_matrix = corrcoef(olive).^2;
imagesc(corr_matrix);
colormap('jet');
colorbar;
ax = gca;

% Asse X e Y: variabili (acidi grassi)
set(ax, 'XTick', 1:m, 'XTickLabel', olive_lab, ...
        'XTickLabelRotation', 45);
set(ax, 'YTick', 1:m, 'YTickLabel', olive_lab);

title('Matrice di correlazione al quadrato (tra variabili)');
exportgraphics(gcf, fullfile(output_folder, 'matrice_correlazione.png'), 'Resolution', 300);
close(gcf);
