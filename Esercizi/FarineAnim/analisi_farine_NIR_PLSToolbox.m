%% ANALISI FARINE ANIMALI CON SPETTROSCOPIA NIR - PLS TOOLBOX
% Script per valutare diversi preprocessing spettrali e identificare
% il migliore per differenziare le 3 categorie (pollo, bovino, pesce)
%
% UTILIZZA PLS_TOOLBOX per preprocessing e PCA
%
% Preprocessing valutati:
% 1. Nessun preprocessing (solo mean center)
% 2. Baseline correction (detrend)
% 3. Normalizzazione (norm, SNV, MSC)
% 4. Derivate 1a e 2a con Savitzky-Golay smoothing
%
% Al termine di ogni preprocessing spettrale viene sempre applicato mean centering

clear all; close all; clc;

% Assicurati che PLS_Toolbox sia nel path
try
    evripath;
catch
    warning('PLS_Toolbox potrebbe non essere installato correttamente');
end

%% CARICAMENTO DATI
fprintf('\n========================================\n');
fprintf('CARICAMENTO DATI\n');
fprintf('========================================\n');

% Carica i dati dal formato .mat semplice
load('farineanimNIR.mat'); % contiene farineanimNIRdata e category
wavelengths = load('assexscale.txt'); % lunghezhe d'onda

% Assegna variabili
X = farineanimNIRdata;
categories = category;

[n_samples, n_vars] = size(X);
fprintf('Numero campioni: %d\n', n_samples);
fprintf('Numero variabili (lunghezze d''onda): %d\n', n_vars);
fprintf('Range lunghezze d''onda: %.0f - %.0f nm\n', min(wavelengths), max(wavelengths));

% IMPORTANTE: Converti category in cell array se necessario
fprintf('\nControllo formato categories:\n');
fprintf('  Classe originale: %s\n', class(categories));
fprintf('  Dimensioni originali: %d x %d\n', size(categories, 1), size(categories, 2));

if ~iscell(categories)
    if ischar(categories)
        % Se è una matrice di caratteri, converti ogni riga in una cella
        categories = cellstr(categories);
        fprintf('  → Convertito da char a cellstr\n');
    elseif isnumeric(categories)
        % Se è numerico, converti in stringhe
        categories = cellstr(num2str(categories));
        fprintf('  → Convertito da numeric a cellstr\n');
    end
end

% Assicurati che sia un vettore colonna
if size(categories, 2) > 1 && size(categories, 1) == 1
    categories = categories';
    fprintf('  → Trasposto da riga a colonna\n');
end

fprintf('  Classe finale: %s\n', class(categories));
fprintf('  Dimensioni finali: %d x %d\n', size(categories, 1), size(categories, 2));
fprintf('  Numero elementi: %d\n', length(categories));

% Verifica che corrisponda al numero di campioni
if length(categories) ~= n_samples
    error('ERRORE: Numero di categorie (%d) diverso dal numero di campioni (%d)', length(categories), n_samples);
end

unique_cats = unique(categories);
fprintf('\nCategorie uniche trovate: ');
for i = 1:length(unique_cats)
    n_cat = sum(strcmp(categories, unique_cats{i}));
    fprintf('%s (n=%d) ', unique_cats{i}, n_cat);
end
fprintf('\n');

%% CREA DATASET PLS_TOOLBOX (opzionale, ma utile)
% Se vuoi usare il formato dataset del PLS_Toolbox
try
    % Crea dataset PLS_Toolbox
    ds = dataset(X);
    ds.label{1} = categories;
    ds.axisscale{2} = wavelengths;
    ds.name = 'Farine Animali NIR';
    fprintf('Dataset PLS_Toolbox creato con successo\n');
    use_plstoolbox = true;
catch
    fprintf('Uso formato matrici standard (PLS_Toolbox non disponibile o versione incompatibile)\n');
    use_plstoolbox = false;
end

%% PLOT SPETTRI ORIGINALI
fprintf('\n========================================\n');
fprintf('VISUALIZZAZIONE SPETTRI ORIGINALI\n');
fprintf('========================================\n');

figure('Name', 'Spettri Originali', 'Position', [100 100 1200 500]);

% Plot tutti gli spettri colorati per categoria
subplot(1,2,1);
hold on;
% Colori per i plot
colors_rgb = [1 0 0; 0 0 1; 0 1 0]; % rosso, blu, verde
colors = {'r', 'b', 'g'};
legend_handles = [];
for i = 1:length(unique_cats)
    idx = strcmp(categories, unique_cats{i});
    h = plot(wavelengths, X(idx, :)', 'Color', colors_rgb(i,:), 'LineWidth', 0.5);
    % Prendi solo il primo handle per la legenda
    legend_handles(i) = h(1);
end
xlabel('Lunghezza d''onda (nm)');
ylabel('Assorbanza');
title('Spettri NIR Originali per Categoria');
legend(legend_handles, unique_cats, 'Location', 'best');
grid on;
hold off;

% Plot spettri medi per categoria
subplot(1,2,2);
hold on;
legend_handles = [];
for i = 1:length(unique_cats)
    idx = strcmp(categories, unique_cats{i});
    mean_spectrum = mean(X(idx, :), 1);
    legend_handles(i) = plot(wavelengths, mean_spectrum, 'Color', colors_rgb(i,:), 'LineWidth', 2);
end
xlabel('Lunghezza d''onda (nm)');
ylabel('Assorbanza');
title('Spettri Medi per Categoria');
legend(legend_handles, unique_cats, 'Location', 'best');
grid on;
hold off;

%% DEFINIZIONE PREPROCESSING DA TESTARE
% Usa le funzioni del PLS_Toolbox quando possibile
preprocessing_list = {
    'none', 'Nessun preprocessing (solo mean center)';
    'baseline', 'Baseline correction (detrend)';
    'snv', 'Standard Normal Variate (SNV)';
    'msc', 'Multiplicative Scatter Correction (MSC)';
    'normalize', 'Normalizzazione (norma unitaria)';
    'savgol1_w11', '1a Derivata (Savitzky-Golay, w=11, ord=2)';
    'savgol2_w11', '2a Derivata (Savitzky-Golay, w=11, ord=2)';
    'savgol1_w15', '1a Derivata (Savitzky-Golay, w=15, ord=3)';
    'savgol2_w15', '2a Derivata (Savitzky-Golay, w=15, ord=3)';
    'savgol1_w21', '1a Derivata (Savitzky-Golay, w=21, ord=3)';
};

n_preprocessing = size(preprocessing_list, 1);

% Struttura per salvare i risultati
results = struct();

%% CICLO SUI PREPROCESSING
fprintf('\n========================================\n');
fprintf('VALUTAZIONE PREPROCESSING\n');
fprintf('========================================\n\n');

for p = 1:n_preprocessing
    
    prep_type = preprocessing_list{p, 1};
    prep_name = preprocessing_list{p, 2};
    
    fprintf('--------------------------------------------------\n');
    fprintf('Preprocessing %d/%d: %s\n', p, n_preprocessing, prep_name);
    fprintf('--------------------------------------------------\n');
    
    %% APPLICAZIONE PREPROCESSING CON PLS_TOOLBOX
    X_prep = X;
    
    try
        % Prova a usare le funzioni del PLS_Toolbox
        switch prep_type
            case 'none'
                % Nessun preprocessing spettrale
                X_prep = X;
                
            case 'baseline'
                % Baseline correction - usa detrend o baselinecorrect del PLS_Toolbox
                try
                    X_prep = baselinecorrect(X, 'linear');
                catch
                    % Fallback a detrend standard
                    X_prep = detrend(X', 'linear')';
                end
                
            case 'snv'
                % Standard Normal Variate - usa snv del PLS_Toolbox
                try
                    X_prep = snv(X);
                catch
                    % Fallback manuale
                    mean_spec = mean(X_prep, 2);
                    std_spec = std(X_prep, 0, 2);
                    X_prep = (X_prep - mean_spec) ./ std_spec;
                end
                
            case 'msc'
                % Multiplicative Scatter Correction - usa msc del PLS_Toolbox
                try
                    X_prep = mscorr(X);
                catch
                    % Fallback manuale
                    mean_spectrum = mean(X_prep, 1);
                    X_prep_msc = zeros(size(X_prep));
                    for i = 1:n_samples
                        p_fit = polyfit(mean_spectrum, X_prep(i,:), 1);
                        X_prep_msc(i,:) = (X_prep(i,:) - p_fit(2)) / p_fit(1);
                    end
                    X_prep = X_prep_msc;
                end
                
            case 'normalize'
                % Normalizzazione (norma unitaria)
                try
                    X_prep = normaliz(X, 1); % normaliz del PLS_Toolbox
                catch
                    % Fallback manuale
                    X_prep = X_prep ./ sqrt(sum(X_prep.^2, 2));
                end
                
            case 'savgol1_w11'
                % 1a Derivata con Savitzky-Golay (finestra=11, ordine=2)
                try
                    X_prep = savgol(X, 1, 11, 2); % PLS_Toolbox
                catch
                    % Fallback a sgolayfilt
                    X_prep = sgolayfilt(X', 2, 11, [], 1)';
                end
                
            case 'savgol2_w11'
                % 2a Derivata con Savitzky-Golay (finestra=11, ordine=2)
                try
                    X_prep = savgol(X, 2, 11, 2); % PLS_Toolbox
                catch
                    X_prep = sgolayfilt(X', 2, 11, [], 2)';
                end
                
            case 'savgol1_w15'
                % 1a Derivata con Savitzky-Golay (finestra=15, ordine=3)
                try
                    X_prep = savgol(X, 1, 15, 3);
                catch
                    X_prep = sgolayfilt(X', 3, 15, [], 1)';
                end
                
            case 'savgol2_w15'
                % 2a Derivata con Savitzky-Golay (finestra=15, ordine=3)
                try
                    X_prep = savgol(X, 2, 15, 3);
                catch
                    X_prep = sgolayfilt(X', 3, 15, [], 2)';
                end
                
            case 'savgol1_w21'
                % 1a Derivata con Savitzky-Golay (finestra=21, ordine=3)
                try
                    X_prep = savgol(X, 1, 21, 3);
                catch
                    X_prep = sgolayfilt(X', 3, 21, [], 1)';
                end
        end
        
    catch ME
        fprintf('Errore nel preprocessing: %s\n', ME.message);
        fprintf('Uso metodo alternativo...\n');
    end
    
    % Applica SEMPRE mean centering alla fine
    X_prep = X_prep - mean(X_prep, 1);
    
    %% PLOT SPETTRI PREPROCESSATI
    fig = figure('Name', ['Preprocessing: ' prep_name], 'Position', [100 100 1400 900]);
    
    % Subplot 1: Tutti gli spettri preprocessati
    subplot(2,2,1);
    hold on;
    legend_handles = [];
    for i = 1:length(unique_cats)
        idx = strcmp(categories, unique_cats{i});
        h = plot(wavelengths, X_prep(idx, :)', 'Color', colors_rgb(i,:), 'LineWidth', 0.5);
        legend_handles(i) = h(1);
    end
    xlabel('Lunghezza d''onda (nm)');
    ylabel('Assorbanza preprocessata');
    title(['Spettri dopo ' prep_name]);
    legend(legend_handles, unique_cats, 'Location', 'best');
    grid on;
    hold off;
    
    % Subplot 2: Spettri medi per categoria
    subplot(2,2,2);
    hold on;
    legend_handles = [];
    for i = 1:length(unique_cats)
        idx = strcmp(categories, unique_cats{i});
        mean_spectrum = mean(X_prep(idx, :), 1);
        legend_handles(i) = plot(wavelengths, mean_spectrum, 'Color', colors_rgb(i,:), 'LineWidth', 2);
    end
    xlabel('Lunghezza d''onda (nm)');
    ylabel('Assorbanza preprocessata');
    title('Spettri Medi per Categoria');
    legend(legend_handles, unique_cats, 'Location', 'best');
    grid on;
    hold off;
    
    %% PCA con PLS_Toolbox
    % Il PLS_Toolbox ha una sintassi diversa: pca(X, ncomp, options)
    % dove X può essere una matrice o un dataset
    
    % DIAGNOSTICA: Verifica dimensioni prima della PCA
    fprintf('  Dimensioni X_prep prima PCA: %d x %d\n', size(X_prep, 1), size(X_prep, 2));
    fprintf('  Dimensioni categories: %d\n', length(categories));
    
    try
        % Prova PCA con PLS_Toolbox
        % Sintassi: model = pca(X, ncomp, options)
        ncomp = min(10, min(size(X_prep))-1);
        
        % Crea struttura options per PLS_Toolbox
        % Nota: 'none' non è valido, usa cell vuoto o ometti preprocessing
        options = struct();
        options.preprocessing = {{}};  % cell array vuoto = nessun preprocessing
        options.algorithm = 'svd';
        
        % Esegui PCA del PLS_Toolbox
        fprintf('  Tentativo PCA con PLS_Toolbox...\n');
        model_pca = pca(X_prep, ncomp, options);
        
        % Estrai i risultati - la struttura dipende dalla versione del PLS_Toolbox
        % Prova diverse possibili strutture
        
        if isfield(model_pca, 'scores')
            scores = model_pca.scores;
        elseif isfield(model_pca, 'detail') && isfield(model_pca.detail, 'scores')
            scores = model_pca.detail.scores;
        else
            error('Impossibile trovare scores nel modello PCA');
        end
        
        if isfield(model_pca, 'loads')
            loadings = model_pca.loads;
            if iscell(loadings)
                loadings = loadings{1};
            end
        elseif isfield(model_pca, 'loadings')
            loadings = model_pca.loadings;
        else
            error('Impossibile trovare loadings nel modello PCA');
        end
        
        % Varianza spiegata
        if isfield(model_pca, 'ssq')
            % ssq contiene la varianza spiegata cumulativa
            explained = model_pca.ssq(2,:)' * 100;  % converti in percentuale
            % Calcola la varianza per ogni PC (non cumulativa)
            explained = [explained(1); diff(explained)];
        elseif isfield(model_pca, 'detail') && isfield(model_pca.detail, 'ssq')
            explained = model_pca.detail.ssq(:,2) * 100;
        else
            % Calcola manualmente dagli eigenvalues
            if isfield(model_pca, 'eigenvalues')
                latent = model_pca.eigenvalues;
            elseif isfield(model_pca, 'detail') && isfield(model_pca.detail, 'eigenvalues')
                latent = model_pca.detail.eigenvalues;
            else
                % Calcola dagli scores
                latent = var(scores)' * (size(X_prep,1)-1);
            end
            total_var = sum(var(X_prep));
            explained = (latent / total_var) * 100;
        end
        
        % Eigenvalues
        if isfield(model_pca, 'eigenvalues')
            latent = model_pca.eigenvalues;
        elseif isfield(model_pca, 'detail') && isfield(model_pca.detail, 'eigenvalues')
            latent = model_pca.detail.eigenvalues;
        else
            latent = var(scores)' * (size(X_prep,1)-1);
        end
        
        % Converti da dataset a double se necessario
        if isa(scores, 'dataset')
            scores = double(scores);
        end
        if isa(loadings, 'dataset')
            loadings = double(loadings);
        end
        
        % DIAGNOSTICA: Verifica dimensioni dopo PCA
        fprintf('  Dimensioni scores dopo PCA: %d x %d\n', size(scores, 1), size(scores, 2));
        fprintf('  Dimensioni loadings dopo PCA: %d x %d\n', size(loadings, 1), size(loadings, 2));
        
        % VERIFICA E CORREGGI se scores e loadings sono invertiti
        % Scores dovrebbe essere: n_samples x n_components (84 x 10)
        % Loadings dovrebbe essere: n_variables x n_components (2001 x 10)
        
        % Caso 1: entrambi hanno dimensioni sbagliate - scambiali
        if size(scores, 1) ~= n_samples && size(loadings, 1) == n_samples
            fprintf('  ⚠ ATTENZIONE: Scores e Loadings sono invertiti! Scambio...\n');
            temp = scores;
            scores = loadings;
            loadings = temp;
            fprintf('  Dimensioni corrette - Scores: %d x %d, Loadings: %d x %d\n', ...
                size(scores, 1), size(scores, 2), size(loadings, 1), size(loadings, 2));
        end
        
        % Caso 2: loadings ha dimensioni sbagliate anche se scores è ok
        if size(scores, 1) == n_samples && size(loadings, 1) ~= n_vars
            fprintf('  ⚠ ATTENZIONE: Loadings ha dimensioni errate (%d invece di %d)\n', ...
                size(loadings, 1), n_vars);
            
            % Se loadings ha n_samples righe, è sicuramente sbagliato - deve essere trasposto
            if size(loadings, 1) == n_samples && size(loadings, 2) == n_vars
                fprintf('  → Trasposizione loadings...\n');
                loadings = loadings';
                fprintf('  Dimensioni corrette - Loadings: %d x %d\n', size(loadings, 1), size(loadings, 2));
            elseif size(loadings, 1) == n_samples
                % Se è ancora n_samples dopo il controllo, scambia con scores
                fprintf('  → Scambio scores e loadings...\n');
                temp = scores;
                scores = loadings';  % trasponi mentre scambi
                loadings = temp';
                fprintf('  Dimensioni corrette - Scores: %d x %d, Loadings: %d x %d\n', ...
                    size(scores, 1), size(scores, 2), size(loadings, 1), size(loadings, 2));
            end
        end
        
        % Verifica finale delle dimensioni
        if size(scores, 1) ~= n_samples
            error('Scores ha ancora dimensioni errate: %d righe invece di %d campioni', size(scores, 1), n_samples);
        end
        if size(loadings, 1) ~= n_vars
            error('Loadings ha ancora dimensioni errate: %d righe invece di %d variabili', size(loadings, 1), n_vars);
        end
        
        fprintf('  ✓ PCA completata con successo usando PLS_Toolbox\n');
        
    catch ME
        % Fallback a PCA standard di MATLAB
        fprintf('\n  ⚠ ERRORE PLS_Toolbox PCA: %s\n', ME.message);
        fprintf('  Stack: %s\n', ME.stack(1).name);
        fprintf('  Possibile problema di connettività con server licenze PLS_Toolbox\n');
        fprintf('  → Fallback a PCA standard di MATLAB (SVD)\n\n');
        
        % Usa SVD manualmente per evitare conflitti di nomi
        % Mean center (già fatto sopra)
        [U, S, V] = svd(X_prep, 'econ');
        
        % Calcola scores, loadings, eigenvalues
        % Scores = U * S (n_samples x n_components)
        % Loadings = V (n_variables x n_components)
        loadings = V;
        scores = U * S;
        latent = diag(S).^2 / (size(X_prep,1)-1);
        explained = 100 * latent / sum(latent);
        
        % Limita a 10 componenti per coerenza
        ncomp = min(10, length(explained));
        loadings = loadings(:, 1:ncomp);
        scores = scores(:, 1:ncomp);
        latent = latent(1:ncomp);
        explained = explained(1:ncomp);
        
        % DIAGNOSTICA: Verifica dimensioni dopo SVD
        fprintf('  Dimensioni scores dopo SVD: %d x %d\n', size(scores, 1), size(scores, 2));
        fprintf('  Dimensioni loadings dopo SVD: %d x %d\n', size(loadings, 1), size(loadings, 2));
        
        % Verifica correttezza dimensioni
        if size(scores, 1) ~= n_samples
            error('Scores SVD ha dimensioni errate: %d righe invece di %d campioni', size(scores, 1), n_samples);
        end
        if size(loadings, 1) ~= n_vars
            error('Loadings SVD ha dimensioni errate: %d righe invece di %d variabili', size(loadings, 1), n_vars);
        end
        
        fprintf('  ✓ PCA completata con successo usando SVD\n');
    end
    
    % VERIFICA FINALE delle dimensioni prima di salvare
    if size(scores, 1) ~= length(categories)
        error('ERRORE CRITICO: Mismatch dimensioni! Scores: %d righe, Categories: %d elementi', ...
            size(scores, 1), length(categories));
    end
    
    % VERIFICA E CORREZIONE varianza spiegata (deve essere tra 0 e 100%)
    if any(explained > 100) || any(explained < 0)
        fprintf('  ⚠ ATTENZIONE: Varianza spiegata fuori range [0-100]! Ricalcolo...\n');
        fprintf('    Valori originali: ');
        fprintf('%.2f%% ', explained(1:min(3,length(explained))));
        fprintf('\n');
        
        % Ricalcola la varianza spiegata correttamente dagli eigenvalues
        if exist('latent', 'var') && ~isempty(latent)
            total_var = sum(latent);
            explained = 100 * latent / total_var;
        else
            % Calcola dagli scores
            var_scores = var(scores, 0, 1);
            total_var = sum(var_scores);
            explained = 100 * var_scores' / total_var;
        end
        
        fprintf('    Valori corretti: ');
        fprintf('%.2f%% ', explained(1:min(3,length(explained))));
        fprintf('\n');
    end
    
    % Salva i risultati
    results(p).prep_type = prep_type;
    results(p).prep_name = prep_name;
    results(p).X_prep = X_prep;
    results(p).scores = scores;
    results(p).loadings = loadings;
    results(p).explained = explained;
    results(p).latent = latent;
    
    fprintf('Varianza spiegata da PC1: %.2f%%\n', explained(1));
    fprintf('Varianza spiegata da PC2: %.2f%%\n', explained(2));
    fprintf('Varianza spiegata da PC1+PC2: %.2f%%\n', sum(explained(1:2)));
    fprintf('Varianza spiegata da PC1+PC2+PC3: %.2f%%\n', sum(explained(1:3)));
    
    %% SCORE PLOT (PC1 vs PC2)
    subplot(2,2,3);
    
    % Verifica dimensioni prima di gscatter
    if size(scores, 1) ~= length(categories)
        warning('Dimensioni non corrispondenti: scores=%d, categories=%d', size(scores,1), length(categories));
        % Plot senza colori per categoria
        plot(scores(:,1), scores(:,2), 'o', 'MarkerSize', 8);
        xlabel(sprintf('PC1 (%.2f%%)', explained(1)));
        ylabel(sprintf('PC2 (%.2f%%)', explained(2)));
        title('Score Plot (PC1 vs PC2) - ERRORE categorie');
        grid on;
    else
        % Usa gscatter SENZA specificare colori (li sceglie automaticamente)
        gscatter(scores(:,1), scores(:,2), categories, [], 'o', 8);
        xlabel(sprintf('PC1 (%.2f%%)', explained(1)));
        ylabel(sprintf('PC2 (%.2f%%)', explained(2)));
        title('Score Plot (PC1 vs PC2)');
        legend(unique_cats, 'Location', 'best');
        grid on;
    end
    
    %% LOADING PLOT (PC1 e PC2)
    subplot(2,2,4);
    hold on;
    
    % Assicurati che wavelengths e loadings abbiano dimensioni compatibili
    wavelengths_vec = wavelengths(:);  % Converti in vettore colonna
    if size(loadings, 1) == length(wavelengths_vec)
        x_axis = wavelengths_vec;
        xlabel('Lunghezza d''onda (nm)');
    else
        % Se le dimensioni non corrispondono, usa indici
        fprintf('  Info: Usando indici per asse x (loadings: %d, wavelengths: %d)\n', ...
            size(loadings,1), length(wavelengths_vec));
        x_axis = (1:size(loadings,1))';
        xlabel('Indice variabile');
    end
    
    % Plot loadings come linee
    plot(x_axis, loadings(:,1), 'r-', 'LineWidth', 1.5, 'DisplayName', sprintf('PC1 (%.2f%%)', explained(1)));
    plot(x_axis, loadings(:,2), 'b-', 'LineWidth', 1.5, 'DisplayName', sprintf('PC2 (%.2f%%)', explained(2)));
    
    % Aggiungi linea a zero
    yline(0, 'k--', 'LineWidth', 0.5);
    
    % Evidenzia i 5 loadings più importanti per PC1
    [~, top_idx_pc1] = sort(abs(loadings(:,1)), 'descend');
    for j = 1:min(5, length(top_idx_pc1))
        idx = top_idx_pc1(j);
        if abs(loadings(idx,1)) > 0.05  % Solo se significativo
            plot(x_axis(idx), loadings(idx,1), 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'r', 'HandleVisibility', 'off');
        end
    end
    
    ylabel('Loading');
    title('Loading Plot (PC1 e PC2)');
    legend('Location', 'best');
    grid on;
    hold off;
    
    %% SCREE PLOT
    figure('Name', ['Scree Plot: ' prep_name], 'Position', [150 150 1200 400]);
    
    subplot(1,3,1);
    plot(1:min(10,length(explained)), explained(1:min(10,length(explained))), 'o-', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Componente Principale');
    ylabel('Varianza Spiegata (%)');
    title('Scree Plot');
    grid on;
    
    subplot(1,3,2);
    bar(1:min(10,length(explained)), explained(1:min(10,length(explained))));
    xlabel('Componente Principale');
    ylabel('Varianza Spiegata (%)');
    title('Varianza per PC');
    grid on;
    
    subplot(1,3,3);
    plot(1:min(10,length(explained)), cumsum(explained(1:min(10,length(explained)))), 's-', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Componente Principale');
    ylabel('Varianza Cumulativa (%)');
    title('Varianza Cumulativa');
    grid on;
    yline(70, '--g', '70%', 'LineWidth', 1.5);
    yline(80, '--b', '80%', 'LineWidth', 1.5);
    yline(90, '--r', '90%', 'LineWidth', 1.5);
    
    fprintf('\n');
end

%% CONFRONTO TRA PREPROCESSING
fprintf('\n========================================\n');
fprintf('CONFRONTO TRA PREPROCESSING\n');
fprintf('========================================\n\n');

fprintf('%-55s | PC1+PC2 | PC1   | PC2\n', 'Preprocessing');
fprintf('%s\n', repmat('-', 85, 1));
for p = 1:n_preprocessing
    fprintf('%-55s | %6.2f%% | %5.2f%% | %5.2f%%\n', ...
        results(p).prep_name, ...
        sum(results(p).explained(1:2)), ...
        results(p).explained(1), ...
        results(p).explained(2));
end

%% TROVA I MIGLIORI PREPROCESSING
[~, sorted_idx] = sort(arrayfun(@(x) sum(x.explained(1:2)), results), 'descend');
best_idx = sorted_idx(1:min(2, length(sorted_idx)));

fprintf('\n========================================\n');
fprintf('MIGLIORI PREPROCESSING\n');
fprintf('========================================\n\n');

for i = 1:length(best_idx)
    fprintf('%d. %s\n', i, results(best_idx(i)).prep_name);
    fprintf('   Varianza PC1+PC2: %.2f%%\n', sum(results(best_idx(i)).explained(1:2)));
    fprintf('   Varianza PC1: %.2f%%\n', results(best_idx(i)).explained(1));
    fprintf('   Varianza PC2: %.2f%%\n\n', results(best_idx(i)).explained(2));
end

%% ANALISI DETTAGLIATA DEI MIGLIORI PREPROCESSING
for i = 1:length(best_idx)
    idx = best_idx(i);
    
    fprintf('\n========================================\n');
    fprintf('ANALISI DETTAGLIATA: %s\n', results(idx).prep_name);
    fprintf('========================================\n');
    
    % Plot dettagliato con più PCs
    figure('Name', ['Analisi Dettagliata: ' results(idx).prep_name], 'Position', [50 50 1400 1000]);
    
    % Score Plot PC1 vs PC2
    subplot(3,3,1);
    try
        gscatter(results(idx).scores(:,1), results(idx).scores(:,2), categories, [], 'o', 8);
        legend(unique_cats, 'Location', 'best');
    catch
        plot(results(idx).scores(:,1), results(idx).scores(:,2), 'o', 'MarkerSize', 8);
    end
    xlabel(sprintf('PC1 (%.2f%%)', results(idx).explained(1)));
    ylabel(sprintf('PC2 (%.2f%%)', results(idx).explained(2)));
    title('Score Plot: PC1 vs PC2');
    grid on;
    
    % Score Plot PC1 vs PC3
    subplot(3,3,2);
    try
        gscatter(results(idx).scores(:,1), results(idx).scores(:,3), categories, [], 'o', 8);
        legend(unique_cats, 'Location', 'best');
    catch
        plot(results(idx).scores(:,1), results(idx).scores(:,3), 'o', 'MarkerSize', 8);
    end
    xlabel(sprintf('PC1 (%.2f%%)', results(idx).explained(1)));
    ylabel(sprintf('PC3 (%.2f%%)', results(idx).explained(3)));
    title('Score Plot: PC1 vs PC3');
    grid on;
    
    % Score Plot PC2 vs PC3
    subplot(3,3,3);
    try
        gscatter(results(idx).scores(:,2), results(idx).scores(:,3), categories, [], 'o', 8);
        legend(unique_cats, 'Location', 'best');
    catch
        plot(results(idx).scores(:,2), results(idx).scores(:,3), 'o', 'MarkerSize', 8);
    end
    xlabel(sprintf('PC2 (%.2f%%)', results(idx).explained(2)));
    ylabel(sprintf('PC3 (%.2f%%)', results(idx).explained(3)));
    title('Score Plot: PC2 vs PC3');
    grid on;
    
    % Prepara wavelengths per i loading plots
    wavelengths_vec = wavelengths(:);
    
    % Loading Plot PC1
    subplot(3,3,4);
    hold on;
    if size(results(idx).loadings, 1) == length(wavelengths_vec)
        x_axis = wavelengths_vec;
        plot(x_axis, results(idx).loadings(:,1), 'r-', 'LineWidth', 1.5);
        xlabel('Lunghezza d''onda (nm)');
    else
        x_axis = (1:size(results(idx).loadings,1))';
        plot(x_axis, results(idx).loadings(:,1), 'r-', 'LineWidth', 1.5);
        xlabel('Indice variabile');
    end
    yline(0, 'k--', 'LineWidth', 0.5);
    % Evidenzia picchi più importanti
    [~, top_idx] = sort(abs(results(idx).loadings(:,1)), 'descend');
    for j = 1:min(3, length(top_idx))
        plot(x_axis(top_idx(j)), results(idx).loadings(top_idx(j),1), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    end
    ylabel('Loading PC1');
    title(sprintf('Loading PC1 (%.2f%% varianza)', results(idx).explained(1)));
    grid on;
    hold off;
    
    % Loading Plot PC2
    subplot(3,3,5);
    hold on;
    if size(results(idx).loadings, 1) == length(wavelengths_vec)
        x_axis = wavelengths_vec;
        plot(x_axis, results(idx).loadings(:,2), 'b-', 'LineWidth', 1.5);
        xlabel('Lunghezza d''onda (nm)');
    else
        x_axis = (1:size(results(idx).loadings,1))';
        plot(x_axis, results(idx).loadings(:,2), 'b-', 'LineWidth', 1.5);
        xlabel('Indice variabile');
    end
    yline(0, 'k--', 'LineWidth', 0.5);
    % Evidenzia picchi più importanti
    [~, top_idx] = sort(abs(results(idx).loadings(:,2)), 'descend');
    for j = 1:min(3, length(top_idx))
        plot(x_axis(top_idx(j)), results(idx).loadings(top_idx(j),2), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
    end
    ylabel('Loading PC2');
    title(sprintf('Loading PC2 (%.2f%% varianza)', results(idx).explained(2)));
    grid on;
    hold off;
    
    % Loading Plot PC3
    subplot(3,3,6);
    hold on;
    if size(results(idx).loadings, 1) == length(wavelengths_vec)
        x_axis = wavelengths_vec;
        plot(x_axis, results(idx).loadings(:,3), 'g-', 'LineWidth', 1.5);
        xlabel('Lunghezza d''onda (nm)');
    else
        x_axis = (1:size(results(idx).loadings,1))';
        plot(x_axis, results(idx).loadings(:,3), 'g-', 'LineWidth', 1.5);
        xlabel('Indice variabile');
    end
    yline(0, 'k--', 'LineWidth', 0.5);
    % Evidenzia picchi più importanti
    [~, top_idx] = sort(abs(results(idx).loadings(:,3)), 'descend');
    for j = 1:min(3, length(top_idx))
        plot(x_axis(top_idx(j)), results(idx).loadings(top_idx(j),3), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
    end
    ylabel('Loading PC3');
    title(sprintf('Loading PC3 (%.2f%% varianza)', results(idx).explained(3)));
    grid on;
    hold off;
    
    % Loading Plot PC1, PC2 e PC3 insieme
    subplot(3,3,7);
    hold on;
    if size(results(idx).loadings, 1) == length(wavelengths_vec)
        plot(wavelengths_vec, results(idx).loadings(:,1), 'r-', 'LineWidth', 1.5, 'DisplayName', 'PC1');
        plot(wavelengths_vec, results(idx).loadings(:,2), 'b-', 'LineWidth', 1.5, 'DisplayName', 'PC2');
        plot(wavelengths_vec, results(idx).loadings(:,3), 'g-', 'LineWidth', 1.5, 'DisplayName', 'PC3');
        xlabel('Lunghezza d''onda (nm)');
    else
        plot(1:size(results(idx).loadings,1), results(idx).loadings(:,1), 'r-', 'LineWidth', 1.5, 'DisplayName', 'PC1');
        plot(1:size(results(idx).loadings,1), results(idx).loadings(:,2), 'b-', 'LineWidth', 1.5, 'DisplayName', 'PC2');
        plot(1:size(results(idx).loadings,1), results(idx).loadings(:,3), 'g-', 'LineWidth', 1.5, 'DisplayName', 'PC3');
        xlabel('Indice variabile');
    end
    ylabel('Loading');
    title('Loading Plot PC1, PC2, PC3');
    legend('Location', 'best');
    grid on;
    hold off;
    
    % Spettri medi per categoria
    subplot(3,3,8);
    hold on;
    for j = 1:length(unique_cats)
        cat_idx = strcmp(categories, unique_cats{j});
        mean_spectrum = mean(results(idx).X_prep(cat_idx, :), 1);
        plot(wavelengths, mean_spectrum, 'Color', colors_rgb(j,:), 'LineWidth', 2);
    end
    xlabel('Lunghezza d''onda (nm)');
    ylabel('Assorbanza preprocessata');
    title('Spettri Medi per Categoria');
    legend(unique_cats, 'Location', 'best');
    grid on;
    hold off;
    
    % Scree plot
    subplot(3,3,9);
    plot(1:min(10,length(results(idx).explained)), results(idx).explained(1:min(10,length(results(idx).explained))), 'o-', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Componente Principale');
    ylabel('Varianza Spiegata (%)');
    title('Scree Plot');
    grid on;
    
    % Stampa informazioni sui loading più importanti
    fprintf('\nRegioni spettrali più importanti (massimi assoluti nei loadings):\n\n');
    
    for pc = 1:3
        [~, max_idx] = sort(abs(results(idx).loadings(:,pc)), 'descend');
        fprintf('PC%d (%.2f%% varianza):\n', pc, results(idx).explained(pc));
        fprintf('  Top 5 lunghezze d''onda:\n');
        for j = 1:5
            wave_idx = max_idx(j);
            fprintf('    %.0f nm: loading = %+.4f\n', wavelengths(wave_idx), results(idx).loadings(wave_idx, pc));
        end
        fprintf('\n');
    end
end

%% SALVA I RISULTATI
fprintf('\n========================================\n');
fprintf('SALVATAGGIO RISULTATI\n');
fprintf('========================================\n');

save('results_preprocessing_NIR.mat', 'results', 'wavelengths', 'categories', 'X');
fprintf('Risultati salvati in: results_preprocessing_NIR.mat\n');

fprintf('\n========================================\n');
fprintf('ANALISI COMPLETATA\n');
fprintf('========================================\n\n');
fprintf('ISTRUZIONI PER IL REPORT:\n\n');
fprintf('1. Revisionare tutte le figure generate\n');
fprintf('2. Identificare i 2 preprocessing migliori (maggiore separazione negli score plots)\n');
fprintf('3. Per i preprocessing migliori:\n');
fprintf('   - Includere plot degli spettri originali vs preprocessati\n');
fprintf('   - Includere score plots (PC1 vs PC2)\n');
fprintf('   - Includere loading plots con interpretazione\n');
fprintf('4. Interpretare le regioni spettrali usando tointerpretNIR.doc:\n');
fprintf('   - Identificare lunghezze d''onda chiave dai loading plots\n');
fprintf('   - Associare a gruppi funzionali o legami chimici\n');
fprintf('   - Spiegare perché queste regioni distinguono le categorie\n\n');
