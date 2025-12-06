# GUIDA INTERFACCIA GRAFICA PLS TOOLBOX 9.5
## Analisi PLS per Predizione Contenuto Proteico del Grano da Spettri NIR

---

## 📌 INTRODUZIONE

Questa guida ti accompagnerà **passo-passo** nell'uso dell'**interfaccia grafica del PLS Toolbox 9.5** per MATLAB 2024 per completare tutti i requisiti dell'esercitazione.

### 🎯 Obiettivo:
Costruire un modello PLS per predire il contenuto proteico di chicchi di grano da spettri NIR (850-1050 nm)

### 📁 Dataset:
- **Training**: `wheat_ds.mat` (X, 415 campioni) + `Calibration_Y.mat` (Y)
- **Test**: `Validation_X.mat` (108 campioni) + `Validation_Y.mat`

---

## 🚀 FASE 1: PREPARAZIONE E CARICAMENTO DATI

### Passo 1.1: Verificare PLS Toolbox

1. Apri MATLAB 2024
2. Nel **Command Window**, digita:
   ```matlab
   plstoolbox
   ```
3. Si aprirà la finestra principale del PLS Toolbox con il menu
4. Verifica che sia la versione 9.5

### Passo 1.2: Caricare i dati nel Workspace

Nel **Command Window**, digita:

```matlab
% Caricare dati di calibrazione
load('wheat_ds.mat')           % Carica la variabile con gli spettri X
load('Calibration_Y.mat')      % Carica Y (contenuto proteico)

% Caricare dati di validazione
load('Validation_X.mat')       % Carica X test
load('Validation_Y.mat')       % Carica Y test

% Verificare i nomi delle variabili caricate
whos
```

**NOTA IMPORTANTE**: Devi identificare i **nomi esatti** delle variabili caricate. Potrebbero chiamarsi:
- `X`, `X_cal`, `Xcal`, `wheat_ds`, ecc.
- `Y`, `Y_cal`, `Ycal`, `Calibration_Y`, ecc.

**Scrivi su un foglio i nomi delle variabili** che vedi nel Workspace perché ti serviranno dopo!

### Passo 1.3: Creare Dataset Objects (opzionale ma consigliato)

Per semplificare, puoi rinominare le variabili in modo standard:

```matlab
% Rinominare per chiarezza (sostituisci con i nomi reali!)
X_cal = wheat_ds;              % SOSTITUISCI 'wheat_ds' con il nome reale
Y_cal = Calibration_Y;         % SOSTITUISCI con il nome reale
X_val = Validation_X;          % SOSTITUISCI con il nome reale
Y_val = Validation_Y;          % SOSTITUISCI con il nome reale

% Verificare dimensioni
size(X_cal)  % Dovrebbe essere 415 x 100
size(Y_cal)  % Dovrebbe essere 415 x 1
```

---

## 🔍 FASE 2: PCA ESPLORATIVA CON INTERFACCIA GRAFICA

### Passo 2.1: Aprire l'Analysis GUI

1. Nel **Command Window**, digita:
   ```matlab
   analysis
   ```
   
2. Si apre la finestra **Analysis GUI** (interfaccia principale del PLS Toolbox)

### Passo 2.2: Importare i dati in Analysis GUI

1. Nella finestra Analysis GUI, clicca sul pulsante **"Import Data"** (icona con cartella)

2. Si apre la finestra **"Import Data"**:
   - Seleziona **"From Workspace"**
   - In **"X-block"**: seleziona la variabile `X_cal` (o il nome che hai annotato)
   - Lascia vuoto Y per ora (faremo prima PCA esplorativa)
   - Clicca **"OK"**

3. I tuoi dati ora sono caricati nell'Analysis GUI

### Passo 2.3: Visualizzare gli spettri grezzi

1. Nella finestra Analysis GUI, nel menu in alto:
   - **Plots → Line Plot**

2. Si apre un grafico con tutti gli spettri sovrapposti
   - Osserva la forma generale degli spettri
   - Cerca anomalie o campioni molto diversi

### Passo 2.4: Applicare PREPROCESSING per PCA

Ora testiamo i diversi preprocessing richiesti dalla professoressa.

#### A) **NESSUN PREPROCESSING (Raw Data)**

1. Nel menu: **Preprocess → None** (o semplicemente non applicare nulla)

2. Eseguire PCA:
   - Menu: **Analysis → PCA**
   - Si apre la finestra **"PCA Options"**
   - **Number of PCs**: metti 10
   - **Preprocessing**: 
     - X-block: seleziona **"Mean Center"** (standard per PCA)
     - Cross-validation: seleziona **"Random Subsets"**, lascia 10 splits
   - Clicca **"OK"**

3. Si aprono automaticamente i grafici:
   - **Scores Plot** (T1 vs T2): Osserva cluster e outliers
   - **Variance Captured**: Quanta varianza spiegano le PC
   - **Loadings Plot**: Regioni spettrali importanti

4. **Salvare i risultati**:
   - Vai su **File → Save Results As**
   - Salva come `PCA_NoPreproc.mat`

5. **Esportare grafici importanti**:
   - Nei grafici aperti: **File → Export → Copy Figure** (oppure salva come immagine)
   - Salva: Scores Plot, Variance, Loadings PC1

#### B) **MSC (Multiplicative Scatter Correction)**

1. Ricomincia con dati freschi:
   - **File → New Analysis** (o chiudi e riapri `analysis`)
   - Importa di nuovo `X_cal`

2. Applicare MSC:
   - Menu: **Preprocess → Standard Pretreatments → MSC**
   - Si apre finestra MSC: lascia opzioni di default
   - Clicca **"Apply"**

3. Visualizza effetto:
   - **Plots → Line Plot**: vedi gli spettri dopo MSC (più uniformi)

4. Eseguire PCA con MSC:
   - Menu: **Analysis → PCA**
   - Number of PCs: 10
   - Preprocessing: **Mean Center** (MSC già applicato)
   - Cross-validation: Random Subsets, 10 splits
   - **OK**

5. **Analizzare i risultati**:
   - Scores Plot: ci sono cluster più definiti?
   - Hotelling T²: Menu **Plots → Diagnostics → Hotelling T²**
     - Identifica outliers sopra la linea rossa (95% confidence)
   - Variance Captured: È migliorata rispetto a Raw?

6. **Salvare**: `PCA_MSC.mat` e esporta grafici

#### C) **BASELINE CORRECTION**

1. **File → New Analysis**
2. Importa `X_cal`

3. Applicare Baseline:
   - Menu: **Preprocess → Standard Pretreatments → Baseline**
   - Opzioni:
     - Method: **"Weighted Least Squares"** o **"Polynomial"**
     - Se polynomial: Order = 2 o 3
   - **Apply**

4. Esegui PCA (stesso procedimento)
5. Salva risultati: `PCA_Baseline.mat`

#### D) **2nd DERIVATIVE (Savitzky-Golay)**

1. **File → New Analysis**
2. Importa `X_cal`

3. Applicare 2nd Derivative:
   - Menu: **Preprocess → Standard Pretreatments → Savitzky-Golay**
   - Opzioni:
     - **Derivative Order**: 2
     - **Window Size**: 15 (o 11-21, sperimenta)
     - **Polynomial Order**: 2 o 3
   - **Apply**

4. **ATTENZIONE**: La 2nd derivative amplifica il rumore!
   - Visualizza: **Plots → Line Plot** - spettri più "nervosi"

5. Esegui PCA
6. Salva: `PCA_2ndDeriv.mat`

#### E) **COMBINAZIONI**

**Baseline + MSC:**
1. **File → New Analysis**, importa `X_cal`
2. **Preprocess → Standard Pretreatments → Baseline** → Apply
3. **Preprocess → Standard Pretreatments → MSC** → Apply (si applica in sequenza)
4. Esegui PCA
5. Salva: `PCA_Baseline_MSC.mat`

**Baseline + 2nd Derivative:**
1. New Analysis, importa `X_cal`
2. Baseline → Apply
3. Savitzky-Golay (deriv 2) → Apply
4. PCA
5. Salva: `PCA_Baseline_2ndDeriv.mat`

**2nd Derivative + MSC:**
1. New Analysis, importa `X_cal`
2. Savitzky-Golay (deriv 2) → Apply
3. MSC → Apply
4. PCA
5. Salva: `PCA_2ndDeriv_MSC.mat`

### Passo 2.5: CONFRONTARE I PREPROCESSING

Ora devi decidere qual è il migliore. Confronta:

1. **Varianza spiegata** dalle prime 2-3 PC:
   - Apri ogni file salvato: `load('PCA_MSC.mat')`
   - Guarda il grafico "Variance Captured"
   - Il migliore mantiene più varianza con meno PC

2. **Separazione nei Scores**:
   - Ci sono cluster più definiti con un preprocessing?
   - Gli outliers sono ridotti?

3. **Loadings interpretabili**:
   - I loadings PC1 mostrano picchi nelle regioni N-H (~variabile 50-80)?

**ANNOTATI quale preprocessing sembra migliore** (es. MSC) - lo userai per il PLS!

---

## 📊 FASE 3: MODELLO PLS CON INTERFACCIA GRAFICA

### Passo 3.1: Avviare nuova analisi PLS

1. **File → New Analysis** (o chiudi e riapri `analysis`)

2. **Import Data**:
   - **From Workspace**
   - **X-block**: `X_cal`
   - **Y-block**: `Y_cal` ← **IMPORTANTE: ora includi anche Y!**
   - **OK**

### Passo 3.2: Applicare il MIGLIOR PREPROCESSING

Applica il preprocessing che ha dato i risultati migliori in PCA (es. MSC):

```
Preprocess → Standard Pretreatments → MSC → Apply
```

Verifica con Line Plot che sia stato applicato.

### Passo 3.3: Costruire il modello PLS

1. Menu: **Analysis → PLS** (o **PLSR** = PLS Regression)

2. Si apre **"PLS Options"**:
   
   **Numero di componenti:**
   - **Number of LVs**: 20 (testiamo fino a 20 componenti)
   
   **Preprocessing:**
   - **X-block Preprocessing**: 
     - Seleziona **"Mean Center"** (MSC già applicato prima)
   - **Y-block Preprocessing**: 
     - Seleziona **"Mean Center"**
   
   **Cross-Validation:**
   - **Cross-validation**: **"Venetian Blinds"** (raccomandato)
   - **Number of splits**: 10
   - Oppure **"Random Subsets"** con 10 splits
   - Oppure **"Leave One Out"** (più lento ma preciso)
   
   ⚠️ **NON selezionare "None" per cross-validation!**
   
3. Clicca **"OK"**

4. Il toolbox calcola il modello... Attendere (può richiedere 1-2 minuti)

### Passo 3.4: Analizzare i risultati PLS automatici

Si aprono automaticamente diversi grafici:

#### A) **RMSECV Plot** (Root Mean Square Error of Cross-Validation)

- Asse X: Numero di LVs
- Asse Y: Errore RMSECV
- **Cercare il MINIMO**: Il numero ottimale di LVs è dove RMSECV è minimo
- Annotati questo numero (es. LV = 7)

#### B) **Variance Captured**

- Mostra quanta varianza in X e Y è spiegata da ogni LV
- Dovrebbe aumentare con il numero di LVs

#### C) **Predicted vs Measured (Calibration)**

- Asse X: Y misurato (vero contenuto proteico)
- Asse Y: Y predetto dal modello
- Punti dovrebbero stare sulla linea diagonale
- R² mostrato nel titolo

#### D) **Predicted vs Measured (Cross-Validation)**

- Come sopra, ma per le predizioni CV
- R²cv deve essere vicino a R²cal (se molto più basso = overfitting)

### Passo 3.5: Scegliere il numero OTTIMALE di LVs

1. Nel grafico RMSECV, identifica il minimo

2. **IMPORTANTE**: Se vuoi ricostruire il modello con solo le LVs ottimali:
   - **File → New Analysis**
   - Reimporta dati e preprocessing
   - **Analysis → PLS**
   - **Number of LVs**: metti il numero ottimale trovato (es. 7)
   - **OK**

3. **Salvare il modello**:
   - **File → Save Results As**
   - Salva come `PLS_Model_Final.mat`

---

## 🔬 FASE 4: DIAGNOSTICA DEL MODELLO

### Passo 4.1: Inner Relations (T vs U)

1. Con il modello PLS aperto, menu:
   - **Plots → Scores → PLS Inner Relation Plot**

2. Si aprono grafici per ogni LV: T1 vs U1, T2 vs U2, ecc.

3. **Cosa cercare**:
   - Relazioni **lineari** e forti (punti vicini alla linea)
   - Se la relazione è debole/scattered → quella LV non è utile

4. Se hai il file `plotinrel.m` nel workspace:
   ```matlab
   load('PLS_Model_Final.mat')
   plotinrel(model)
   ```

5. **Esportare grafici**: File → Export → Save As Image

### Passo 4.2: Residui

1. Menu: **Plots → Diagnostics → Residuals Plot**

2. **Residuals vs Predicted**:
   - Punti dovrebbero essere sparsi casualmente intorno allo zero
   - NO pattern sistematici (curve, trend)

3. **Histogram dei Residui**:
   - Menu: **Plots → Diagnostics → Histogram of Residuals**
   - Dovrebbe essere approssimativamente una distribuzione normale (campana)

4. Esporta grafici

### Passo 4.3: Leverage e Outliers (Hotelling T²)

1. Menu: **Plots → Diagnostics → Hotelling T²**

2. Grafico con:
   - Asse X: Sample number
   - Asse Y: T² statistic
   - Linea rossa: 95% confidence limit

3. **Identificare outliers**:
   - Samples **sopra la linea rossa** hanno alto leverage (influenti/anomali)
   - Annotati i numeri dei samples outliers

4. **Q-Residuals (SPE)**:
   - Menu: **Plots → Diagnostics → Q-Residuals**
   - Mostra samples che non sono ben descritti dal modello

5. Esporta grafici

### Passo 4.4: Y Measured vs Y Fitted (già visto ma rivisitalo)

1. Menu: **Plots → Predicted vs Measured**

2. Guarda entrambi:
   - **Calibration**: R² alto (es. >0.95)
   - **Cross-Validation**: R² leggermente più basso ma simile (es. >0.90)

3. Se R²cv << R²cal → OVERFITTING! Riduci numero LVs

4. Annota i valori:
   - R² Calibration: _____
   - R² Cross-Validation: _____
   - RMSEC: _____
   - RMSECV: _____

---

## 🔍 FASE 5: INTERPRETAZIONE - REGIONI SPETTRALI IMPORTANTI

### Passo 5.1: PLS Weights

1. Menu: **Plots → Loadings → PLS Weights**

2. Si apre grafico dei weights per ogni LV:
   - Asse X: Numero variabile (lunghezza d'onda)
   - Asse Y: Weight

3. **Interpretazione**:
   - **Picchi positivi/negativi** = variabili importanti per quella LV
   - Per LV1 (il più importante): dove sono i picchi maggiori?
   - Confronta con il grafico fornito dalla prof: sono nelle regioni N-H (~1000-1030 nm)?

4. Esporta il grafico Weight LV1 e LV2

### Passo 5.2: Regression Coefficients

1. Menu: **Plots → Loadings → Regression Coefficients**

2. Grafico con:
   - Asse X: Variabile (lunghezza d'onda)
   - Asse Y: Coefficiente di regressione β

3. **Interpretazione**:
   - **β positivo alto**: Aumento assorbanza → aumento proteine
   - **β negativo alto**: Aumento assorbanza → diminuzione proteine
   - Le regioni con |β| grande sono le più importanti per predire Y

4. **Confrontare con la teoria**:
   - Proteine hanno banda N-H ~1000-1030 nm (variabili ~50-80)
   - I tuoi coefficienti hanno picchi lì?

5. Esporta grafico

### Passo 5.3: VIP (Variable Importance in Projection)

1. Menu: **Plots → Loadings → VIP**

2. Grafico VIP:
   - Asse X: Variabile
   - Asse Y: VIP score
   - Linea orizzontale a VIP = 1 (threshold)

3. **Interpretazione**:
   - **VIP > 1**: Variabile importante
   - **VIP < 0.8**: Variabile poco rilevante

4. **Identificare regioni importanti**:
   - Quali variabili (range) hanno VIP > 1?
   - Annotale: "Variabili importanti: da X a Y"

5. Esporta grafico

### Passo 5.4: Selectivity Ratio (opzionale)

1. Menu: **Plots → Loadings → Selectivity Ratio**

2. Mostra il rapporto segnale/rumore per ogni variabile

3. Variabili con alto selectivity ratio sono più affidabili

---

## 🎯 FASE 6: PREDIZIONE SUL TEST SET

### Passo 6.1: Applicare il modello al Test Set

**METODO 1: Usando la GUI**

1. Con il modello PLS aperto, menu:
   - **Predict → New Predictions**

2. Si apre finestra **"Apply Model"**:
   - **Select new X-block**: Dal workspace, seleziona `X_val`
   - **Apply same preprocessing**: ✅ ASSICURATI CHE SIA SPUNTATO!
   - **OK**

3. Il toolbox calcola le predizioni Y_pred

4. Si aprono automaticamente grafici:
   - **Predicted vs Measured** (se fornisci anche Y_val)

**METODO 2: Command line (se GUI non funziona)**

```matlab
% Caricare il modello salvato
load('PLS_Model_Final.mat')

% Applicare stesso preprocessing al test set
% (il toolbox dovrebbe farlo automaticamente, ma verifica)

% Predire
Y_pred_val = predict(model, X_val);

% Se vuoi vedere specifiche LVs:
optimal_lv = 7;  % IL TUO NUMERO OTTIMALE
Y_pred_val = predict(model, X_val, optimal_lv);
```

### Passo 6.2: Calcolare statistiche di predizione

Se usi Command line:

```matlab
% Calcolare RMSEP (Root Mean Square Error of Prediction)
rmse_pred = sqrt(mean((Y_val - Y_pred_val).^2));

% Calcolare R² di predizione
SS_res = sum((Y_val - Y_pred_val).^2);
SS_tot = sum((Y_val - mean(Y_val)).^2);
r2_pred = 1 - (SS_res / SS_tot);

fprintf('RMSEP: %.4f\n', rmse_pred);
fprintf('R² Prediction: %.4f\n', r2_pred);
```

### Passo 6.3: Plot Y Measured vs Y Predicted (Test Set)

**Con GUI (se hai dato Y_val in "Apply Model")**:
- Il grafico si apre automaticamente

**Con Command line**:

```matlab
figure;
plot(Y_val, Y_pred_val, 'o', 'MarkerSize', 8, 'LineWidth', 1.5)
hold on
plot([min(Y_val) max(Y_val)], [min(Y_val) max(Y_val)], 'r--', 'LineWidth', 2)
xlabel('Y Misurato (Contenuto Proteico) - Test Set')
ylabel('Y Predetto')
title(sprintf('Test Set: R^2 = %.3f, RMSEP = %.3f', r2_pred, rmse_pred))
grid on
axis equal

% Aggiungere bande ±1 RMSEP
plot([min(Y_val) max(Y_val)], [min(Y_val)+rmse_pred max(Y_val)+rmse_pred], 'g--')
plot([min(Y_val) max(Y_val)], [min(Y_val)-rmse_pred max(Y_val)-rmse_pred], 'g--')
legend('Predizioni', 'Linea Ideale', '±1 RMSEP')
```

### Passo 6.4: Residui del Test Set

```matlab
% Residui
residuals_test = Y_val - Y_pred_val;

% Plot residui vs Y predetto
figure;
subplot(2,1,1)
plot(Y_pred_val, residuals_test, 'o', 'MarkerSize', 8)
xlabel('Y Predetto - Test Set')
ylabel('Residui')
title('Residui di Predizione vs Y Predetto')
hold on
plot([min(Y_pred_val) max(Y_pred_val)], [0 0], 'r--', 'LineWidth', 2)
grid on

% Istogramma residui
subplot(2,1,2)
histogram(residuals_test, 15)
xlabel('Residui')
ylabel('Frequenza')
title('Distribuzione Residui - Test Set')
```

### Passo 6.5: Confronto Calibrazione vs Cross-Validation vs Test

```matlab
% Creare confronto grafico
figure;

subplot(1,3,1)
plot(Y_cal, model.pred.yhat{1,1}(:,optimal_lv), 'bo')
hold on
plot([min(Y_cal) max(Y_cal)], [min(Y_cal) max(Y_cal)], 'r--')
title('Calibrazione')
xlabel('Y Misurato')
ylabel('Y Predetto')
axis equal
grid on

subplot(1,3,2)
plot(Y_cal, model.pred.cvpred{1,1}(:,optimal_lv), 'go')
hold on
plot([min(Y_cal) max(Y_cal)], [min(Y_cal) max(Y_cal)], 'r--')
title('Cross-Validation')
xlabel('Y Misurato')
ylabel('Y Predetto CV')
axis equal
grid on

subplot(1,3,3)
plot(Y_val, Y_pred_val, 'ro')
hold on
plot([min(Y_val) max(Y_val)], [min(Y_val) max(Y_val)], 'r--')
title('Test Set')
xlabel('Y Misurato')
ylabel('Y Predetto')
axis equal
grid on
```

---

## ✂️ FASE 7: SELEZIONE REGIONI SPETTRALI (OPZIONALE)

### Passo 7.1: Identificare variabili importanti da VIP

Dal grafico VIP che hai esportato:

1. Identifica quali variabili hanno VIP > 1
2. Annota i range (es. variabili 20-35, 50-80)

### Passo 7.2: Creare subset di variabili

```matlab
% Esempio: selezionare variabili con VIP > 1
load('PLS_Model_Final.mat')
vip_scores = model.vip(:, optimal_lv);  % VIP per il modello con LV ottimali

% Trovare indici con VIP > 1
selected_vars = find(vip_scores > 1);

fprintf('Variabili selezionate: %d su %d\n', length(selected_vars), length(vip_scores));

% Creare nuovi dataset con solo variabili selezionate
X_cal_selected = X_cal(:, selected_vars);
X_val_selected = X_val(:, selected_vars);
```

### Passo 7.3: Ricostruire modello con variabili selezionate

1. **File → New Analysis**

2. **Import Data**:
   - X: Usa `X_cal_selected` (dal workspace)
   - Y: `Y_cal`

3. Applica stesso preprocessing (MSC)

4. **Analysis → PLS**:
   - Number of LVs: stesso numero ottimale trovato prima
   - Stesse opzioni
   - **OK**

5. Confronta i risultati:
   - RMSECV è migliorato?
   - R² è migliorato?

6. Predici test set con variabili selezionate

7. Confronta RMSEP: è meglio o peggio?

---

## 📝 FASE 8: PREPARARE IL REPORT

### Checklist grafici da includere:

**PCA Esplorativa:**
- [ ] Scores Plot (T1 vs T2) - miglior preprocessing
- [ ] Loadings PC1 - regioni spettrali importanti
- [ ] Hotelling T² - outliers
- [ ] Variance Captured

**Modello PLS:**
- [ ] RMSECV vs Number of LVs (scelta LVs ottimali)
- [ ] Inner Relations (T vs U) - almeno LV1 e LV2
- [ ] Residui di calibrazione vs Y predicted
- [ ] Hotelling T² (leverage)
- [ ] Y measured vs Y fitted (Calibration)
- [ ] Y measured vs Y CV (Cross-Validation)

**Interpretazione:**
- [ ] PLS Weights (LV1, LV2)
- [ ] Regression Coefficients
- [ ] VIP scores

**Test Set:**
- [ ] Y measured vs Y predicted (Test Set)
- [ ] Residui test set

### Tabella riassuntiva risultati:

| Metrica | Calibrazione | Cross-Validation | Test Set |
|---------|--------------|------------------|----------|
| R² | _____ | _____ | _____ |
| RMSE | _____ | _____ | _____ |

### Conclusioni da scrivere:

1. **Preprocessing scelto**: MSC (o altro) - Perché?
   - Migliore separazione in PCA
   - Riduzione outliers
   - Varianza spiegata

2. **Numero LVs ottimale**: ___ (es. 7)
   - Minimo RMSECV
   - Buon compromesso complessità/performance

3. **Performance modello**:
   - R²cal vs R²cv vs R²test (sono simili? Buon segno!)
   - RMSECV vs RMSEP (sono simili? Nessun drift temporale!)

4. **Regioni spettrali importanti**:
   - Da VIP, Weights, Regression Coeff
   - Corrispondono a N-H (~1000-1030 nm) per proteine? ✅

5. **Capacità predittiva**:
   - Il modello predice bene il test set?
   - Outliers nel test set?

---

## 🔧 TIPS E TROUBLESHOOTING

### Problema: "Cannot find Analysis GUI"

```matlab
% Verificare path
which analysis

% Se non trovato, aggiungere path PLS Toolbox
addpath(genpath('C:\Program Files\MATLAB\PLS_Toolbox'))  % Windows
addpath(genpath('/Applications/MATLAB/PLS_Toolbox'))     % Mac

% Salvare path
savepath
```

### Problema: Preprocessing non si applica

- Assicurati di cliccare **"Apply"** dopo aver selezionato il preprocessing
- Verifica con **Line Plot** che gli spettri siano cambiati

### Problema: Cross-validation molto lenta

- Usa "Venetian Blinds" invece di "Leave One Out"
- Riduci numero di splits a 5-7

### Problema: VIP non disponibile

VIP potrebbe non essere calcolato automaticamente. Calcolo manuale:

```matlab
% Con il modello caricato
load('PLS_Model_Final.mat')

% Accedere ai componenti del modello
W = model.loads{2,1};  % Weights
T = model.scores{1,1}; % Scores X
q = model.loads{2,2};  % Loadings Y

% Calcolare SSY (sum of squares spiegata da ogni LV)
num_lv = optimal_lv;
SSY = zeros(num_lv, 1);
for lv = 1:num_lv
    y_contrib = T(:,lv) * q(lv);
    SSY(lv) = sum(y_contrib.^2);
end

% Calcolare VIP
p = size(W, 1);  % Numero variabili
vip = sqrt(p * sum((W(:,1:num_lv).^2) .* SSY', 2) / sum(SSY));

% Plot VIP
figure;
plot(vip, 'LineWidth', 2)
hold on
plot([1 length(vip)], [1 1], 'r--', 'LineWidth', 2)
xlabel('Numero Variabile (Lunghezza d''onda)')
ylabel('VIP Score')
title('Variable Importance in Projection')
legend('VIP', 'Threshold = 1')
grid on
```

### Accedere ai risultati del modello (per command line)

```matlab
load('PLS_Model_Final.mat')

% Struttura del modello PLS Toolbox:
model.scores{1,1}        % Scores T (X-block)
model.scores{1,2}        % Scores U (Y-block)
model.loads{1,1}         % Loadings P di X
model.loads{2,1}         % Weights W
model.loads{2,2}         % Loadings q di Y
model.reg.B              % Regression coefficients
model.pred.yhat{1,1}     % Y predicted (calibrazione)
model.pred.cvpred{1,1}   % Y predicted (cross-validation)
model.detail.ssq         % Sum of squares
model.detail.rmsec       % RMSE calibration
model.detail.rmsecv      % RMSE cross-validation
```

---

## 🎓 INTERPRETAZIONE SCIENTIFICA

### Regioni spettrali NIR del grano (850-1050 nm):

Dal grafico fornito dalla professoressa:

- **~900-950 nm**: C-H (fats - grassi) - 3° overtone stretching
- **~1000-1040 nm**: O-H (carbohydrates, water - carboidrati, acqua) - 2° overtone stretching
- **~1000-1030 nm**: N-H (proteins - proteine) - 2° overtone stretching

**Aspettative per il contenuto proteico:**

- I tuoi **Weights/VIP/Regression Coeff** dovrebbero avere picchi importanti nella regione **N-H (~1000-1030 nm)**
- Possibili contributi negativi/positivi da O-H (acqua correlata inversamente con proteine in alcuni casi)
- C-H meno rilevante per proteine (più per grassi)

**Nel tuo report, discuti**:
- Le regioni identificate dal modello corrispondono alla teoria?
- Ci sono picchi inaspettati? Perché?
- La banda N-H è la più importante? (Dovrebbe esserlo!)

---

## ✅ RIEPILOGO WORKFLOW COMPLETO

1. ✅ Caricare dati in MATLAB workspace
2. ✅ Aprire `analysis` GUI
3. ✅ PCA esplorativa con tutti i preprocessing richiesti
4. ✅ Scegliere miglior preprocessing (es. MSC)
5. ✅ Costruire modello PLS con cross-validation
6. ✅ Determinare numero ottimale di LVs (minimo RMSECV)
7. ✅ Analizzare diagnostica (inner relations, residui, leverage)
8. ✅ Interpretare regioni spettrali (Weights, Regression Coeff, VIP)
9. ✅ Predire test set
10. ✅ Confrontare performance (Cal vs CV vs Test)
11. ✅ (Opzionale) Selezione variabili importanti
12. ✅ Esportare tutti i grafici per il report

---

## 📚 RIFERIMENTI RAPIDI MENU GUI

**Analysis GUI - Menu Structure:**

```
File
  ├── New Analysis
  ├── Open
  ├── Save Results As
  └── Export

Preprocess
  ├── None
  └── Standard Pretreatments
      ├── MSC
      ├── Baseline
      ├── Savitzky-Golay
      ├── SNV (Standard Normal Variate)
      └── Normalize

Analysis
  ├── PCA
  ├── PLS (PLSR)
  └── Other methods...

Plots
  ├── Line Plot
  ├── Scores
  │   ├── Scores Plot
  │   └── PLS Inner Relation Plot
  ├── Loadings
  │   ├── Loadings Plot
  │   ├── PLS Weights
  │   ├── Regression Coefficients
  │   ├── VIP
  │   └── Selectivity Ratio
  ├── Diagnostics
  │   ├── Residuals Plot
  │   ├── Histogram of Residuals
  │   ├── Hotelling T²
  │   └── Q-Residuals (SPE)
  └── Predicted vs Measured

Predict
  └── New Predictions
```

---

## 🚀 PRONTO PER INIZIARE!

Hai tutto quello che ti serve! Segui questa guida passo-passo e crea tutti i grafici richiesti usando l'interfaccia grafica.

**Buon lavoro con l'esercitazione! 🎉**
