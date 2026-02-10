# Analisi Chemometrica — Mosti di Uva

## Dataset
- **File**: `mosti.mat` — 98 campioni × 6 variabili (aree % HPLC antociani)
- **Varietali**: Ancellotta (A), Montepulciano (M), Lambrusco Pugliese (LP), Sangiovese (S), Nero d'Avola (N)
- **Annate**: 2000, 2001
- **Variabili**: DPD%, CYD%, PTD%, PND%, MVD%, R lib/lrg

## Approccio Metodologico

### Preprocessing
- **Autoscaling** (mean centering + unit variance scaling)
- Motivazione: variabili con stessa unità (%) ma range molto diversi; autoscaling garantisce peso uguale a tutte le variabili nella PCA.

### PCA Esplorativa
- Modello con max 6 PC
- Score plot PC1 vs PC2 e PC1 vs PC3 per visualizzazione cluster
- Loading plot e biplot per interpretazione variabili
- Hotelling T² e Q residuals per diagnostica
- VIP-like da PCA per importanza variabili

### Train/Test Split
- **70/30** stratificato per classe
- Seed fisso (`rng(42)`) per riproducibilità

### SIMCA
- Modello PCA locale per ciascuna delle 5 classi
- N. PC per classe: selezionate via Leave-One-Out CV (min PRESS)
- Classificazione basata su distanze normalizzate Q + T²
- Coomans plot per distanze inter-classe

### PLS-DA
- Y codificata come dummy matrix (one-hot)
- CV: Venetian Blinds (10 splits) per selezione LV
- VIP calcolato per importanza variabili
- Metriche: Accuracy, Sensitivity, Specificity, R², RMSEC, RMSECV, RMSEP

## Esecuzione
```matlab
cd('/path/to/mosti')
run('analisi_mosti.m')
```

## Output
Tutti i grafici salvati automaticamente in `plots/` (formato PNG, 300 dpi).

## Requisiti
- MATLAB R2020b+
- PLS_Toolbox (Eigenvector Research)
- Statistics and Machine Learning Toolbox
