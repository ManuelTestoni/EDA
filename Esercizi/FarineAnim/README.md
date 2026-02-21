# Farine Animali NIR - Analisi Completa

Questa cartella contiene l'analisi completa degli spettri NIR di farine animali (pollo, bovino, pesce) tramite due assignment distinti.

---

## Struttura della cartella

```
FarineAnim/
├── farineanimNIR.mat              # Dati grezzi (spettri + categorie)
├── animal_feedNIR.mat             # Dataset in formato PLS Toolbox
├── assexscale.txt                 # Wavenumbers (6000-4000 cm⁻¹)
│
├── ── ASSIGNMENT 1: PRE-PROCESSING & PCA ──
├── analisi_farine_NIR_PLSToolbox.m   # Script preprocessing + PCA esplorativa
├── results_preprocessing_NIR.mat     # Risultati preprocessing
├── plots/                            # Grafici del preprocessing
│   ├── none/                         # Solo mean centering
│   ├── baseline/                     # Baseline correction
│   ├── snv/                          # Standard Normal Variate
│   ├── msc/                          # Multiplicative Scatter Correction
│   ├── normalize/                    # Normalizzazione
│   ├── 1-derivate/                   # Prima derivata (Savitzky-Golay)
│   └── 2-derivate/                   # Seconda derivata (Savitzky-Golay)
│
├── ── ASSIGNMENT 2: PLS REGRESSION ──
├── analisi_PLS_regression.m          # Script regressione PLS / PLS-DA
├── genera_report_PLS.py              # Generatore PDF professionale
├── pls_plots/                        # Grafici della regressione PLS
│   ├── 01_PCA_scores_PC1_PC2.png
│   ├── 02_PCA_loadings.png
│   ├── 03_PCA_T2_vs_Q.png
│   ├── 04_spettri_originali.png
│   ├── 05_confronto_preprocessing_RMSECV.png
│   ├── 06_scree_plot_LV.png
│   ├── 07_T2_vs_Q_PLS.png
│   ├── 08_scores_LV1_LV2.png
│   ├── 09_xx_Y_pred_vs_meas_[classe].png
│   ├── 10_Y_vs_CV_residuals.png
│   ├── 11_leverage_vs_residuals.png
│   ├── 12_inner_relations.png
│   ├── 13_PLS_weights.png
│   ├── 14_regression_coefficients.png
│   ├── 15_VIP_scores.png
│   ├── 16_selectivity_ratio.png
│   ├── 17_test_set_prediction.png
│   ├── 18_confusion_matrix.png
│   ├── 19_spettri_preprocessati.png
│   ├── results_PLS.mat               # Risultati modello salvati
│   └── report_PLS_regression.pdf     # Report PDF generato da Python
│
├── ── UTILITÀ ──
├── mypca.m                           # Funzione PCA custom
├── mypca_prep.m                      # Funzione preprocessing custom
├── pca_stat.m                        # PCA con Statistics Toolbox
└── importa_dati.m                    # Importazione dati per PLS Toolbox
```

---

## Assignment 1: Pre-Processing & PCA Esplorativa

**File:** `analisi_farine_NIR_PLSToolbox.m`

**Obiettivo:** Valutare diversi preprocessing spettrali e identificare il migliore per differenziare le 3 categorie di farine animali (pollo, bovino, pesce) tramite PCA.

**Preprocessing testati:**
1. Nessun preprocessing (solo mean centering)
2. Baseline correction (detrend)
3. Standard Normal Variate (SNV)
4. Multiplicative Scatter Correction (MSC)
5. Normalizzazione (norma unitaria)
6. 1ª Derivata (Savitzky-Golay, finestra=31, ordine=3)
7. 2ª Derivata (Savitzky-Golay, finestra=31, ordine=3)

**Output:** Per ogni preprocessing vengono generati:
- Score plots (PC1 vs PC2, PC1 vs PC3, PC2 vs PC3)
- Loading plots (ogni PC + overlay)
- Spettri medi preprocessati per categoria
- Scree plot della varianza spiegata

**Grafici:** Salvati nella cartella `plots/`

---

## Assignment 2: PLS Regression / PLS-DA

**File:** `analisi_PLS_regression.m`

**Obiettivo:** Costruire un modello di regressione PLS (PLS-DA, data la natura categorica della variabile risposta) per la classificazione delle farine animali, seguendo la metodologia descritta in `Assignment_Collega.pdf`.

**Workflow:**

| Step | Descrizione |
|------|-------------|
| 1 | **Caricamento dati** e divisione calibrazione/test (70/30 stratificata) |
| 2 | **PCA esplorativa** con mean centering (2 PCs) per outlier detection |
| 3 | **Confronto preprocessing** con 8 combinazioni diverse (RMSECV come criterio) |
| 4 | **Scelta numero LV** tramite scree plot RMSEC/RMSECV |
| 5 | **Modello PLS finale** con preprocessing ottimale e CV Venetian Blind (15 split, thickness 3) |
| 6 | **Grafici diagnostici** — T² vs Q, Leverage vs Residuals, Y pred vs meas, Inner Relations |
| 7 | **Importanza variabili** — Weights, Regression Coefficients, VIP, Selectivity Ratio |
| 8 | **Validazione test set** — RMSEP, R², Confusion Matrix |

**Preprocessing testati (ordine):**
1. Mean Centering
2. SNV + MC
3. MSC + MC
4. 1ª Derivata (SG w=31, p=2) + MC
5. 2ª Derivata (SG w=31, p=2) + MC
6. 2ª Derivata + MSC + MC (consigliato dal collega)
7. 1ª Derivata + SNV + MC
8. Detrend + MC

**Funzioni PLS Toolbox utilizzate (non GUI):**
- `pls()` — modello PLS con cross-validation
- `pca()` — PCA esplorativa
- `preprocess()` — definizione catene di preprocessing
- `vip()` — Variable Importance in Projection
- `selectratio()` — Selectivity Ratio
- `dataset()` — creazione dataset PLS Toolbox

**Grafici:** Salvati nella cartella `pls_plots/`

---

## Generazione Report PDF

**File:** `genera_report_PLS.py`

**Dipendenze Python:**
```bash
pip install reportlab Pillow
```

**Uso:**
```bash
python genera_report_PLS.py
```

Genera automaticamente un report PDF professionale (`pls_plots/report_PLS_regression.pdf`) che include tutti i grafici con spiegazioni dettagliate per ognuno.

---

## Dati

| Variabile | Dimensioni | Descrizione |
|-----------|-----------|-------------|
| `farineanimNIRdata` | 84 × 2001 | Spettri NIR (righe = campioni, colonne = wavenumbers) |
| `category` | 84 × 1 | Indice di classe (pollo, bovino, pesce) |
| `assexscale` | 1 × 2001 | Wavenumbers (6000–4000 cm⁻¹) |

---

## Requisiti

- **MATLAB** con PLS Toolbox (Eigenvector Research Inc.)
- **Python 3.7+** con `reportlab` e `Pillow` per la generazione del PDF
