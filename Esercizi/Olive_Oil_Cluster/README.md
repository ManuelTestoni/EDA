# 🫒 Olive Oil Clustering Analysis

Analisi completa di clustering sul dataset **Olive Oil** utilizzando metodi gerarchici e non gerarchici in MATLAB e Python.

---

## 📋 **Indice**

1. [Dataset](#-dataset)
2. [File del Progetto](#-file-del-progetto)
3. [Requisiti](#-requisiti)
4. [Guida Rapida](#-guida-rapida)
5. [Script Dettagliati](#-script-dettagliati)
6. [Parametri Configurabili](#-parametri-configurabili)
7. [Output e Interpretazione](#-output-e-interpretazione)
8. [Troubleshooting](#-troubleshooting)
9. [Note Tecniche](#-note-tecniche)

---

## 📊 **Dataset**

### **Olive Oil Dataset**
- **Campioni**: 572 campioni di olio d'oliva
- **Variabili**: 7 caratteristiche chimiche
- **Categorie**: 8 varietà/regioni di provenienza
- **Formato**: Disponibile in `.csv`, `.xlsx`, `.mat`

### **Variabili Misurate**
Le 7 caratteristiche chimiche (standardizzate) che descrivono ogni campione di olio.

### **File Disponibili**
- `oliveoil.csv` - Formato CSV (solo dati numerici)
- `olivedata.xlsx` - Formato Excel (con intestazioni e possibili etichette)
- `oliveoil.mat` - Formato MATLAB (variabile `olivdata`)

---

## 📁 **File del Progetto**

### **Script MATLAB**

| File | Descrizione | Figure Generate | Tempo Esecuzione |
|------|-------------|-----------------|------------------|
| `clustering_fun.m` | **Analisi gerarchica iterativa**<br>Testa ~15 combinazioni linkage/distanza<br>Analizza top 2 metodi in dettaglio | ~23 figure | ~30-60 sec |
| `kmeans_etal.m` | **K-means, DBSCAN, OPTICS**<br>Metodi non gerarchici con diagnostiche | ~7 figure | ~10-20 sec |
| `optics.m` | Implementazione algoritmo OPTICS | - | (Funzione) |

### **Script Python**

| File | Descrizione | Figure Generate | Tempo Esecuzione |
|------|-------------|-----------------|------------------|
| `analisi_completa_oliveoil.py` | **Analisi completa con scikit-learn**<br>Equivalente MATLAB in Python | ~25 figure PNG | ~15-30 sec |

### **Dati**
- `oliveoil.csv` - Dataset formato CSV
- `olivedata.xlsx` - Dataset formato Excel
- `oliveoil.mat` - Dataset formato MATLAB

### **Documentazione**
- `README.md` - Questo file
- `ClusterPLStoolbox.pdf` - Documentazione PLS Toolbox
- `Ese_cluster.docx` - Esercizi originali

---

## 🔧 **Requisiti**

### **MATLAB**
```matlab
MATLAB R2020a o superiore
Toolbox richiesti:
  ✓ Statistics and Machine Learning Toolbox (OBBLIGATORIO)
  ○ PLS Toolbox (OPZIONALE - solo per DBSCAN)
```

**Funzioni critiche utilizzate:**
- `normalize()`, `linkage()`, `cluster()`, `dendrogram()`
- `kmeans()`, `silhouette()`, `pdist()`, `gscatter()`
- `svds()` per PCA

### **Python**
```bash
Python 3.8+
Librerie richieste:
  numpy>=1.20.0
  pandas>=1.3.0
  matplotlib>=3.4.0
  seaborn>=0.11.0
  scikit-learn>=1.0.0
  scipy>=1.7.0
```

**Installazione:**
```bash
cd Olive_Oil_Cluster
python -m venv venv
source venv/bin/activate  # Su macOS/Linux
# venv\Scripts\activate   # Su Windows
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

---

## 🚀 **Guida Rapida**

### **MATLAB**

#### **1. Analisi Gerarchica Completa**
```matlab
% Testa tutte le combinazioni e analizza i migliori 2 metodi
run('clustering_fun.m')
```

**Output:**
- ~15 dendrogrammi (tutte combinazioni)
- Tabella ranking con silhouette scores
- 6 figure dettagliate (top 2 metodi)
- Clustering variabili + heatmap ordinata

#### **2. K-means, DBSCAN, OPTICS**
```matlab
% Metodi non gerarchici con diagnostiche
run('kmeans_etal.m')
```

**Output:**
- K-means con centroidi e silhouette
- DBSCAN con k-distance plot diagnostico
- OPTICS con reachability plot

### **Python**

```bash
# Attiva virtual environment
source venv/bin/activate

# Esegui analisi completa
python analisi_completa_oliveoil.py
```

**Output:**
- ~25 figure salvate come PNG numerati
- Report testuale nella console
- Confronto tutti i metodi

---

## 📖 **Script Dettagliati**

### **clustering_fun.m - Analisi Gerarchica**

#### **Struttura in 3 Fasi**

##### **FASE 1: Test Iterativo (~15 combinazioni)**
```matlab
Linkage:  ward, single, complete, average, centroid
Distanze: euclidean, cityblock, correlation, cosine
```

**Per ogni combinazione:**
1. Calcola linkage matrix
2. Estrai `nClusters` cluster
3. Calcola silhouette score
4. Genera dendrogramma colorato
5. Salva risultati

**Output Fase 1:**
```
=== CLUSTERING GERARCHICO ITERATIVO ===
Testando combinazioni linkage × distanza:
Linkage      Distanza     Silhouette   Note
------------------------------------------------------------
ward         euclidean    0.4523       ✓
complete     euclidean    0.4401       ✓
...

RANKING:
1. ward + euclidean: 0.4523
2. complete + euclidean: 0.4401
```

##### **FASE 2: Analisi Dettagliata Top 2**

Per ciascuno dei 2 migliori metodi genera:
1. **Dendrogramma con cutoff** colorato
2. **Scatter PC1 vs PC2** con cluster
3. **Silhouette plot** dettagliato

##### **FASE 3: Clustering Variabili**

Usando il **miglior metodo** per campioni:
1. Clustering variabili con correlation distance
2. Figura 2×2 combinata:
   - Info box (metodo migliore)
   - Sample dendrogram
   - Variable dendrogram
   - Heatmap ordinata

---

### **kmeans_etal.m - Metodi Non Gerarchici**

#### **K-means**
```matlab
[idx,C] = kmeans(data, k_clusters, 'Replicates', 10);
```
- **Replicates**: 10 esecuzioni, prende la migliore
- **Output**: Cluster assignments + centroidi
- **Figure**: Scatter PC1/PC2 + silhouette plot

#### **DBSCAN (PLS Toolbox)**
```matlab
[cls,epsdist] = dbscan(data, minpts);
```
- **Correzione automatica**: Se trova 1 cluster, prova eps più piccoli
- **K-distance plot**: Diagnostica per scegliere eps ottimale
- **Noise detection**: Punti classificati come rumore (cls=0)

**Interpretazione K-distance plot:**
```
    |
  d |     /-------- Plateau = rumore
  i |    /
  s |   /
  t |  /____  <-- "GOMITO" = eps ottimale!
  a | /    ----____
    |/              ------_____
    +-------------------------->
```

#### **OPTICS**
```matlab
[RD, CD, order] = optics(data, k_optics);
```
- **k parameter**: Numero vicini (default: n_campioni/25)
- **Reachability plot**: Valli = cluster densi, Picchi = separazioni
- **Scatter colorato**: Reachability distance come colore

---

### **analisi_completa_oliveoil.py - Python**

Equivalente MATLAB completo in Python/scikit-learn:

```python
# Struttura
1. Caricamento dati + StandardScaler
2. PCA per visualizzazione (2 componenti)
3. Loop 12 combinazioni gerarchiche:
   - euclidean/minkowski/mahalanobis
   - × single/average/complete/centroid
4. Figure per miglior metodo (3 figure)
5. K-means (2 figure)
6. DBSCAN (2 figure)
7. OPTICS (2 figure)
8. Confronto finale (1 figura 2×2)
9. Report sommario
```

**Figure salvate automaticamente:**
```
fig_01_single_euclidean_dendrogram.png
fig_02_single_euclidean_scatter.png
...
fig_25_comparison_all_methods.png
```

---

## ⚙️ **Parametri Configurabili**

### **Numero di Cluster**

Il dataset ha **8 categorie naturali**. Puoi testare:

| k | Interpretazione | Quando Usare |
|---|-----------------|--------------|
| 3 | Macro-gruppi (famiglie di oli) | Esplorazione iniziale |
| 5 | Gruppi intermedi | Analisi struttura |
| 8 | Categorie reali | Validazione con etichette |

**Modificare in clustering_fun.m:**
```matlab
% Linea 86
nClusters = 8;  % Cambia: 3, 5, 8, etc.
```

**Modificare in kmeans_etal.m:**
```matlab
% Linea 11
k_clusters = 8;  % Cambia: 3, 5, 8, etc.
```

**Modificare in Python:**
```python
# Linea 45 circa
n_clusters = 8  # Cambia: 3, 5, 8, etc.
```

---

### **DBSCAN - Parametri**

```matlab
% kmeans_etal.m, linea 50
minpts = 5;  % Minimo punti per cluster core

% Se eps automatico non funziona, prova manualmente:
eps = 0.5;  % Raggio eps
cls = dbscan(data, minpts, eps);
```

**Guida scelta parametri:**
- **minpts basso (3-5)**: Più permissivo, meno noise
- **minpts alto (7-10)**: Più restrittivo, più noise
- **eps**: Guarda k-distance plot per gomito

---

### **OPTICS - Parametro k**

```matlab
% kmeans_etal.m, linea 131
k_optics = round(nSamples / 25);  % Default: ~23 per 572 campioni

% Per cluster più piccoli:
k_optics = 10;

% Per cluster più grandi:
k_optics = 30;
```

---

## 📈 **Output e Interpretazione**

### **Silhouette Score**

**Range**: -1 a +1

| Score | Interpretazione |
|-------|-----------------|
| 0.7 - 1.0 | Clustering eccellente |
| 0.5 - 0.7 | Clustering buono |
| 0.3 - 0.5 | Clustering moderato |
| < 0.3 | Clustering debole |

**Nel progetto:**
- Ward + euclidean: ~0.45 (buono)
- Single linkage: ~0.30 (moderato)
- DBSCAN: varia con eps

---

### **Dendrogrammi**

**Come leggerli:**
```
Height (y-axis) = Distanza di fusione
    |
  4 |           /\
    |          /  \
  3 |    /\   /    \
    |   /  \ /      \
  2 | /\   /\       /\
    |/  \_/  \_____/  \
    +--------------------
      Campioni ordinati
```

**Cutoff line**: Dove tagli determina il numero di cluster
- Alto cutoff → Pochi cluster grandi
- Basso cutoff → Molti cluster piccoli

---

### **Scatter PC1 vs PC2**

**Interpretazione:**
- **Cluster separati**: Metodo ha funzionato bene
- **Cluster sovrapposti**: Difficile separare con questi dati
- **Varianza spiegata**: PC1+PC2 tipicamente 40-60% del totale

---

### **K-distance Plot (DBSCAN)**

**Trova eps ottimale:**
1. Cerca il "gomito" nella curva
2. Il valore y del gomito = eps ottimale
3. Linea rossa (eps usato) dovrebbe essere vicina al gomito

**Se non c'è gomito chiaro:**
- Dati troppo uniformi
- Prova minpts diversi
- DBSCAN potrebbe non essere adatto

---

### **Reachability Plot (OPTICS)**

**Interpretazione struttura:**
```
High peak
    |    |           |
    |    |           |
Low  \___|  Valley  |___  Valley
     ^^^^^^^^^      ^^^^^^^^^
     Cluster 1      Cluster 2
```

- **Valli profonde**: Cluster densi
- **Picchi alti**: Separazione tra cluster
- **Plateau alto**: Rumore/outliers

---

## 🔧 **Troubleshooting**

### **Errore: "cluster() - Input OPTIONS must be a structure"**

**Causa**: Conflitto PLS Toolbox vs Statistics Toolbox

**Soluzione**: Già implementata in `clustering_fun.m`:
```matlab
evrimovepath('bottom');  % Sposta PLS Toolbox in fondo al path
```

---

### **Errore: "Unrecognized function 'dbscan'"**

**Causa**: PLS Toolbox non installato

**Soluzione**: DBSCAN è opzionale
```matlab
% Lo script continua comunque
try
    [cls,epsdist] = dbscan(data, minpts);
catch
    fprintf('DBSCAN non disponibile\n');
end
```

---

### **DBSCAN trova solo 1 cluster**

**Causa**: eps automatico troppo grande

**Soluzione**: Già implementata - lo script prova automaticamente:
```matlab
eps_values = [epsdist*0.3, epsdist*0.5, epsdist*0.7];
```

Oppure imposta manualmente:
```matlab
eps = 0.5;  % Prova valori tra 0.3 e 1.0
cls = dbscan(data, minpts, eps);
```

---

### **Warning: "Matrix is singular"**

**Causa**: Mahalanobis distance con dati collineari

**Soluzione**: Normale per questo dataset, viene gestito automaticamente nel codice

---

### **Python: "ModuleNotFoundError: No module named 'numpy'"**

**Causa**: Pacchetti non installati nel venv

**Soluzione**:
```bash
source venv/bin/activate  # Attiva venv
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

---

### **Troppi cluster nel dendrogramma (illeggibile)**

**Soluzione**: Limita numero label
```matlab
% Mostra solo 100 label invece di 572
dendrogram(Z, 100);
```

---

## 📝 **Note Tecniche**

### **Normalizzazione Dati**

**MATLAB:**
```matlab
data = normalize(olivdata);  % zscore: mean=0, std=1
```

**Python:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data = scaler.fit_transform(olivdata)
```

**Perché necessaria:**
- Le 7 variabili hanno scale diverse
- Senza normalizzazione, variabili con valori grandi dominano
- Tutti i metodi di clustering sono sensibili alla scala

---

### **PCA per Visualizzazione**

```matlab
[u, s, v] = svds(data, 2);  % Top 2 componenti
scores = u * s;  % Scores PC1 e PC2
```

**Limitazione:**
- PC1+PC2 spiegano solo ~40-60% della varianza
- Cluster ben separati in PC1/PC2 potrebbero sovrapporsi nelle altre 5 dimensioni
- È solo una **proiezione 2D** di dati 7D

---

### **Differenze Linkage Methods**

| Metodo | Distanza Cluster | Pro | Contro |
|--------|------------------|-----|--------|
| **Ward** | Minimizza varianza intra-cluster | Cluster compatti, bilanciati | Solo euclidean |
| **Single** | Minima distanza punto-punto | Trova cluster allungati | Effetto "catena" |
| **Complete** | Massima distanza punto-punto | Cluster compatti | Sensibile a outlier |
| **Average** | Media di tutte le distanze | Compromesso bilanciato | - |
| **Centroid** | Distanza tra centroidi | Intuitivo | Può dare inversioni |

---

### **Differenze Distance Metrics**

| Metrica | Formula | Quando Usare |
|---------|---------|--------------|
| **Euclidean** | √Σ(xi-yi)² | Default, dati continui |
| **Cityblock** | Σ\|xi-yi\| | Manhattan, meno sensibile a outlier |
| **Correlation** | 1 - corr(x,y) | Pattern simili, non magnitudine |
| **Cosine** | 1 - cos(θ) | Direzione, non lunghezza |
| **Mahalanobis** | Considera covarianza | Variabili correlate |
| **Minkowski (p)** | (Σ\|xi-yi\|^p)^(1/p) | Generalizzazione euclidean |

---

### **Validazione Clustering**

**Metodi interni** (senza etichette vere):
- ✅ Silhouette score (usato nel progetto)
- Davies-Bouldin index
- Calinski-Harabasz index

**Metodi esterni** (con etichette vere):
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Purity

**Nel progetto**: Usiamo solo silhouette (metodi interni) perché trattiamo il dataset come non etichettato.

---

## 📚 **Riferimenti**

### **Documentazione**
- [MATLAB Statistics Toolbox](https://www.mathworks.com/help/stats/)
- [scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Silhouette Score](https://en.wikipedia.org/wiki/Silhouette_(clustering))

### **Algoritmi**
- **OPTICS**: Ankerst et al. (1999) - "OPTICS: Ordering Points To Identify the Clustering Structure"
- **DBSCAN**: Ester et al. (1996) - "A Density-Based Algorithm for Discovering Clusters"
- **Ward**: Ward (1963) - "Hierarchical Grouping to Optimize an Objective Function"

### **Dataset**
- Olive Oil dataset: Analisi chimica di oli d'oliva italiani da diverse regioni

---

## 👥 **Autori**

**Progetto**: Elaborazione Dati Scientifici - Università
**Anno Accademico**: 2024/2025
**Dataset**: Olive Oil (572 campioni, 7 variabili, 8 categorie)

---

## 📄 **Licenza**

Materiale didattico per scopi educativi.

---

## 🆘 **Supporto**

Per problemi o domande:
1. Controlla la sezione [Troubleshooting](#-troubleshooting)
2. Verifica di avere tutti i [Requisiti](#-requisiti)
3. Rivedi i [Parametri Configurabili](#-parametri-configurabili)

---

## 📊 **Quick Reference Card**

### **Comandi Rapidi**

```matlab
% MATLAB - Analisi gerarchica completa
run('clustering_fun.m')

% MATLAB - K-means, DBSCAN, OPTICS
run('kmeans_etal.m')

% Python - Tutto in uno
python analisi_completa_oliveoil.py
```

### **Modifiche Comuni**

```matlab
% Cambia numero cluster (in tutti i file)
nClusters = 8;      % clustering_fun.m
k_clusters = 8;     % kmeans_etal.m

% Cambia parametri DBSCAN
minpts = 5;         % 3-10
eps = 0.5;          % Guarda k-distance plot

% Cambia parametro OPTICS
k_optics = 15;      % 10-30
```

### **Interpretazione Veloce**

| Metrica | Buono | Attenzione |
|---------|-------|------------|
| Silhouette | > 0.5 | < 0.3 |
| N. cluster DBSCAN | 2-8 | 1 o >15 |
| PC1+PC2 varianza | > 50% | < 30% |

---

**Ultima revisione**: Novembre 2025
**Versione**: 2.0 - Iterativa con k configurabile
