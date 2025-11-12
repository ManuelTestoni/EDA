"""
ANALISI COMPLETA CLUSTERING - OLIVE OIL DATASET
Script Python con scikit-learn che replica l'analisi MATLAB

Genera tutti i grafici richiesti:
1. Dendrogrammi con diverse combinazioni distanza/linkage
2. PC1 vs PC2 colorati per cluster
3. Figura combinata con clustering variabili e heatmap ordinata
4. Confronto con K-means, DBSCAN e OPTICS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings('ignore')

# Impostazioni grafici
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 60)
print("=== ANALISI COMPLETA CLUSTERING - OLIVE OIL DATASET ===")
print("=" * 60)

# ============================================================================
# CARICAMENTO E PREPROCESSING DATI
# ============================================================================
print("\n--- CARICAMENTO DATI ---")
df = pd.read_csv('oliveoil.csv', header=None)
data_raw = df.values
n_samples, n_vars = data_raw.shape
print(f"Dataset: {n_samples} campioni x {n_vars} variabili")

# Standardizzazione (autoscaling)
scaler = StandardScaler()
data = scaler.fit_transform(data_raw)
print("Dati standardizzati (media=0, std=1)")

# PCA per visualizzazione (prime 2 componenti)
pca = PCA(n_components=2)
scores = pca.fit_transform(data)
print(f"PCA: varianza spiegata = {pca.explained_variance_ratio_.sum()*100:.2f}%")

# ============================================================================
# PARTE 1: CLUSTERING GERARCHICO - TEST DIVERSE COMBINAZIONI
# ============================================================================
print("\n" + "=" * 60)
print("--- PARTE 1: CLUSTERING GERARCHICO ---")
print("=" * 60)

# Combinazioni da testare
linkage_methods = ['single', 'average', 'complete', 'centroid']
distance_metrics = ['euclidean', 'minkowski', 'mahalanobis']

# Per Mahalanobis serve la matrice di covarianza inversa
VI = np.linalg.inv(np.cov(data.T))

n_clusters = 3  # Numero di cluster da cercare
results = []
fig_num = 1

print("\nTestando combinazioni di linkage e distanza:")

for link_method in linkage_methods:
    for dist_metric in distance_metrics:
        try:
            # Calcola matrice delle distanze
            if dist_metric == 'mahalanobis':
                distances = pdist(data, metric='mahalanobis', VI=VI)
            elif dist_metric == 'minkowski':
                distances = pdist(data, metric='minkowski', p=2)  # p=2 è euclidean
            else:
                distances = pdist(data, metric=dist_metric)
            
            # Calcola linkage
            Z = linkage(distances, method=link_method)
            
            # Ottieni cluster
            clusters = fcluster(Z, n_clusters, criterion='maxclust')
            
            # Calcola silhouette
            sil_score = silhouette_score(data, clusters)
            
            print(f"  {link_method:10s} + {dist_metric:12s}: silhouette = {sil_score:.4f}")
            
            # Salva risultati
            results.append({
                'linkage': link_method,
                'distance': dist_metric,
                'silhouette': sil_score,
                'Z': Z,
                'clusters': clusters
            })
            
            # GRAFICO: Dendrogramma
            plt.figure(figsize=(14, 6))
            dendrogram(Z, no_labels=True, color_threshold=None)
            plt.title(f'Dendrogram: {link_method} linkage + {dist_metric} distance\n' + 
                     f'Silhouette = {sil_score:.3f}', fontsize=14, fontweight='bold')
            plt.xlabel('Sample index', fontsize=12)
            plt.ylabel('Distance', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'fig_{fig_num:02d}_{link_method}_{dist_metric}_dendrogram.png', 
                       dpi=150, bbox_inches='tight')
            print(f"    → Salvata figura {fig_num}: fig_{fig_num:02d}_{link_method}_{dist_metric}_dendrogram.png")
            fig_num += 1
            
        except Exception as e:
            print(f"  {link_method:10s} + {dist_metric:12s}: ERRORE - {str(e)}")

# Trova il migliore
best = max(results, key=lambda x: x['silhouette'])
print(f"\n*** MIGLIORE COMBINAZIONE: {best['linkage']} + {best['distance']} " + 
      f"(silhouette = {best['silhouette']:.4f}) ***")

# ============================================================================
# PARTE 2: FIGURE RICHIESTE CON IL CLUSTERING MIGLIORE
# ============================================================================
print("\n" + "=" * 60)
print("--- PARTE 2: FIGURE RICHIESTE (MIGLIORE CLUSTERING) ---")
print("=" * 60)

best_Z = best['Z']
best_clusters = best['clusters']
best_linkage = best['linkage']
best_distance = best['distance']
best_silhouette = best['silhouette']

# Calcola cutoff per avere esattamente n_clusters
# Il cutoff è la distanza al (n_samples - n_clusters)-esimo merge
cutoff = best_Z[-(n_clusters-1), 2]

# FIGURA 1: Samples dendrogram con cutoff colorato
print(f"\nFIGURA RICHIESTA #1: Samples dendrogram (cutoff = {cutoff:.4f})")
plt.figure(figsize=(16, 6))
dendrogram(best_Z, no_labels=True, color_threshold=cutoff)
plt.axhline(y=cutoff, color='r', linestyle='--', linewidth=2, label=f'Cutoff = {cutoff:.3f}')
plt.title(f'Samples Dendrogram - {best_linkage} + {best_distance}\n' +
         f'Cutoff = {cutoff:.3f}', fontsize=14, fontweight='bold')
plt.xlabel('Sample index', fontsize=12)
plt.ylabel('Linkage distance', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'fig_{fig_num:02d}_BEST_dendrogram_with_cutoff.png', dpi=150, bbox_inches='tight')
print(f"  → Salvata: fig_{fig_num:02d}_BEST_dendrogram_with_cutoff.png")
fig_num += 1

# FIGURA 2: PC1 vs PC2 colorati per cluster
print(f"\nFIGURA RICHIESTA #2: PC1 vs PC2 colorati per cluster")
plt.figure(figsize=(10, 8))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=best_clusters, 
                     cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.colorbar(scatter, label='Cluster')
plt.title(f'PC1 vs PC2 colored by clusters\n' +
         f'{best_linkage} + {best_distance}, Cutoff = {cutoff:.3f}',
         fontsize=14, fontweight='bold')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'fig_{fig_num:02d}_BEST_PC1_PC2_clusters.png', dpi=150, bbox_inches='tight')
print(f"  → Salvata: fig_{fig_num:02d}_BEST_PC1_PC2_clusters.png")
fig_num += 1

# FIGURA 3: Clustering variabili + heatmap ordinata (slide 85)
print(f"\nFIGURA RICHIESTA #3: Clustering variabili + heatmap ordinata")

# Clustering delle variabili usando correlazione
data_transposed = data.T  # Trasponi per clustering variabili
correlation_distances = pdist(data_transposed, metric='correlation')
Z_vars = linkage(correlation_distances, method='average')

# Ottieni ordine dal dendrogramma campioni
dn_samples = dendrogram(best_Z, no_plot=True)
sample_order = dn_samples['leaves']

# Ottieni ordine dal dendrogramma variabili
dn_vars = dendrogram(Z_vars, no_plot=True)
var_order = dn_vars['leaves']

# Crea figura 2x2 come slide 85
fig = plt.figure(figsize=(14, 12))

# Subplot 1 (top-left): Info/titolo
ax1 = plt.subplot(2, 2, 1)
ax1.axis('off')
info_text = f'Olive Oil Clustering\n\n' + \
           f'Method: {best_linkage} + {best_distance}\n' + \
           f'Cutoff: {cutoff:.3f}\n' + \
           f'Silhouette: {best_silhouette:.3f}\n' + \
           f'N. Clusters: {n_clusters}'
ax1.text(0.5, 0.5, info_text, ha='center', va='center', 
        fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Subplot 2 (top-right): Dendrogramma campioni (orizzontale)
ax2 = plt.subplot(2, 2, 2)
dendrogram(best_Z, no_labels=True, color_threshold=cutoff, ax=ax2)
ax2.set_title('Sample Dendrogram', fontsize=12, fontweight='bold')
ax2.set_ylabel('Distance', fontsize=10)
ax2.set_xlabel('')
ax2.tick_params(labelbottom=False)

# Subplot 3 (bottom-left): Dendrogramma variabili (verticale)
ax3 = plt.subplot(2, 2, 3)
dendrogram(Z_vars, orientation='left', no_labels=True, ax=ax3)
ax3.set_title('Variable Dendrogram\n(correlation)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Distance', fontsize=10)
ax3.set_ylabel('')
ax3.tick_params(labelleft=False)

# Subplot 4 (bottom-right): Heatmap ordinata
ax4 = plt.subplot(2, 2, 4)
data_ordered = data[sample_order, :][:, var_order]
im = ax4.imshow(data_ordered.T, cmap='jet', aspect='auto', interpolation='nearest')
ax4.set_title('Standardized data\n(ordered by dendrograms)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Samples (ordered)', fontsize=10)
ax4.set_ylabel('Variables (ordered)', fontsize=10)
plt.colorbar(im, ax=ax4, label='Standardized value')

plt.tight_layout()
plt.savefig(f'fig_{fig_num:02d}_BEST_combined_heatmap.png', dpi=150, bbox_inches='tight')
print(f"  → Salvata: fig_{fig_num:02d}_BEST_combined_heatmap.png")
fig_num += 1

# ============================================================================
# PARTE 3: K-MEANS
# ============================================================================
print("\n" + "=" * 60)
print("--- PARTE 3: K-MEANS ---")
print("=" * 60)

kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
kmeans_clusters = kmeans.fit_predict(data)
kmeans_silhouette = silhouette_score(data, kmeans_clusters)
print(f"K-means: silhouette = {kmeans_silhouette:.4f}")

# GRAFICO: PC1 vs PC2 per K-means
plt.figure(figsize=(10, 8))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=kmeans_clusters,
                     cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0],
           pca.transform(kmeans.cluster_centers_)[:, 1],
           c='red', s=300, alpha=0.8, edgecolors='black', linewidth=2, 
           marker='*', label='Centroids')
plt.colorbar(scatter, label='Cluster')
plt.title(f'PC1 vs PC2 - K-means (k={n_clusters})\n' +
         f'Silhouette = {kmeans_silhouette:.3f}',
         fontsize=14, fontweight='bold')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'fig_{fig_num:02d}_kmeans_PC1_PC2.png', dpi=150, bbox_inches='tight')
print(f"  → Salvata: fig_{fig_num:02d}_kmeans_PC1_PC2.png")
fig_num += 1

# GRAFICO: Silhouette plot K-means
plt.figure(figsize=(10, 6))
silhouette_vals = silhouette_samples(data, kmeans_clusters)
y_lower = 10
for i in range(n_clusters):
    cluster_silhouette_vals = silhouette_vals[kmeans_clusters == i]
    cluster_silhouette_vals.sort()
    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                     alpha=0.7, label=f'Cluster {i}')
    y_lower = y_upper + 10
plt.axvline(x=kmeans_silhouette, color='red', linestyle='--', linewidth=2,
           label=f'Mean = {kmeans_silhouette:.3f}')
plt.title(f'Silhouette Plot - K-means', fontsize=14, fontweight='bold')
plt.xlabel('Silhouette coefficient', fontsize=12)
plt.ylabel('Cluster', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'fig_{fig_num:02d}_kmeans_silhouette.png', dpi=150, bbox_inches='tight')
print(f"  → Salvata: fig_{fig_num:02d}_kmeans_silhouette.png")
fig_num += 1

# ============================================================================
# PARTE 4: DBSCAN
# ============================================================================
print("\n" + "=" * 60)
print("--- PARTE 4: DBSCAN ---")
print("=" * 60)

# DBSCAN - prova con diversi parametri
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_clusters = dbscan.fit_predict(data)
n_clusters_dbscan = len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)
n_noise = list(dbscan_clusters).count(-1)

print(f"DBSCAN: eps=0.5, min_samples=5")
print(f"  Cluster trovati: {n_clusters_dbscan}")
print(f"  Noise points: {n_noise}")

if n_clusters_dbscan > 1:
    # Calcola silhouette solo per punti non-noise
    mask = dbscan_clusters != -1
    if mask.sum() > 0:
        dbscan_silhouette = silhouette_score(data[mask], dbscan_clusters[mask])
        print(f"  Silhouette (no noise): {dbscan_silhouette:.4f}")
    else:
        dbscan_silhouette = None
else:
    dbscan_silhouette = None

# GRAFICO: PC1 vs PC2 per DBSCAN
plt.figure(figsize=(10, 8))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=dbscan_clusters,
                     cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.colorbar(scatter, label='Cluster (-1 = noise)')
title_text = f'PC1 vs PC2 - DBSCAN\n' + \
            f'eps=0.5, min_samples=5, {n_clusters_dbscan} clusters + {n_noise} noise'
if dbscan_silhouette:
    title_text += f'\nSilhouette = {dbscan_silhouette:.3f}'
plt.title(title_text, fontsize=14, fontweight='bold')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'fig_{fig_num:02d}_dbscan_PC1_PC2.png', dpi=150, bbox_inches='tight')
print(f"  → Salvata: fig_{fig_num:02d}_dbscan_PC1_PC2.png")
fig_num += 1

# ============================================================================
# PARTE 5: OPTICS
# ============================================================================
print("\n" + "=" * 60)
print("--- PARTE 5: OPTICS ---")
print("=" * 60)

optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)
optics_clusters = optics.fit_predict(data)
n_clusters_optics = len(set(optics_clusters)) - (1 if -1 in optics_clusters else 0)
n_noise_optics = list(optics_clusters).count(-1)

print(f"OPTICS: min_samples=5")
print(f"  Cluster trovati: {n_clusters_optics}")
print(f"  Noise points: {n_noise_optics}")

# GRAFICO: Reachability plot
plt.figure(figsize=(14, 6))
reachability = optics.reachability_[optics.ordering_]
plt.bar(range(len(reachability)), reachability, width=1.0)
plt.title('OPTICS Reachability Plot', fontsize=14, fontweight='bold')
plt.xlabel('Sample order', fontsize=12)
plt.ylabel('Reachability distance', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'fig_{fig_num:02d}_optics_reachability.png', dpi=150, bbox_inches='tight')
print(f"  → Salvata: fig_{fig_num:02d}_optics_reachability.png")
fig_num += 1

# GRAFICO: PC1 vs PC2 per OPTICS
plt.figure(figsize=(10, 8))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=optics_clusters,
                     cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.colorbar(scatter, label='Cluster (-1 = noise)')
plt.title(f'PC1 vs PC2 - OPTICS\n' +
         f'{n_clusters_optics} clusters + {n_noise_optics} noise',
         fontsize=14, fontweight='bold')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'fig_{fig_num:02d}_optics_PC1_PC2.png', dpi=150, bbox_inches='tight')
print(f"  → Salvata: fig_{fig_num:02d}_optics_PC1_PC2.png")
fig_num += 1

# ============================================================================
# PARTE 6: CONFRONTO FINALE
# ============================================================================
print("\n" + "=" * 60)
print("--- PARTE 6: CONFRONTO METODI ---")
print("=" * 60)

# GRAFICO: Confronto side-by-side
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Hierarchical
ax = axes[0, 0]
scatter = ax.scatter(scores[:, 0], scores[:, 1], c=best_clusters,
                    cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
ax.set_title(f'Hierarchical ({best_linkage}+{best_distance})\n' +
            f'Silhouette = {best_silhouette:.3f}',
            fontsize=12, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Cluster')

# K-means
ax = axes[0, 1]
scatter = ax.scatter(scores[:, 0], scores[:, 1], c=kmeans_clusters,
                    cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
ax.scatter(pca.transform(kmeans.cluster_centers_)[:, 0],
          pca.transform(kmeans.cluster_centers_)[:, 1],
          c='red', s=200, alpha=0.8, edgecolors='black', linewidth=2, marker='*')
ax.set_title(f'K-means (k={n_clusters})\n' +
            f'Silhouette = {kmeans_silhouette:.3f}',
            fontsize=12, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Cluster')

# DBSCAN
ax = axes[1, 0]
scatter = ax.scatter(scores[:, 0], scores[:, 1], c=dbscan_clusters,
                    cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
title_text = f'DBSCAN\n{n_clusters_dbscan} clusters + {n_noise} noise'
if dbscan_silhouette:
    title_text += f'\nSilhouette = {dbscan_silhouette:.3f}'
ax.set_title(title_text, fontsize=12, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Cluster')

# OPTICS
ax = axes[1, 1]
scatter = ax.scatter(scores[:, 0], scores[:, 1], c=optics_clusters,
                    cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
ax.set_title(f'OPTICS\n{n_clusters_optics} clusters + {n_noise_optics} noise',
            fontsize=12, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Cluster')

plt.suptitle('Confronto Metodi di Clustering', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'fig_{fig_num:02d}_CONFRONTO_tutti_metodi.png', dpi=150, bbox_inches='tight')
print(f"  → Salvata: fig_{fig_num:02d}_CONFRONTO_tutti_metodi.png")
fig_num += 1

# ============================================================================
# RIEPILOGO FINALE
# ============================================================================
print("\n" + "=" * 60)
print("=== RIEPILOGO RISULTATI FINALI ===")
print("=" * 60)
print(f"Dataset: {n_samples} campioni, {n_vars} variabili")
print(f"\n1. CLUSTERING GERARCHICO (MIGLIORE):")
print(f"   Linkage: {best_linkage}")
print(f"   Distanza: {best_distance}")
print(f"   Cutoff: {cutoff:.4f}")
print(f"   Silhouette medio: {best_silhouette:.4f}")
print(f"   Numero cluster: {n_clusters}")

print(f"\n2. K-MEANS:")
print(f"   k: {n_clusters}")
print(f"   Silhouette medio: {kmeans_silhouette:.4f}")
print(f"   Differenza vs Hierarchical: {abs(kmeans_silhouette - best_silhouette):.4f}")

print(f"\n3. DBSCAN:")
print(f"   eps: 0.5, min_samples: 5")
print(f"   Cluster trovati: {n_clusters_dbscan}")
print(f"   Noise points: {n_noise}")
if dbscan_silhouette:
    print(f"   Silhouette medio (no noise): {dbscan_silhouette:.4f}")

print(f"\n4. OPTICS:")
print(f"   min_samples: 5")
print(f"   Cluster trovati: {n_clusters_optics}")
print(f"   Noise points: {n_noise_optics}")

print("\n" + "=" * 60)
print("CONCLUSIONE CONFRONTO:")
if kmeans_silhouette > best_silhouette:
    print(f">>> K-means ha performance MIGLIORE (Δsil = +{kmeans_silhouette - best_silhouette:.4f})")
else:
    print(f">>> Hierarchical ha performance MIGLIORE (Δsil = +{best_silhouette - kmeans_silhouette:.4f})")
print("=" * 60)

print(f"\nTotale figure generate: {fig_num - 1}")
print("\n=== ANALISI COMPLETA TERMINATA ===")
print("Tutte le figure sono state salvate nella cartella corrente.")
