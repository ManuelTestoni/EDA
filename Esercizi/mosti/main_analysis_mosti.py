#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
 MAIN_ANALYSIS_MOSTI.py - Classification Analysis of Mosti (Grape Must)
 Methods: PCA (exploratory), SIMCA (class modeling), PLS-DA (discriminant)
 Dataset: mosti.mat (98 samples, 6 anthocyanin HPLC variables, 5 grape varieties)
==========================================================================
 Python port of main_analysis_mosti.m
 Requires: numpy, scipy, matplotlib, seaborn, scikit-learn, pandas

 Grape varieties:
   1 = Ancellotta (A)
   2 = Montepulciano (M)
   3 = Lambrusco Pugliese (LP)
   4 = Sangiovese (S)
   5 = Nero d'Avola (N)

 Variables (anthocyanin HPLC areas %):
   DPD%, CYD%, PTD%, PND%, MVD%, R lib/lrg

 Two vintages: 2000 and 2001 (combined for classification).
==========================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import f as f_dist
from scipy.special import erfinv
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.size': 10,
})

print('=' * 60)
print('  MOSTI (GRAPE MUST) CLASSIFICATION ANALYSIS')
print(f'  SIMCA + PLS-DA | {datetime.now().strftime("%d-%b-%Y %H:%M:%S")}')
print('=' * 60 + '\n')

# %% ---- 0. SETUP -----------------------------------------------------------
basePath = os.path.dirname(os.path.abspath(__file__))
plotDir = os.path.join(basePath, 'plots_py')
os.makedirs(plotDir, exist_ok=True)

np.random.seed(42)

# Class color palette (5 grape varieties)
classColors = np.array([
    [0.20, 0.40, 0.80],   # A  - blue
    [0.85, 0.20, 0.20],   # M  - red
    [0.15, 0.70, 0.30],   # LP - green
    [0.80, 0.20, 0.80],   # S  - magenta
    [0.95, 0.60, 0.10],   # N  - orange
])

# %% ---- HELPER FUNCTIONS ---------------------------------------------------

def build_simca_model(X, y, ncomp_vec, cl=0.95):
    """Build SIMCA models (one PCA per class) - replicates MATLAB build_simca_model."""
    classes = np.unique(y)
    nClasses = len(classes)
    nsamp, nvar = X.shape

    if np.isscalar(ncomp_vec) or len(ncomp_vec) == 1:
        nc_val = int(ncomp_vec) if np.isscalar(ncomp_vec) else int(ncomp_vec[0])
        ncomp_vec = np.array([nc_val] * nClasses)
    else:
        ncomp_vec = np.array(ncomp_vec, dtype=int)

    sensspec = np.zeros((nClasses, nClasses))
    model = {
        'PCmodels': [None] * nClasses,
        'SIMCA': {
            'accept': [None] * nClasses,
            'crit': np.zeros((nsamp, nClasses)),
            'sensspec': None,
            'ncomp': None,
            'nclass': nClasses,
            'cl': cl,
            'predclass': None,
        }
    }

    for ic in range(nClasses):
        cl_ind = np.where(y == classes[ic])[0]
        Xd1 = X[cl_ind, :]
        Xd2 = X.copy()

        mx_c = np.nanmean(Xd1, axis=0)
        sx_c = np.nanstd(Xd1, axis=0, ddof=1)
        sx_c[sx_c == 0] = 1.0

        Xd1_sc = (Xd1 - mx_c) / sx_c
        Xd2_sc = (Xd2 - mx_c) / sx_c

        nc = min(int(ncomp_vec[ic]), min(Xd1_sc.shape) - 1)
        if nc < 1:
            nc = 1
        ncomp_vec[ic] = nc

        U, S_diag, Vt = np.linalg.svd(Xd1_sc, full_matrices=False)
        P_class = Vt[:nc, :].T  # nvar x nc
        T_class = U[:, :nc] * S_diag[:nc]

        lambda_diag = S_diag ** 2 / (Xd1.shape[0] - 1)
        eigs_all = lambda_diag.copy()

        tot_var = np.sum(lambda_diag)
        ev = 100.0 * lambda_diag / tot_var
        cv_var = np.cumsum(ev)

        # Project all data
        T1 = Xd2_sc @ P_class  # nsamp x nc
        lambda_mat = np.diag(lambda_diag[:nc])
        lambda_mat_inv = np.diag(1.0 / lambda_diag[:nc])

        tsq = np.sum((T1 @ lambda_mat_inv) * T1, axis=1)

        Xd2_recon = T1 @ P_class.T
        E = Xd2_sc - Xd2_recon
        q = np.sum(E ** 2, axis=1)

        # T2 limit
        N_class = Xd1.shape[0]
        A = nc
        try:
            F_crit = f_dist.ppf(cl, A, N_class - A)
        except:
            z = np.sqrt(2) * erfinv(2 * cl - 1)
            chi2_approx = A * (1 - 2 / (9 * A) + z * np.sqrt(2 / (9 * A))) ** 3
            F_crit = chi2_approx / A
        t2lim = A * (N_class - 1) / (N_class - A) * F_crit

        # Q limit
        if len(eigs_all) > nc:
            theta1 = np.sum(eigs_all[nc:])
            theta2 = np.sum(eigs_all[nc:] ** 2)
            theta3 = np.sum(eigs_all[nc:] ** 3)
        else:
            theta1 = theta2 = theta3 = 0.0

        if theta1 == 0:
            qlim = 0.0
        else:
            h0 = 1 - 2 * theta1 * theta3 / (3 * theta2 ** 2)
            if h0 < 0.001:
                h0 = 0.001
            ca = np.sqrt(2) * erfinv(2 * cl - 1)
            h1 = ca * np.sqrt(2 * theta2 * h0 ** 2) / theta1
            h2 = theta2 * h0 * (h0 - 1) / theta1 ** 2
            qlim = theta1 * (1 + h1 + h2) ** (1 / h0)

        cr_i = np.zeros(nsamp)
        cl_acc = np.zeros(nsamp)

        for j in range(nsamp):
            if qlim > 0 and t2lim > 0:
                cr_i[j] = np.sqrt((tsq[j] / t2lim) ** 2 + (q[j] / qlim) ** 2)
            else:
                cr_i[j] = tsq[j] / max(t2lim, 1e-16)

            if cr_i[j] <= np.sqrt(2):
                cl_acc[j] = 1
                if y[j] == classes[ic]:
                    sensspec[ic, int(y[j]) - 1] += 1
            else:
                if y[j] != classes[ic]:
                    sensspec[ic, int(y[j]) - 1] += 1

        model['PCmodels'][ic] = {
            'prepr': (mx_c, sx_c),
            'res': E,
            'scores': T1,
            'loads': P_class,
            'eigs': eigs_all,
            'variance': (ev, cv_var),
            'critvals': (t2lim, qlim),
            'tsq': tsq,
            'qres': q,
        }
        model['SIMCA']['accept'][ic] = cl_acc
        model['SIMCA']['crit'][:, ic] = cr_i

    model['SIMCA']['sensspec'] = sensspec
    model['SIMCA']['ncomp'] = ncomp_vec.copy()
    model['SIMCA']['predclass'] = np.argmin(model['SIMCA']['crit'], axis=1) + 1  # 1-based
    return model


def predict_simca(Xpred, ypred, simcamod):
    """Predict class membership using a trained SIMCA model."""
    nClasses = simcamod['SIMCA']['nclass']
    ncomp = simcamod['SIMCA']['ncomp']
    nspred = Xpred.shape[0]
    has_labels = ypred is not None and len(ypred) > 0

    sensspec = np.zeros((nClasses, nClasses))
    pred = {
        'PCmodels': [None] * nClasses,
        'SIMCA': {
            'accept': [None] * nClasses,
            'crit': np.zeros((nspred, nClasses)),
            'sensspec': None,
            'ncomp': ncomp,
            'nclass': nClasses,
            'predclass': None,
        }
    }

    for ic in range(nClasses):
        mx_c, sx_c = simcamod['PCmodels'][ic]['prepr']
        P = simcamod['PCmodels'][ic]['loads']
        eigs_all = simcamod['PCmodels'][ic]['eigs']
        t2lim, qlim = simcamod['PCmodels'][ic]['critvals']

        Xp_sc = (Xpred - mx_c) / sx_c

        nc = int(ncomp[ic])
        Tp = Xp_sc @ P
        lambda_mat_inv = np.diag(1.0 / eigs_all[:nc])

        tsq = np.sum((Tp @ lambda_mat_inv) * Tp, axis=1)
        Ep = Xp_sc - Tp @ P.T
        q = np.sum(Ep ** 2, axis=1)

        cr_i = np.zeros(nspred)
        cl_acc = np.zeros(nspred)

        for j in range(nspred):
            if qlim > 0 and t2lim > 0:
                cr_i[j] = np.sqrt((tsq[j] / t2lim) ** 2 + (q[j] / qlim) ** 2)
            else:
                cr_i[j] = tsq[j] / max(t2lim, 1e-16)

            if cr_i[j] <= np.sqrt(2):
                cl_acc[j] = 1
                if has_labels and int(ypred[j]) == ic + 1:
                    sensspec[ic, int(ypred[j]) - 1] += 1
            else:
                if has_labels and int(ypred[j]) != ic + 1:
                    sensspec[ic, int(ypred[j]) - 1] += 1

        pred['PCmodels'][ic] = {
            'scores': Tp,
            'loads': P,
            'critvals': (t2lim, qlim),
            'tsq': tsq,
            'qres': q,
        }
        pred['SIMCA']['accept'][ic] = cl_acc
        pred['SIMCA']['crit'][:, ic] = cr_i

    if has_labels:
        pred['SIMCA']['sensspec'] = sensspec
    pred['SIMCA']['predclass'] = np.argmin(pred['SIMCA']['crit'], axis=1) + 1
    return pred


def nipals_pls2(X, Y, ncomp):
    """NIPALS PLS2 algorithm - replicates MATLAB nipals_pls2."""
    n, p = X.shape
    q = Y.shape[1]

    T = np.zeros((n, ncomp))
    P = np.zeros((p, ncomp))
    W = np.zeros((p, ncomp))
    Q = np.zeros((q, ncomp))
    bvec = np.zeros(ncomp)

    E = X.copy()
    F = Y.copy()

    for a in range(ncomp):
        maxcol = np.argmax(np.sum(F ** 2, axis=0))
        u = F[:, maxcol].copy()

        for it in range(500):
            w = E.T @ u
            w = w / np.linalg.norm(w)
            t = E @ w
            qq = F.T @ t / (t @ t)

            if q == 1:
                u = F @ qq / (qq @ qq)
                break

            u_new = F @ qq / (qq @ qq)
            if np.linalg.norm(u_new - u) / (np.linalg.norm(u_new) + 1e-16) < 1e-12:
                u = u_new
                break
            u = u_new

        pp = E.T @ t / (t @ t)
        b = u @ t / (t @ t)

        T[:, a] = t
        P[:, a] = pp
        W[:, a] = w
        Q[:, a] = qq
        bvec[a] = b

        E = E - np.outer(t, pp)
        F = F - b * np.outer(t, qq)

    Bpls = W @ np.linalg.inv(P.T @ W) @ np.diag(bvec) @ Q.T

    return {
        'T': T, 'P': P, 'W': W, 'Q': Q, 'B': bvec, 'Bpls': Bpls
    }


def compute_vip(plsmodel, X, Y):
    """Compute Variable Importance in Projection (VIP) scores."""
    W = plsmodel['W']
    T = plsmodel['T']
    Q = plsmodel['Q']
    p, ncomp = W.shape

    SS = np.zeros(ncomp)
    for a in range(ncomp):
        b = plsmodel['B'][a]
        SS[a] = b ** 2 * (T[:, a] @ T[:, a]) * (Q[:, a] @ Q[:, a])
    SStotal = np.sum(SS)

    VIP = np.zeros(p)
    for j in range(p):
        s = 0.0
        for a in range(ncomp):
            w_norm = np.linalg.norm(W[:, a])
            s += SS[a] * (W[j, a] / w_norm) ** 2
        VIP[j] = np.sqrt(p * s / SStotal)
    return VIP


def compute_discriminant_power(X, y, simcamod):
    """Calculate SIMCA discriminant power for each variable."""
    nClasses = simcamod['SIMCA']['nclass']
    M = X.shape[1]
    s2in = np.zeros(M)
    s2not = np.zeros(M)

    for ic in range(nClasses):
        iin = np.where(y == ic + 1)[0]
        inot = np.where(y != ic + 1)[0]
        res = simcamod['PCmodels'][ic]['res']
        q_res = res ** 2
        A = int(simcamod['SIMCA']['ncomp'][ic])

        s2in += (M / (M - A)) * np.sum(q_res[iin, :], axis=0) / len(iin)
        s2not += (M / (M - A)) * np.sum(q_res[inot, :], axis=0) / len(inot)

    dpow = np.sqrt(s2not / np.maximum(s2in, 1e-16)) - 1
    dpow = np.maximum(dpow, 0)
    return dpow


def compute_class_metrics(y_true, y_pred, nClasses):
    """Compute sensitivity, specificity, efficiency for each class."""
    sens = np.zeros(nClasses)
    spec = np.zeros(nClasses)

    for ic in range(nClasses):
        c = ic + 1
        TP = np.sum((y_true == c) & (y_pred == c))
        FN = np.sum((y_true == c) & (y_pred != c))
        TN = np.sum((y_true != c) & (y_pred != c))
        FP = np.sum((y_true != c) & (y_pred == c))

        sens[ic] = TP / max(TP + FN, 1)
        spec[ic] = TN / max(TN + FP, 1)

    eff = np.sqrt(sens * spec)
    acc = np.sum(y_true == y_pred) / len(y_true)
    return sens, spec, eff, acc


def plot_confusion_matrix_custom(ax, y_true, y_pred, classLabels, titleStr):
    """Plot a professional confusion matrix on given axes."""
    nCl = len(classLabels)
    cm = np.zeros((nCl, nCl), dtype=int)
    for i in range(nCl):
        for j in range(nCl):
            cm[i, j] = np.sum((y_true == i + 1) & (y_pred == j + 1))

    ax.imshow(cm, cmap='bone_r', aspect='auto')
    total_per_row = cm.sum(axis=1)
    for i in range(nCl):
        for j in range(nCl):
            pct = 100 * cm[i, j] / max(total_per_row[i], 1)
            color = 'green' if i == j else ('red' if cm[i, j] > 0 else 'gray')
            txt = f'{cm[i,j]}\n({pct:.0f}%)' if cm[i, j] > 0 else '0'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                    fontweight='bold', color=color)

    ax.set_xticks(range(nCl))
    ax.set_xticklabels(classLabels)
    ax.set_yticks(range(nCl))
    ax.set_yticklabels(classLabels)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_title(titleStr)
    acc = 100 * np.trace(cm) / cm.sum()
    ax.text(0.5, -0.15, f'Accuracy: {acc:.1f}%', transform=ax.transAxes,
            fontsize=10, fontweight='bold', ha='center')


# %% ---- 1. DATA LOADING ----------------------------------------------------
print('[1] Loading data...')
dataFile = os.path.join(basePath, 'mosti.mat')
loadedData = loadmat(dataFile, squeeze_me=False)

X_full = np.array(loadedData['mosti'], dtype=float)  # 98 x 6
classid_v = np.array(loadedData['classid_v']).flatten()
y_full = classid_v.copy()

# Sample names
nameobj_raw = loadedData['nameobj_mosti'].flatten()
nameobj_mosti = [str(n).strip() for n in nameobj_raw]

# Variable names
namevar_raw = loadedData['namevar_mosti'].flatten()
namevar_mosti = [str(n).strip() for n in namevar_raw]

if len(namevar_mosti) == X_full.shape[1]:
    featNames = namevar_mosti
else:
    featNames = ['DPD%', 'CYD%', 'PTD%', 'PND%', 'MVD%', 'R lib/lrg']

classNames = ['A', 'M', 'LP', 'S', 'N']
classFullNames = ['Ancellotta', 'Montepulciano', 'Lambrusco Pugliese',
                  'Sangiovese', "Nero d'Avola"]

N_total, M_vars = X_full.shape
nClasses = len(classNames)

# Extract vintage info from sample names
annata = np.zeros(N_total, dtype=int)
for i in range(N_total):
    nome = nameobj_mosti[i]
    if '_00' in nome or nome.endswith('00'):
        annata[i] = 2000
    elif '_01' in nome or nome.endswith('01'):
        annata[i] = 2001

print(f'   Samples: {N_total} | Variables: {M_vars} | Classes: {nClasses}')
for ic in range(nClasses):
    n00 = np.sum((y_full == ic + 1) & (annata == 2000))
    n01 = np.sum((y_full == ic + 1) & (annata == 2001))
    total_c = np.sum(y_full == ic + 1)
    print(f'   Class {ic+1} ({classFullNames[ic]}): {total_c} samples  '
          f'[2000: {n00} | 2001: {n01}]')
print()


# %% ---- 1b. PREPROCESSING ANALYSIS ----------------------------------------
print('[1b] Preprocessing Analysis...')

preprocessing_method = 'Autoscaling (Mean-Centering + Unit Variance)'

raw_means = np.mean(X_full, axis=0)
raw_stds = np.std(X_full, axis=0, ddof=1)
raw_mins = np.min(X_full, axis=0)
raw_maxs = np.max(X_full, axis=0)
raw_ranges = raw_maxs - raw_mins

print(f'   Preprocessing: {preprocessing_method}')
print('   Raw Variable Statistics:')
print(f'   {"Variable":<12s}  {"Mean":>7s}  {"Std":>7s}  {"Min":>7s}  {"Max":>7s}  {"Range":>7s}')
for iv in range(M_vars):
    print(f'   {featNames[iv]:<12s}  {raw_means[iv]:7.2f}  {raw_stds[iv]:7.2f}  '
          f'{raw_mins[iv]:7.2f}  {raw_maxs[iv]:7.2f}  {raw_ranges[iv]:7.2f}')

# Plot: Raw vs Autoscaled comparison
X_auto_preview = (X_full - raw_means) / raw_stds

fig_prepr, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.boxplot([X_full[:, i] for i in range(M_vars)], labels=featNames)
ax1.set_ylabel('Raw Value (area %)')
ax1.set_title('Raw Data')
ax1.grid(True, alpha=0.3)
plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')

ax2.boxplot([X_auto_preview[:, i] for i in range(M_vars)], labels=featNames)
ax2.set_ylabel('Autoscaled Value')
ax2.set_title('After Autoscaling')
ax2.grid(True, alpha=0.3)
plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')

fig_prepr.suptitle(f'Preprocessing: {preprocessing_method}', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '01b_preprocessing_comparison.png'))
plt.close(fig_prepr)
print('   Preprocessing plot saved.\n')


# %% ---- 2. EXPLORATORY DATA ANALYSIS (EDA) --------------------------------
print('[2] Exploratory Data Analysis...')

# ---- 2.1 Boxplot of raw variables ----
fig1, ax = plt.subplots(figsize=(9, 5))
ax.boxplot([X_full[:, i] for i in range(M_vars)], labels=featNames)
ax.set_ylabel('Area (%)')
ax.set_title('Distribution of Anthocyanin Variables')
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '01_raw_data_boxplot.png'))
plt.close(fig1)

# ---- 2.2 Boxplot per class ----
fig2, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()
for iv in range(M_vars):
    ax = axes[iv]
    data_groups = [X_full[y_full == ic + 1, iv] for ic in range(nClasses)]
    bp = ax.boxplot(data_groups, labels=classNames, patch_artist=True)
    for patch, color in zip(bp['boxes'], classColors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_title(featNames[iv], fontsize=10)
    ax.set_ylabel('Area (%)')
    ax.grid(True, alpha=0.3)
fig2.suptitle('Anthocyanin Distribution by Grape Variety', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '02_boxplot_by_class.png'))
plt.close(fig2)

# ---- 2.2b Boxplot per vintage ----
fig2b, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()
vintageColors_bp = ['#339933', '#CC4400']
for iv in range(M_vars):
    ax = axes[iv]
    data_groups = [X_full[annata == yr, iv] for yr in [2000, 2001]]
    bp = ax.boxplot(data_groups, labels=['2000', '2001'], patch_artist=True)
    for patch, col in zip(bp['boxes'], vintageColors_bp):
        patch.set_facecolor(col)
        patch.set_alpha(0.5)
    ax.set_title(featNames[iv], fontsize=10)
    ax.set_ylabel('Area (%)')
    ax.grid(True, alpha=0.3)
fig2b.suptitle('Anthocyanin Distribution by Vintage (2000 vs 2001)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '02b_boxplot_by_vintage.png'))
plt.close(fig2b)

# ---- 2.3 Correlation matrix ----
fig3, ax = plt.subplots(figsize=(6.5, 5.5))
corrMat = np.corrcoef(X_full, rowvar=False)
im = ax.imshow(corrMat, cmap='jet', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(M_vars))
ax.set_xticklabels(featNames, rotation=30, ha='right')
ax.set_yticks(range(M_vars))
ax.set_yticklabels(featNames)
ax.set_title('Correlation Matrix of Anthocyanin Variables')
for ii in range(M_vars):
    for jj in range(M_vars):
        ax.text(jj, ii, f'{corrMat[ii, jj]:.2f}', ha='center', va='center', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '03_correlation_matrix.png'))
plt.close(fig3)

# ---- 2.4 PCA on full dataset (autoscaled) ----
print('   Running PCA...')
mx = np.mean(X_full, axis=0)
sx = np.std(X_full, axis=0, ddof=1)
X_auto = (X_full - mx) / sx  # autoscaling

U_pca, S_diag_pca, Vt_pca = np.linalg.svd(X_auto, full_matrices=False)
nPC_max = min(N_total, M_vars)
eigenvalues = S_diag_pca ** 2 / (N_total - 1)
explainedVar = 100.0 * eigenvalues / np.sum(eigenvalues)
cumVar = np.cumsum(explainedVar)
scores_pca = U_pca * S_diag_pca  # N x M
loadings_pca = Vt_pca.T  # M x M

print('   Explained variance: ', end='')
for ipc in range(min(5, nPC_max)):
    print(f'PC{ipc+1}={explainedVar[ipc]:.1f}% ', end='')
print()

# ---- 2.5 Scree plot ----
fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.bar(range(1, nPC_max + 1), explainedVar[:nPC_max], color=[0.3, 0.5, 0.8])
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance (%)')
ax1.set_title('Scree Plot')
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, nPC_max + 1), cumVar[:nPC_max], '-o', linewidth=2, color=[0.3, 0.5, 0.8])
ax2.axhline(y=95, color='r', linestyle='--', linewidth=1.5)
ax2.text(nPC_max, 95, '95%', color='r', fontsize=9, va='bottom')
ax2.set_xlabel('Number of PCs')
ax2.set_ylabel('Cumulative Variance (%)')
ax2.set_title('Cumulative Explained Variance')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '04_pca_scree_plot.png'))
plt.close(fig4)

# ---- 2.6 Score plots (colored by variety) ----
fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
for ic in range(nClasses):
    idx = y_full == ic + 1
    ax1.scatter(scores_pca[idx, 0], scores_pca[idx, 1], s=40,
                c=[classColors[ic]], edgecolors='k', linewidths=0.3,
                label=classFullNames[ic])
ax1.set_xlabel(f'PC1 ({explainedVar[0]:.1f}%)')
ax1.set_ylabel(f'PC2 ({explainedVar[1]:.1f}%)')
ax1.set_title('PCA Score Plot: PC1 vs PC2 (by Variety)')
ax1.legend(fontsize=7, loc='best')
ax1.grid(True, alpha=0.3)

for ic in range(nClasses):
    idx = y_full == ic + 1
    ax2.scatter(scores_pca[idx, 0], scores_pca[idx, 2], s=40,
                c=[classColors[ic]], edgecolors='k', linewidths=0.3,
                label=classFullNames[ic])
ax2.set_xlabel(f'PC1 ({explainedVar[0]:.1f}%)')
ax2.set_ylabel(f'PC3 ({explainedVar[2]:.1f}%)')
ax2.set_title('PCA Score Plot: PC1 vs PC3 (by Variety)')
ax2.legend(fontsize=7, loc='best')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '05_pca_scores_variety.png'))
plt.close(fig5)

# ---- 2.6b Score plots colored by vintage ----
fig5v, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
vintageColors = np.array([[0.2, 0.6, 0.2], [0.8, 0.3, 0.1]])
vintageLabels = ['2000', '2001']
for iv_yr, yr in enumerate([2000, 2001]):
    idx = annata == yr
    ax1.scatter(scores_pca[idx, 0], scores_pca[idx, 1], s=40,
                c=[vintageColors[iv_yr]], edgecolors='k', linewidths=0.3,
                label=vintageLabels[iv_yr])
ax1.set_xlabel(f'PC1 ({explainedVar[0]:.1f}%)')
ax1.set_ylabel(f'PC2 ({explainedVar[1]:.1f}%)')
ax1.set_title('PCA Score Plot: PC1 vs PC2 (by Vintage)')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

for iv_yr, yr in enumerate([2000, 2001]):
    idx = annata == yr
    ax2.scatter(scores_pca[idx, 0], scores_pca[idx, 2], s=40,
                c=[vintageColors[iv_yr]], edgecolors='k', linewidths=0.3,
                label=vintageLabels[iv_yr])
ax2.set_xlabel(f'PC1 ({explainedVar[0]:.1f}%)')
ax2.set_ylabel(f'PC3 ({explainedVar[2]:.1f}%)')
ax2.set_title('PCA Score Plot: PC1 vs PC3 (by Vintage)')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '05b_pca_scores_vintage.png'))
plt.close(fig5v)

# ---- 2.7 3D Score plot ----
fig5b = plt.figure(figsize=(7, 6))
ax3d = fig5b.add_subplot(111, projection='3d')
for ic in range(nClasses):
    idx = y_full == ic + 1
    ax3d.scatter(scores_pca[idx, 0], scores_pca[idx, 1], scores_pca[idx, 2],
                 s=40, c=[classColors[ic]], edgecolors='k', linewidths=0.3,
                 label=classFullNames[ic])
ax3d.set_xlabel(f'PC1 ({explainedVar[0]:.1f}%)')
ax3d.set_ylabel(f'PC2 ({explainedVar[1]:.1f}%)')
ax3d.set_zlabel(f'PC3 ({explainedVar[2]:.1f}%)')
ax3d.set_title('PCA 3D Score Plot (by Variety)')
ax3d.legend(fontsize=7, loc='best')
ax3d.view_init(elev=25, azim=30)
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '06_pca_scores_3d.png'))
plt.close(fig5b)

# ---- 2.8 Loading plot ----
fig6, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
nLoadPC = min(3, nPC_max)
x_pos = np.arange(M_vars)
width = 0.25
for ipc in range(nLoadPC):
    ax1.bar(x_pos + ipc * width, loadings_pca[:, ipc], width, label=f'PC{ipc+1}')
ax1.set_xticks(x_pos + width)
ax1.set_xticklabels(featNames, rotation=30, ha='right')
ax1.set_ylabel('Loading Value')
ax1.set_title(f'PCA Loadings (PC1-PC{nLoadPC})')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

for iv in range(M_vars):
    ax2.plot([0, loadings_pca[iv, 0]], [0, loadings_pca[iv, 1]], '-', linewidth=1.5)
    ax2.text(loadings_pca[iv, 0] * 1.08, loadings_pca[iv, 1] * 1.08,
             featNames[iv], fontsize=9)
ax2.set_xlabel(f'PC1 Loading ({explainedVar[0]:.1f}%)')
ax2.set_ylabel(f'PC2 Loading ({explainedVar[1]:.1f}%)')
ax2.set_title('Loading Plot (PC1 vs PC2)')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')
theta = np.linspace(0, 2 * np.pi, 200)
ax2.plot(np.cos(theta), np.sin(theta), '--', color=[0.7, 0.7, 0.7])
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '07_pca_loadings.png'))
plt.close(fig6)

# ---- 2.9 Biplot (scores + loadings on same plot) ----
fig_biplot, ax = plt.subplots(figsize=(8, 6.5))
sc_norm = scores_pca[:, :2] / np.max(np.abs(scores_pca[:, :2]), axis=0)
for ic in range(nClasses):
    idx = y_full == ic + 1
    ax.scatter(sc_norm[idx, 0], sc_norm[idx, 1], s=30, c=[classColors[ic]],
               alpha=0.5, label=classFullNames[ic])
for iv in range(M_vars):
    ax.annotate('', xy=(loadings_pca[iv, 0], loadings_pca[iv, 1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='k', lw=2))
    ax.text(loadings_pca[iv, 0] * 1.12, loadings_pca[iv, 1] * 1.12,
            featNames[iv], fontsize=10, fontweight='bold', color=[0.1, 0.1, 0.1])
ax.set_xlabel(f'PC1 ({explainedVar[0]:.1f}%)')
ax.set_ylabel(f'PC2 ({explainedVar[1]:.1f}%)')
ax.set_title('PCA Biplot (Scores + Loadings)')
ax.legend(fontsize=7, loc='best')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '07b_pca_biplot.png'))
plt.close(fig_biplot)

print('   EDA plots saved.\n')


# %% ---- 3. TRAIN / TEST SPLIT (Stratified 70/30) --------------------------
print('[3] Stratified Train/Test Split (70/30)...')
trainIdx = np.zeros(N_total, dtype=bool)
testIdx = np.zeros(N_total, dtype=bool)

for ic in range(nClasses):
    class_idx = np.where(y_full == ic + 1)[0]
    nClass = len(class_idx)
    nTrain = round(0.7 * nClass)
    perm = np.random.permutation(class_idx)
    trainIdx[perm[:nTrain]] = True
    testIdx[perm[nTrain:]] = True

X_train = X_full[trainIdx, :]
y_train = y_full[trainIdx]
X_test = X_full[testIdx, :]
y_test = y_full[testIdx]

print(f'   Training set: {np.sum(trainIdx)} samples')
print(f'   Test set:     {np.sum(testIdx)} samples')
for ic in range(nClasses):
    print(f'   Class {ic+1} ({classNames[ic]}): Train={np.sum(y_train==ic+1)}, '
          f'Test={np.sum(y_test==ic+1)}')
print()


# %% ---- 4. SIMCA ANALYSIS -------------------------------------------------
print('[4] SIMCA Analysis...')
print('   Running cross-validation...')

maxPC_simca = min(5, M_vars - 1)
nSegCV = 5
conf_level = 0.95

N_tr = X_train.shape[0]

sens_cv = np.zeros((nClasses, maxPC_simca))
spec_cv = np.zeros((nClasses, maxPC_simca))
eff_cv = np.zeros((nClasses, maxPC_simca))

for npc in range(1, maxPC_simca + 1):
    ssmatrix = np.zeros((nClasses, nClasses))
    predclass_cv = np.zeros(N_tr, dtype=int)

    for seg in range(nSegCV):
        test_cv_idx = np.arange(seg, N_tr, nSegCV)
        train_cv_idx = np.setdiff1d(np.arange(N_tr), test_cv_idx)

        Xseg_tr = X_train[train_cv_idx, :]
        yseg_tr = y_train[train_cv_idx]
        Xseg_ts = X_train[test_cv_idx, :]
        yseg_ts = y_train[test_cv_idx]

        simca_seg = build_simca_model(Xseg_tr, yseg_tr,
                                       np.full(nClasses, npc), conf_level)
        pred_seg = predict_simca(Xseg_ts, yseg_ts, simca_seg)

        ssmatrix += pred_seg['SIMCA']['sensspec']
        predclass_cv[test_cv_idx] = pred_seg['SIMCA']['predclass']

    for ic in range(nClasses):
        nc_ic = np.sum(y_train == ic + 1)
        sens_cv[ic, npc - 1] = ssmatrix[ic, ic] / nc_ic
        others = [j for j in range(nClasses) if j != ic]
        nc_others = N_tr - nc_ic
        spec_cv[ic, npc - 1] = np.sum(ssmatrix[ic, others]) / nc_others

eff_cv = np.sqrt(sens_cv * spec_cv)

# ---- 4.1 Plot: Sensitivity, Specificity, Efficiency in CV ----
fig7, axes = plt.subplots(3, 1, figsize=(10, 7))
pcs_range = np.arange(1, maxPC_simca + 1)

for ic in range(nClasses):
    axes[0].plot(pcs_range, sens_cv[ic, :], '-o', linewidth=1.5,
                 color=classColors[ic], label=classFullNames[ic])
axes[0].set_title('SIMCA Cross-Validation: Sensitivity')
axes[0].set_xlabel('Number of PCs')
axes[0].set_ylabel('Sensitivity')
axes[0].legend(fontsize=7, loc='best')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1.05])

for ic in range(nClasses):
    axes[1].plot(pcs_range, spec_cv[ic, :], '-o', linewidth=1.5,
                 color=classColors[ic], label=classFullNames[ic])
axes[1].set_title('SIMCA Cross-Validation: Specificity')
axes[1].set_xlabel('Number of PCs')
axes[1].set_ylabel('Specificity')
axes[1].legend(fontsize=7, loc='best')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 1.05])

for ic in range(nClasses):
    axes[2].plot(pcs_range, eff_cv[ic, :], '-o', linewidth=1.5,
                 color=classColors[ic], label=classFullNames[ic])
axes[2].set_title('SIMCA Cross-Validation: Efficiency')
axes[2].set_xlabel('Number of PCs')
axes[2].set_ylabel('Efficiency = sqrt(Sens*Spec)')
axes[2].legend(fontsize=7, loc='best')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(os.path.join(plotDir, '08_simca_cv_metrics.png'))
plt.close(fig7)

# ---- 4.2 Select optimal PCs per class (max efficiency) ----
optPC_simca = np.zeros(nClasses, dtype=int)
for ic in range(nClasses):
    optPC_simca[ic] = np.argmax(eff_cv[ic, :]) + 1

print('   Optimal PCs per class (max CV efficiency):')
for ic in range(nClasses):
    pc_idx = optPC_simca[ic] - 1
    print(f'   Class {ic+1} ({classNames[ic]}): {optPC_simca[ic]} PCs '
          f'(Eff={eff_cv[ic, pc_idx]:.3f}, Sens={sens_cv[ic, pc_idx]:.3f}, '
          f'Spec={spec_cv[ic, pc_idx]:.3f})')

# ---- 4.3 Build final SIMCA model ----
print('   Building final SIMCA model...')
simca_final = build_simca_model(X_train, y_train, optPC_simca, conf_level)

# ---- 4.4 Predict training set ----
pred_train_simca = predict_simca(X_train, y_train, simca_final)
predclass_train_simca = pred_train_simca['SIMCA']['predclass']

# ---- 4.5 Predict test set ----
print('   Predicting test set...')
pred_test_simca = predict_simca(X_test, y_test, simca_final)
predclass_test_simca = pred_test_simca['SIMCA']['predclass']

# ---- 4.6 Compute and display metrics ----
print('\n   === SIMCA RESULTS ===')
print('   --- Training Set ---')
sens_tr_s, spec_tr_s, eff_tr_s, acc_tr_s = compute_class_metrics(
    y_train, predclass_train_simca, nClasses)
for ic in range(nClasses):
    print(f'   Class {ic+1} ({classNames[ic]}): Sens={sens_tr_s[ic]:.3f} '
          f'Spec={spec_tr_s[ic]:.3f} Eff={eff_tr_s[ic]:.3f}')
print(f'   Overall accuracy: {acc_tr_s*100:.1f}%')

print('   --- Test Set ---')
sens_ts_s, spec_ts_s, eff_ts_s, acc_ts_s = compute_class_metrics(
    y_test, predclass_test_simca, nClasses)
for ic in range(nClasses):
    print(f'   Class {ic+1} ({classNames[ic]}): Sens={sens_ts_s[ic]:.3f} '
          f'Spec={spec_ts_s[ic]:.3f} Eff={eff_ts_s[ic]:.3f}')
print(f'   Overall accuracy: {acc_ts_s*100:.1f}%\n')

# ---- 4.7 Plot: Score Distance vs Orthogonal Distance per class ----
for ic in range(nClasses):
    fig_sd, ax = plt.subplots(figsize=(8, 6))
    t2lim = simca_final['PCmodels'][ic]['critvals'][0]
    qlim = simca_final['PCmodels'][ic]['critvals'][1]

    tsq_tr = simca_final['PCmodels'][ic]['tsq']
    q_tr = simca_final['PCmodels'][ic]['qres']
    tsq_ts = pred_test_simca['PCmodels'][ic]['tsq']
    q_ts = pred_test_simca['PCmodels'][ic]['qres']

    for jc in range(nClasses):
        idx_tr = y_train == jc + 1
        ax.scatter(tsq_tr[idx_tr] / t2lim, q_tr[idx_tr] / max(qlim, 1e-16),
                   s=40, facecolors='none', edgecolors=classColors[jc], linewidths=1.2)
    for jc in range(nClasses):
        idx_ts = y_test == jc + 1
        ax.scatter(tsq_ts[idx_ts] / t2lim, q_ts[idx_ts] / max(qlim, 1e-16),
                   s=60, c=[classColors[jc]], marker='D', edgecolors='k')

    theta_c = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.sqrt(2) * np.cos(theta_c), np.sqrt(2) * np.sin(theta_c),
            '-r', linewidth=1.5)

    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_xlabel('Score Distance (T² / T²_lim)')
    ax.set_ylabel('Orthogonal Distance (Q / Q_lim)')
    ax.set_title(f'SIMCA: Class {ic+1} ({classFullNames[ic]}) | {optPC_simca[ic]} PCs')

    legend_entries = ([f'{classNames[jc]} (train)' for jc in range(nClasses)] +
                      [f'{classNames[jc]} (test)' for jc in range(nClasses)])
    ax.legend(legend_entries, fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plotDir, f'09_simca_SDvsOD_class{ic+1}.png'))
    plt.close(fig_sd)

# ---- 4.8 Plot: Confusion matrices ----
fig_cm1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
plot_confusion_matrix_custom(ax1, y_train, predclass_train_simca, classNames,
                             'SIMCA - Training Set')
plot_confusion_matrix_custom(ax2, y_test, predclass_test_simca, classNames,
                             'SIMCA - Test Set')
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '10_simca_confusion_matrices.png'))
plt.close(fig_cm1)

# ---- 4.9 Discriminant Power ----
print('   Computing discriminant power...')
dpow = compute_discriminant_power(X_train, y_train, simca_final)
fig_dp, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(M_vars), dpow, color=[0.3, 0.6, 0.8])
p95 = np.percentile(dpow, 95)
p99 = np.percentile(dpow, 99)
ax.axhline(y=p95, color='g', linestyle='--', linewidth=1.5)
ax.text(M_vars - 0.5, p95, '95th pctl', color='g', fontsize=8, va='bottom')
ax.axhline(y=p99, color='r', linestyle='--', linewidth=1.5)
ax.text(M_vars - 0.5, p99, '99th pctl', color='r', fontsize=8, va='bottom')
ax.axhline(y=np.mean(dpow), color='k', linestyle=':', linewidth=1.2)
ax.text(M_vars - 0.5, np.mean(dpow), 'Mean', color='k', fontsize=8, va='bottom')
ax.set_xticks(range(M_vars))
ax.set_xticklabels(featNames, rotation=30, ha='right')
ax.set_xlabel('Variable')
ax.set_ylabel('Discriminant Power')
ax.set_title('SIMCA Discriminant Power')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '11_simca_discriminant_power.png'))
plt.close(fig_dp)

# ---- 4.10 Loadings per class ----
maxPC_plot = int(np.max(optPC_simca))
fig_ld, axes = plt.subplots(nClasses, maxPC_plot, figsize=(12, 8),
                             squeeze=False)
for ic in range(nClasses):
    for ipc in range(maxPC_plot):
        ax = axes[ic, ipc]
        if ipc < optPC_simca[ic]:
            ax.bar(range(M_vars), simca_final['PCmodels'][ic]['loads'][:, ipc],
                   color=classColors[ic])
            ax.set_xticks(range(M_vars))
            ax.set_xticklabels(featNames, fontsize=6, rotation=45, ha='right')
            ax.set_title(f'C{ic+1} PC{ipc+1}', fontsize=9)
            ax.set_ylabel('Loading')
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')
fig_ld.suptitle('SIMCA: Loadings per Class', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '12_simca_loadings.png'))
plt.close(fig_ld)

# ---- 4.11 SIMCA Classification Summary Table ----
fig_sum_s, ax = plt.subplots(figsize=(9, 3))
ax.axis('off')
table_data = []
for ic in range(nClasses):
    table_data.append([
        classFullNames[ic], str(optPC_simca[ic]),
        f'{sens_tr_s[ic]:.3f}', f'{spec_tr_s[ic]:.3f}', f'{eff_tr_s[ic]:.3f}',
        f'{sens_ts_s[ic]:.3f}', f'{spec_ts_s[ic]:.3f}', f'{eff_ts_s[ic]:.3f}',
    ])
col_labels = ['Class', 'nPCs', 'Sens_Train', 'Spec_Train', 'Eff_Train',
              'Sens_Test', 'Spec_Test', 'Eff_Test']
tab = ax.table(cellText=table_data, colLabels=col_labels, loc='center',
               cellLoc='center')
tab.auto_set_font_size(False)
tab.set_fontsize(9)
tab.scale(1.2, 1.5)
ax.set_title('SIMCA Summary', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '13_simca_summary_table.png'))
plt.close(fig_sum_s)

print('   SIMCA plots saved.\n')


# %% ---- 5. PLS-DA ANALYSIS ------------------------------------------------
print('[5] PLS-DA Analysis...')

# Create dummy Y matrix (one-hot encoding)
Y_train = np.zeros((X_train.shape[0], nClasses))
for ic in range(nClasses):
    Y_train[y_train == ic + 1, ic] = 1
Y_test_mat = np.zeros((X_test.shape[0], nClasses))
for ic in range(nClasses):
    Y_test_mat[y_test == ic + 1, ic] = 1

# Autoscale X
mx_tr = np.mean(X_train, axis=0)
sx_tr = np.std(X_train, axis=0, ddof=1)
X_train_sc = (X_train - mx_tr) / sx_tr
X_test_sc = (X_test - mx_tr) / sx_tr

# Mean center Y
my_tr = np.mean(Y_train, axis=0)
Y_train_mc = Y_train - my_tr

maxLV = min(M_vars, 6)

# ---- 5.1 Cross-validation (venetian blinds, 5 segments) ----
print('   Running PLS-DA cross-validation...')
nSegPLS = 5
err_cv_pls = np.zeros((nClasses, maxLV))
corr_cv_pls = np.zeros((nClasses, maxLV))
misclass_cv_pls = np.zeros(maxLV, dtype=int)
rmsecv = np.zeros((nClasses, maxLV))

for nlv in range(1, maxLV + 1):
    pred_cv_all = np.zeros((X_train.shape[0], nClasses))

    for seg in range(nSegPLS):
        cv_ts_idx = np.arange(seg, X_train.shape[0], nSegPLS)
        cv_tr_idx = np.setdiff1d(np.arange(X_train.shape[0]), cv_ts_idx)

        Xcv_tr = X_train[cv_tr_idx, :]
        Ycv_tr = Y_train[cv_tr_idx, :]
        Xcv_ts = X_train[cv_ts_idx, :]

        mx_cv = np.mean(Xcv_tr, axis=0)
        sx_cv = np.std(Xcv_tr, axis=0, ddof=1)
        sx_cv[sx_cv == 0] = 1.0
        my_cv = np.mean(Ycv_tr, axis=0)

        Xcv_tr_sc = (Xcv_tr - mx_cv) / sx_cv
        Xcv_ts_sc = (Xcv_ts - mx_cv) / sx_cv
        Ycv_tr_mc = Ycv_tr - my_cv

        plsmod = nipals_pls2(Xcv_tr_sc, Ycv_tr_mc, nlv)
        Ypred_cv = Xcv_ts_sc @ plsmod['Bpls'] + my_cv
        pred_cv_all[cv_ts_idx, :] = Ypred_cv

    pred_class_cv = np.argmax(pred_cv_all, axis=1) + 1

    for ic in range(nClasses):
        idx_ic = y_train == ic + 1
        n_ic = np.sum(idx_ic)
        err_cv_pls[ic, nlv - 1] = np.sum(pred_class_cv[idx_ic] != ic + 1)
        corr_cv_pls[ic, nlv - 1] = 100.0 * (n_ic - err_cv_pls[ic, nlv - 1]) / n_ic
        rmsecv[ic, nlv - 1] = np.sqrt(np.mean(
            (pred_cv_all[idx_ic, ic] - Y_train[idx_ic, ic]) ** 2))
    misclass_cv_pls[nlv - 1] = np.sum(pred_class_cv != y_train)

# ---- 5.2 Plot: CV errors and correct classification vs LVs ----
fig_plscv, axes = plt.subplots(2, 2, figsize=(10, 7))
lv_range = np.arange(1, maxLV + 1)

for ic in range(nClasses):
    axes[0, 0].plot(lv_range, corr_cv_pls[ic, :], '-o', linewidth=1.5,
                    color=classColors[ic], label=classFullNames[ic])
axes[0, 0].set_title('PLS-DA CV: % Correct Classification')
axes[0, 0].set_xlabel('Latent Variables')
axes[0, 0].set_ylabel('% Correct')
axes[0, 0].legend(fontsize=7, loc='best')
axes[0, 0].grid(True, alpha=0.3)

for ic in range(nClasses):
    axes[1, 0].plot(lv_range, err_cv_pls[ic, :], '-o', linewidth=1.5,
                    color=classColors[ic], label=classFullNames[ic])
axes[1, 0].set_title('PLS-DA CV: Misclassified Samples')
axes[1, 0].set_xlabel('Latent Variables')
axes[1, 0].set_ylabel('# Misclassified')
axes[1, 0].legend(fontsize=7, loc='best')
axes[1, 0].grid(True, alpha=0.3)

axes[0, 1].plot(lv_range, np.mean(corr_cv_pls, axis=0), '-o', linewidth=2,
                color=[0.2, 0.4, 0.8])
axes[0, 1].set_title('Mean % Correct (higher = better)')
axes[0, 1].set_xlabel('Latent Variables')
axes[0, 1].set_ylabel('Mean % Correct')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 1].plot(lv_range, misclass_cv_pls, '-o', linewidth=2, color=[0.8, 0.2, 0.2])
axes[1, 1].set_title('Total Misclassified (lower = better)')
axes[1, 1].set_xlabel('Latent Variables')
axes[1, 1].set_ylabel('Total Misclassified')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plotDir, '14_plsda_cv_error.png'))
plt.close(fig_plscv)

# ---- 5.2b Plot: RMSECV ----
fig_rmsecv, ax = plt.subplots(figsize=(8, 4))
for ic in range(nClasses):
    ax.plot(lv_range, rmsecv[ic, :], '-o', linewidth=1.5,
            color=classColors[ic], label=classFullNames[ic])
ax.set_title('PLS-DA: RMSECV per Class')
ax.set_xlabel('Latent Variables')
ax.set_ylabel('RMSECV')
ax.legend(fontsize=7, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '15_plsda_rmsecv.png'))
plt.close(fig_rmsecv)

# ---- 5.3 Select optimal number of LVs ----
optLV = int(np.argmin(misclass_cv_pls) + 1)
if optLV < 2:
    optLV = 2
print(f'   Optimal number of LVs: {optLV} '
      f'(CV misclassified: {misclass_cv_pls[optLV-1]} / {X_train.shape[0]})')

# ---- 5.4 Build final PLS-DA model ----
print(f'   Building final PLS-DA model with {optLV} LVs...')
plsda_final = nipals_pls2(X_train_sc, Y_train_mc, optLV)

# Predictions on training set
Ypred_train = X_train_sc @ plsda_final['Bpls'] + my_tr
predclass_train_pls = np.argmax(Ypred_train, axis=1) + 1

# Predictions on test set
Ypred_test = X_test_sc @ plsda_final['Bpls'] + my_tr
predclass_test_pls = np.argmax(Ypred_test, axis=1) + 1

# RMSEP
rmsep_pls = np.zeros(nClasses)
for ic in range(nClasses):
    idx_ic = y_test == ic + 1
    rmsep_pls[ic] = np.sqrt(np.mean(
        (Ypred_test[idx_ic, ic] - Y_test_mat[idx_ic, ic]) ** 2))

# ---- 5.5 Display results ----
print('\n   === PLS-DA RESULTS ===')
print('   --- Training Set ---')
sens_tr_p, spec_tr_p, eff_tr_p, acc_tr_p = compute_class_metrics(
    y_train, predclass_train_pls, nClasses)
for ic in range(nClasses):
    print(f'   Class {ic+1} ({classNames[ic]}): Sens={sens_tr_p[ic]:.3f} '
          f'Spec={spec_tr_p[ic]:.3f} Eff={eff_tr_p[ic]:.3f}')
print(f'   Overall accuracy: {acc_tr_p*100:.1f}%')

print('   --- Test Set ---')
sens_ts_p, spec_ts_p, eff_ts_p, acc_ts_p = compute_class_metrics(
    y_test, predclass_test_pls, nClasses)
for ic in range(nClasses):
    print(f'   Class {ic+1} ({classNames[ic]}): Sens={sens_ts_p[ic]:.3f} '
          f'Spec={spec_ts_p[ic]:.3f} Eff={eff_ts_p[ic]:.3f} '
          f'RMSEP={rmsep_pls[ic]:.4f}')
print(f'   Overall accuracy: {acc_ts_p*100:.1f}%\n')

# ---- 5.6 Plot: Y predicted vs samples (training) ----
fig_ytr, axes = plt.subplots(nClasses, 1, figsize=(11, 9))
n_tr = X_train.shape[0]
for ic in range(nClasses):
    ax = axes[ic]
    ax.plot(range(n_tr), Ypred_train[:, ic], '-', color=[0.7, 0.7, 0.7], linewidth=0.5)
    for jc in range(nClasses):
        idx = y_train == jc + 1
        if jc == ic:
            ax.scatter(np.where(idx)[0], Ypred_train[idx, ic], s=25,
                       c=[classColors[jc]], zorder=5)
        else:
            ax.scatter(np.where(idx)[0], Ypred_train[idx, ic], s=15,
                       c=[classColors[jc]], marker='o', facecolors='none',
                       edgecolors=classColors[jc], linewidths=0.8)
    ax.axhline(y=0.5, color='r', linestyle='--', linewidth=1)
    ax.set_ylabel(f'Y_pred C{ic+1}')
    ax.set_title(f'Predicted Y for {classFullNames[ic]} (Training)', fontsize=9)
    ax.grid(True, alpha=0.3)
    if ic == nClasses - 1:
        ax.set_xlabel('Sample #')
fig_ytr.suptitle(f'PLS-DA | Preprocessing: Autoscaling | LVs = {optLV} | Y: Mean-Centered Dummy',
                 fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '16_plsda_ypred_train.png'))
plt.close(fig_ytr)

# ---- 5.7 Plot: Y predicted vs samples (test) ----
fig_yts, axes = plt.subplots(nClasses, 1, figsize=(11, 9))
n_ts = X_test.shape[0]
for ic in range(nClasses):
    ax = axes[ic]
    ax.plot(range(n_ts), Ypred_test[:, ic], '-', color=[0.7, 0.7, 0.7], linewidth=0.5)
    for jc in range(nClasses):
        idx = y_test == jc + 1
        if jc == ic:
            ax.scatter(np.where(idx)[0], Ypred_test[idx, ic], s=25,
                       c=[classColors[jc]], zorder=5)
        else:
            ax.scatter(np.where(idx)[0], Ypred_test[idx, ic], s=15,
                       c=[classColors[jc]], marker='o', facecolors='none',
                       edgecolors=classColors[jc], linewidths=0.8)
    ax.axhline(y=0.5, color='r', linestyle='--', linewidth=1)
    ax.set_ylabel(f'Y_pred C{ic+1}')
    ax.set_title(f'Predicted Y for {classFullNames[ic]} (Test)', fontsize=9)
    ax.grid(True, alpha=0.3)
    if ic == nClasses - 1:
        ax.set_xlabel('Sample #')
fig_yts.suptitle(f'PLS-DA | Preprocessing: Autoscaling | LVs = {optLV} | Y: Mean-Centered Dummy',
                 fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '17_plsda_ypred_test.png'))
plt.close(fig_yts)

# ---- 5.8 Plot: Confusion matrices (PLS-DA) ----
fig_cm2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
plot_confusion_matrix_custom(ax1, y_train, predclass_train_pls, classNames,
                             'PLS-DA - Training Set')
plot_confusion_matrix_custom(ax2, y_test, predclass_test_pls, classNames,
                             'PLS-DA - Test Set')
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '18_plsda_confusion_matrices.png'))
plt.close(fig_cm2)

# ---- 5.9 Plot: PLS Scores (LV1 vs LV2) ----
fig_plssc, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
for ic in range(nClasses):
    idx = y_train == ic + 1
    ax1.scatter(plsda_final['T'][idx, 0], plsda_final['T'][idx, 1], s=40,
                c=[classColors[ic]], edgecolors='k', linewidths=0.3,
                label=classFullNames[ic])
ax1.set_xlabel('LV1 Score')
ax1.set_ylabel('LV2 Score')
ax1.set_title('PLS-DA Scores: Training Set')
ax1.legend(fontsize=7, loc='best')
ax1.grid(True, alpha=0.3)

# Project test data
PtW = plsda_final['P'].T @ plsda_final['W']
T_test_pls = X_test_sc @ plsda_final['W'] @ np.linalg.inv(PtW)
for ic in range(nClasses):
    idx = y_test == ic + 1
    ax2.scatter(T_test_pls[idx, 0], T_test_pls[idx, 1], s=40,
                c=[classColors[ic]], edgecolors='k', linewidths=0.3,
                label=classFullNames[ic])
ax2.set_xlabel('LV1 Score')
ax2.set_ylabel('LV2 Score')
ax2.set_title('PLS-DA Scores: Test Set')
ax2.legend(fontsize=7, loc='best')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '19_plsda_scores.png'))
plt.close(fig_plssc)

# ---- 5.10 Plot: Regression coefficients ----
fig_bpls, axes = plt.subplots(1, nClasses, figsize=(10, 5))
for ic in range(nClasses):
    ax = axes[ic]
    ax.bar(range(M_vars), plsda_final['Bpls'][:, ic], color=classColors[ic])
    ax.set_xticks(range(M_vars))
    ax.set_xticklabels(featNames, fontsize=7, rotation=45, ha='right')
    ax.set_title(f'B_PLS - {classNames[ic]}', fontsize=9)
    ax.set_ylabel('Coefficient')
    ax.grid(True, alpha=0.3)
fig_bpls.suptitle('PLS-DA Regression Coefficients', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '20_plsda_regression_coefficients.png'))
plt.close(fig_bpls)

# ---- 5.11 VIP scores ----
VIP = compute_vip(plsda_final, X_train_sc, Y_train_mc)

fig_vip, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(M_vars), VIP, color=[0.4, 0.6, 0.3])
ax.axhline(y=1, color='r', linestyle='--', linewidth=1.5)
ax.text(M_vars - 0.5, 1.02, 'VIP=1', color='r', fontsize=9)
ax.axhline(y=0.8, color='k', linestyle=':', linewidth=1)
ax.text(M_vars - 0.5, 0.82, 'VIP=0.8', color='k', fontsize=9)
ax.set_xticks(range(M_vars))
ax.set_xticklabels(featNames, rotation=30, ha='right')
ax.set_xlabel('Variable')
ax.set_ylabel('VIP Score')
ax.set_title('PLS-DA: Variable Importance in Projection (VIP)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '21_plsda_vip.png'))
plt.close(fig_vip)

# ---- 5.12 PLS-DA Summary Table ----
fig_sum_p, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')
table_data_p = []
for ic in range(nClasses):
    table_data_p.append([
        classFullNames[ic],
        f'{sens_tr_p[ic]:.3f}', f'{spec_tr_p[ic]:.3f}', f'{eff_tr_p[ic]:.3f}',
        f'{sens_ts_p[ic]:.3f}', f'{spec_ts_p[ic]:.3f}', f'{eff_ts_p[ic]:.3f}',
        f'{rmsep_pls[ic]:.4f}',
    ])
col_labels_p = ['Class', 'Sens_Train', 'Spec_Train', 'Eff_Train',
                'Sens_Test', 'Spec_Test', 'Eff_Test', 'RMSEP']
tab_p = ax.table(cellText=table_data_p, colLabels=col_labels_p, loc='center',
                 cellLoc='center')
tab_p.auto_set_font_size(False)
tab_p.set_fontsize(9)
tab_p.scale(1.2, 1.5)
ax.set_title(f'PLS-DA Summary ({optLV} LVs)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '22_plsda_summary_table.png'))
plt.close(fig_sum_p)

print('   PLS-DA plots saved.\n')


# %% ---- 5b. APPLICABILITY DOMAIN ------------------------------------------
print('[5b] Applicability Domain Analysis...')
print('     Checking if test samples fall within the training domain.')

# Leverage-based Applicability Domain
XtX_inv = np.linalg.inv(X_train_sc.T @ X_train_sc)
h_train = np.array([X_train_sc[i, :] @ XtX_inv @ X_train_sc[i, :].T
                     for i in range(X_train_sc.shape[0])])
h_test = np.array([X_test_sc[i, :] @ XtX_inv @ X_test_sc[i, :].T
                    for i in range(X_test_sc.shape[0])])

h_star = 3 * (M_vars + 1) / X_train_sc.shape[0]

# Standardized residuals for Williams plot
Ypred_train_full = X_train_sc @ plsda_final['Bpls'] + my_tr
Ypred_test_full = X_test_sc @ plsda_final['Bpls'] + my_tr

res_train = np.array([Y_train[i, int(y_train[i]) - 1] - Ypred_train_full[i, int(y_train[i]) - 1]
                       for i in range(len(y_train))])
res_test = np.array([Y_test_mat[i, int(y_test[i]) - 1] - Ypred_test_full[i, int(y_test[i]) - 1]
                      for i in range(len(y_test))])

sigma_res = np.std(res_train, ddof=1)
std_res_train = res_train / sigma_res
std_res_test = res_test / sigma_res

n_test_outside = np.sum(h_test > h_star)
n_test_total = len(h_test)
print(f'   Leverage threshold (h*): {h_star:.4f}')
print(f'   Test samples outside AD: {n_test_outside} / {n_test_total} '
      f'({100 * n_test_outside / n_test_total:.1f}%)')

# Williams Plot
fig_ad, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
for ic in range(nClasses):
    idx_tr = y_train == ic + 1
    ax1.scatter(h_train[idx_tr], std_res_train[idx_tr], s=30,
                facecolors='none', edgecolors=classColors[ic], linewidths=1)
for ic in range(nClasses):
    idx_ts = y_test == ic + 1
    ax1.scatter(h_test[idx_ts], std_res_test[idx_ts], s=50,
                c=[classColors[ic]], marker='D', edgecolors='k')
ax1.axvline(x=h_star, color='r', linestyle='--', linewidth=1.5)
ax1.axhline(y=3, color='k', linestyle=':', linewidth=1)
ax1.axhline(y=-3, color='k', linestyle=':', linewidth=1)
ax1.text(h_star * 1.02, 3.5, f'h*={h_star:.3f}', color='r', fontsize=8)
ax1.set_xlabel('Leverage (h_i)')
ax1.set_ylabel('Standardized Residual')
ax1.set_title('Williams Plot - Applicability Domain')
legend_ad = ([f'{classNames[jc]} (train)' for jc in range(nClasses)] +
             [f'{classNames[jc]} (test)' for jc in range(nClasses)])
ax1.legend(legend_ad, fontsize=7, loc='best')
ax1.grid(True, alpha=0.3)

# Leverage bar chart for test samples
bar_colors_test = [classColors[int(y_test[i]) - 1] for i in range(n_test_total)]
ax2.bar(range(n_test_total), h_test, color=bar_colors_test)
ax2.axhline(y=h_star, color='r', linestyle='--', linewidth=1.5)
ax2.text(n_test_total * 0.7, h_star * 1.1, f'h*={h_star:.3f}', color='r', fontsize=9)
ax2.set_xlabel('Test Sample Index')
ax2.set_ylabel('Leverage')
ax2.set_title(f'Test Set Leverage ({n_test_outside}/{n_test_total} outside AD)')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '24_applicability_domain.png'))
plt.close(fig_ad)

# Hotelling T2 + Q for global PCA-based AD
nPC_ad = optLV
U_ad, S_ad, Vt_ad = np.linalg.svd(X_train_sc, full_matrices=False)
P_ad = Vt_ad[:nPC_ad, :].T
lambda_ad = S_ad ** 2 / (X_train_sc.shape[0] - 1)

T_train_ad = X_train_sc @ P_ad
lam_diag_inv = np.diag(1.0 / lambda_ad[:nPC_ad])
T2_train = np.sum((T_train_ad @ lam_diag_inv) * T_train_ad, axis=1)
E_train_ad = X_train_sc - T_train_ad @ P_ad.T
Q_train = np.sum(E_train_ad ** 2, axis=1)

T_test_ad = X_test_sc @ P_ad
T2_test = np.sum((T_test_ad @ lam_diag_inv) * T_test_ad, axis=1)
E_test_ad = X_test_sc - T_test_ad @ P_ad.T
Q_test = np.sum(E_test_ad ** 2, axis=1)

# Limits
N_ad = X_train_sc.shape[0]
A_ad = nPC_ad
try:
    F_ad = f_dist.ppf(0.95, A_ad, N_ad - A_ad)
except:
    F_ad = 3.0
T2_lim = A_ad * (N_ad - 1) / (N_ad - A_ad) * F_ad

if len(lambda_ad) > nPC_ad:
    th1 = np.sum(lambda_ad[nPC_ad:])
    th2 = np.sum(lambda_ad[nPC_ad:] ** 2)
    th3 = np.sum(lambda_ad[nPC_ad:] ** 3)
    h0_q = 1 - 2 * th1 * th3 / (3 * th2 ** 2)
    if h0_q < 0.001:
        h0_q = 0.001
    ca_q = np.sqrt(2) * erfinv(0.9)
    Q_lim = th1 * (1 + ca_q * np.sqrt(2 * th2 * h0_q ** 2) / th1 +
                    th2 * h0_q * (h0_q - 1) / th1 ** 2) ** (1 / h0_q)
else:
    Q_lim = np.max(Q_train) * 1.5

fig_ad2, ax = plt.subplots(figsize=(8, 6))
for ic in range(nClasses):
    idx_tr = y_train == ic + 1
    ax.scatter(T2_train[idx_tr] / T2_lim, Q_train[idx_tr] / Q_lim,
               s=30, facecolors='none', edgecolors=classColors[ic], linewidths=1)
for ic in range(nClasses):
    idx_ts = y_test == ic + 1
    ax.scatter(T2_test[idx_ts] / T2_lim, Q_test[idx_ts] / Q_lim,
               s=60, c=[classColors[ic]], marker='D', edgecolors='k')
theta_el = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(theta_el), np.sin(theta_el), '-r', linewidth=1.5)
ax.axvline(x=1, color='r', linestyle=':', linewidth=1)
ax.axhline(y=1, color='r', linestyle=':', linewidth=1)
ax.set_xlabel('T² / T²_lim')
ax.set_ylabel('Q / Q_lim')
ax.set_title(f'Applicability Domain: T² vs Q (PCA {nPC_ad} PCs)')
legend_ad2 = ([f'{classNames[jc]} (train)' for jc in range(nClasses)] +
              [f'{classNames[jc]} (test)' for jc in range(nClasses)])
ax.legend(legend_ad2, fontsize=7, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '25_applicability_domain_T2Q.png'))
plt.close(fig_ad2)

n_outside_T2Q = np.sum((T2_test / T2_lim > 1) | (Q_test / Q_lim > 1))
print(f'   Test samples outside T2/Q domain: {n_outside_T2Q} / {n_test_total} '
      f'({100 * n_outside_T2Q / n_test_total:.1f}%)')
print('   Applicability Domain plots saved.\n')


# %% ---- 6. COMPARISON SUMMARY ---------------------------------------------
print('[6] Final Comparison...\n')
print(f'   {"":25s}  SIMCA     PLS-DA')
print(f'   {"":25s}  --------  --------')
print(f'   {"Train Accuracy":25s}  {acc_tr_s*100:.1f}%     {acc_tr_p*100:.1f}%')
print(f'   {"Test Accuracy":25s}  {acc_ts_s*100:.1f}%     {acc_ts_p*100:.1f}%')
pc_str = ','.join([f'{pc}PC' for pc in optPC_simca])
print(f'   {"Complexity":25s}  {pc_str}  {optLV}LV')

# Final comparison figure
fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
x_pos = np.arange(nClasses)
width = 0.35

bars1 = ax1.bar(x_pos - width / 2, sens_ts_s, width, label='SIMCA',
                color=[0.3, 0.5, 0.8])
bars2 = ax1.bar(x_pos + width / 2, sens_ts_p, width, label='PLS-DA',
                color=[0.8, 0.4, 0.3])
ax1.set_xticks(x_pos)
ax1.set_xticklabels(classNames)
ax1.set_ylabel('Sensitivity')
ax1.set_title('Test Set Sensitivity')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1.1])

bars3 = ax2.bar(x_pos - width / 2, spec_ts_s, width, label='SIMCA',
                color=[0.3, 0.5, 0.8])
bars4 = ax2.bar(x_pos + width / 2, spec_ts_p, width, label='PLS-DA',
                color=[0.8, 0.4, 0.3])
ax2.set_xticks(x_pos)
ax2.set_xticklabels(classNames)
ax2.set_ylabel('Specificity')
ax2.set_title('Test Set Specificity')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1.1])

fig_comp.suptitle('SIMCA vs PLS-DA Comparison', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(plotDir, '23_comparison_simca_plsda.png'))
plt.close(fig_comp)

print('\n' + '=' * 60)
print('  ANALYSIS COMPLETE')
print(f'  All plots saved in: {plotDir}')
print('=' * 60)
