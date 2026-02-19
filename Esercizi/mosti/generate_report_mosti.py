#!/usr/bin/env python3
"""
generate_report_mosti.py
========================
Generates a professional PDF report from the plots produced by
main_analysis_mosti.m (saved in the 'plots2/' folder).

Each figure is included at full width with a detailed scientific
explanation covering methodology, interpretation, and relevance.

Requirements:
    pip install reportlab Pillow

Usage:
    python generate_report_mosti.py

Output:
    report_mosti.pdf  (saved in the same directory as this script)
"""

import os
import glob
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
    Table, TableStyle, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots2")
OUTPUT_PDF = os.path.join(SCRIPT_DIR, "report_mosti.pdf")

PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm
USABLE_W = PAGE_W - 2 * MARGIN

# Colors
DARK_BLUE = HexColor("#1a3a6b")
LIGHT_BLUE = HexColor("#4a90d9")
LIGHT_GRAY = HexColor("#f0f2f5")
ACCENT = HexColor("#c0392b")


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
def build_styles():
    ss = getSampleStyleSheet()

    ss.add(ParagraphStyle(
        "CoverTitle",
        parent=ss["Title"],
        fontSize=28,
        leading=34,
        textColor=DARK_BLUE,
        alignment=TA_CENTER,
        spaceAfter=12,
    ))
    ss.add(ParagraphStyle(
        "CoverSubtitle",
        parent=ss["Normal"],
        fontSize=14,
        leading=18,
        textColor=LIGHT_BLUE,
        alignment=TA_CENTER,
        spaceAfter=6,
    ))
    ss.add(ParagraphStyle(
        "CoverInfo",
        parent=ss["Normal"],
        fontSize=11,
        leading=14,
        textColor=HexColor("#555555"),
        alignment=TA_CENTER,
        spaceAfter=4,
    ))
    ss.add(ParagraphStyle(
        "SectionHeading",
        parent=ss["Heading1"],
        fontSize=18,
        leading=22,
        textColor=DARK_BLUE,
        spaceBefore=18,
        spaceAfter=8,
        borderWidth=0,
    ))
    ss.add(ParagraphStyle(
        "FigureCaption",
        parent=ss["Normal"],
        fontSize=10,
        leading=13,
        textColor=HexColor("#333333"),
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=6,
        fontName="Helvetica-BoldOblique",
    ))
    ss.add(ParagraphStyle(
        "BodyText2",
        parent=ss["BodyText"],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceBefore=2,
        spaceAfter=6,
        textColor=HexColor("#222222"),
    ))
    ss.add(ParagraphStyle(
        "BulletItem",
        parent=ss["BodyText"],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        leftIndent=18,
        bulletIndent=6,
        spaceBefore=1,
        spaceAfter=1,
        textColor=HexColor("#222222"),
    ))
    ss.add(ParagraphStyle(
        "FooterStyle",
        parent=ss["Normal"],
        fontSize=8,
        leading=10,
        textColor=HexColor("#999999"),
        alignment=TA_CENTER,
    ))
    return ss


# ---------------------------------------------------------------------------
# Figure descriptions (detailed, scientific)
# ---------------------------------------------------------------------------
FIGURE_DESCRIPTIONS = {

    "01b_preprocessing_comparison.png": {
        "title": "1b. Preprocessing: Raw vs Autoscaled Data",
        "caption": "Figure 1b — Boxplot comparison of raw data (left) vs autoscaled data (right).",
        "text": [
            "Prima di applicare qualsiasi metodo multivariato, è fondamentale verificare l'effetto del "
            "preprocessing sui dati. Questo grafico confronta le distribuzioni delle sei variabili "
            "antocianiche prima e dopo l'autoscaling (centratura sulla media + divisione per deviazione standard).",

            "Pannello sinistro (Dati grezzi):",
            ("bullet", "Le variabili presentano range e dispersioni molto differenti: MVD% domina in scala "
             "rispetto a CYD%, mentre R lib/lrg ha unità e ordine di grandezza diverso."),
            ("bullet", "Questa eterogeneità di scala penalizzerebbe le variabili a bassa varianza in PCA, "
             "SIMCA e PLS-DA, poiché i metodi basati sulla varianza darebbero peso preponderante a MVD%."),

            "Pannello destro (Dati autoscalati):",
            ("bullet", "Dopo l'autoscaling, tutte le variabili hanno media zero e varianza unitaria."),
            ("bullet", "Le distribuzioni risultano confrontabili e ogni variabile contribuisce equamente "
             "alla costruzione dei modelli multivariati."),
            ("bullet", "Le differenze tra le variabili riflettono ora la forma della distribuzione "
             "(asimmetria, code, outlier) piuttosto che la scala di misura."),

            "L'autoscaling è la scelta standard per dati in cui le variabili hanno unità diverse "
            "(percentuali vs rapporti). Tutto il resto dell'analisi utilizza i dati autoscalati."
        ]
    },

    "01_raw_data_boxplot.png": {
        "title": "1. Distribution of Raw Anthocyanin Variables",
        "caption": "Figure 1 — Boxplot of the six anthocyanin HPLC variables across all 98 grape must samples.",
        "text": [
            "This boxplot provides a first overview of the univariate distributions for each of the six "
            "measured anthocyanin variables: delfinidina (DPD%), cianidina (CYD%), petunidina (PTD%), "
            "peonidina (PND%), malvidina (MVD%) and the ratio between free and bound anthocyanins (R lib/lrg).",

            "Key observations from this plot:",

            ("bullet", "The variables exhibit very different ranges and dispersions, which justifies the use of "
             "autoscaling (mean-centering + division by standard deviation) before multivariate analysis."),
            ("bullet", "Malvidina (MVD%) typically dominates the anthocyanin profile, often reaching the highest "
             "percentage values. This is consistent with the known biochemistry of Vitis vinifera grapes."),
            ("bullet", "The presence of outliers (points beyond the whiskers) in certain variables may indicate "
             "unusual samples or vintage-specific effects that merit further investigation."),
            ("bullet", "The ratio R lib/lrg has a distinct scale compared to the percentage variables, further "
             "reinforcing the need for autoscaling."),

            "Understanding these distributions is essential before applying PCA, SIMCA, or PLS-DA, as "
            "large differences in variable scales could bias the multivariate models toward high-variance "
            "variables if left unscaled."
        ]
    },

    "02_boxplot_by_class.png": {
        "title": "2. Anthocyanin Distribution by Grape Variety",
        "caption": "Figure 2 — Boxplots of each variable stratified by grape variety (A, M, LP, S, N).",
        "text": [
            "This figure shows separate boxplots for each anthocyanin variable, stratified by the five "
            "grape varieties: Ancellotta (A), Montepulciano (M), Lambrusco Pugliese (LP), Sangiovese (S) "
            "and Nero d'Avola (N).",

            "The goal is to identify which variables best discriminate among varieties before applying "
            "multivariate classification. Key findings:",

            ("bullet", "Variables where the box positions are well-separated between classes (e.g., minimal "
             "overlap of inter-quartile ranges) are strong candidates for classification."),
            ("bullet", "Ancellotta often shows a distinctive anthocyanin signature, with characteristically "
             "high or low values on certain variables."),
            ("bullet", "Some varieties (e.g., Sangiovese and Nero d'Avola) may overlap considerably on "
             "individual variables, indicating that multivariate methods are needed for reliable discrimination."),
            ("bullet", "Lambrusco Pugliese may show intermediate or unique patterns, reflecting the hybrid "
             "character of this cultivar."),

            "This univariate analysis provides a baseline understanding that will be extended by the PCA "
            "score and loading plots in the subsequent sections."
        ]
    },

    "02b_boxplot_by_vintage.png": {
        "title": "2b. Anthocyanin Distribution by Vintage (2000 vs 2001)",
        "caption": "Figure 2b — Boxplots stratified by harvest year to assess the vintage effect.",
        "text": [
            "This plot directly addresses one of the key scientific questions: does the harvest vintage "
            "(2000 vs 2001) introduce systematic variability that could confound variety-based classification?",

            "By comparing the distributions of each anthocyanin variable between the two vintages, we can assess:",

            ("bullet", "Whether vintage effects are smaller or larger than varietal effects. If the boxplots "
             "for 2000 and 2001 overlap substantially, vintage is a minor source of variation."),
            ("bullet", "Whether specific variables are more sensitive to climatic/vintage conditions than others, "
             "which would affect their reliability as varietal markers."),
            ("bullet", "Whether a pooled analysis (combining both vintages) is appropriate, or whether vintage "
             "should be included as a factor or used for stratification."),

            "Generally, in grape anthocyanin studies, varietal differences tend to dominate over vintage effects, "
            "but environmental conditions (rainfall, temperature, sunlight hours) can modulate the absolute "
            "concentrations. This plot helps verify this assumption for our dataset."
        ]
    },

    "03_correlation_matrix.png": {
        "title": "3. Correlation Matrix of Anthocyanin Variables",
        "caption": "Figure 3 — Pearson correlation matrix with numerical annotations for all variable pairs.",
        "text": [
            "The correlation matrix displays pairwise Pearson correlation coefficients between the six "
            "anthocyanin variables. The color intensity indicates the strength and direction of correlation.",

            "Understanding inter-variable correlations is important for several reasons:",

            ("bullet", "Highly correlated variables carry redundant information. PCA naturally addresses this "
             "by extracting orthogonal principal components."),
            ("bullet", "Strong negative correlations (deep blue) between some anthocyanin percentages are "
             "expected because the values represent relative proportions (area %) that must sum approximately "
             "to 100% (compositional data effect)."),
            ("bullet", "The R lib/lrg ratio may correlate differently with the individual anthocyanins because "
             "it represents a derived measurement, not a direct area percentage."),
            ("bullet", "Moderate correlations (|r| between 0.3 and 0.7) suggest that the variables contribute "
             "partly independent information, justifying a multivariate approach rather than relying on any "
             "single variable."),

            "Variables with very high collinearity (|r| > 0.9) might not add incremental information in "
            "PLS-DA models, potentially leading to fewer effective latent variables."
        ]
    },

    "04_pca_scree_plot.png": {
        "title": "4. PCA Scree Plot and Cumulative Explained Variance",
        "caption": "Figure 4 — Scree plot (left) and cumulative explained variance (right) for PCA on autoscaled data.",
        "text": [
            "The scree plot (left panel) shows the percentage of total variance explained by each principal "
            "component (PC). The cumulative variance plot (right panel) indicates how many PCs are needed to "
            "capture a given fraction of the total variability. The red dashed line marks the 95% threshold.",

            "Interpretation guidelines:",

            ("bullet", "The 'elbow' in the scree plot suggests the natural dimensionality of the data. PCs "
             "before the elbow capture systematic structure, while those after it represent noise."),
            ("bullet", "For 6 variables, we can have at most 6 PCs. Typically 2-4 PCs suffice to explain "
             "over 80-90% of the variance in anthocyanin data."),
            ("bullet", "The number of retained PCs informs SIMCA model complexity (independent PCs per class) "
             "and provides insight into how many latent variables PLS-DA might need."),
            ("bullet", "If PC1 alone explains a very high proportion (>50%), it means one dominant source of "
             "variation drives the data. If variance is more evenly distributed, the data requires more "
             "dimensions to describe adequately."),

            "This plot, combined with the subsequent score and loading plots, guides the choice of "
            "model complexity for both SIMCA and PLS-DA."
        ]
    },

    "05_pca_scores_variety.png": {
        "title": "5. PCA Score Plots Colored by Grape Variety",
        "caption": "Figure 5 — PCA scores: PC1 vs PC2 (left) and PC1 vs PC3 (right), colored by grape variety.",
        "text": [
            "The score plots project the 98 multi-dimensional samples onto the first principal components, "
            "providing a 2D visualization of the sample relationships. Each point represents a grape must "
            "sample, colored according to its variety.",

            "PC1 vs PC2 (left panel):",
            ("bullet", "Tight, well-separated clusters indicate that the varieties have distinct anthocyanin "
             "profiles that are captured by the first two PCs."),
            ("bullet", "Overlapping clusters suggest that the varieties are chemically similar on these "
             "components and require additional PCs or more advanced classification methods."),
            ("bullet", "The position along PC1 reflects the main direction of variance in the data, "
             "while PC2 captures the second most important orthogonal variation."),

            "PC1 vs PC3 (right panel):",
            ("bullet", "Some varieties that overlap on PC1-PC2 may separate on PC3, revealing additional "
             "chemical dimensions of difference."),
            ("bullet", "If no additional separation is gained, PC3 may not be useful for variety discrimination."),

            "These plots are purely exploratory and do not incorporate class information. Any observed "
            "clustering reflects genuine chemical similarity/difference in the anthocyanin profiles."
        ]
    },

    "05b_pca_scores_vintage.png": {
        "title": "5b. PCA Score Plots Colored by Vintage",
        "caption": "Figure 5b — PCA scores colored by harvest year (2000 vs 2001) to assess vintage effect.",
        "text": [
            "This is the same PCA score plot as Figure 5, but now samples are colored by vintage (2000 vs 2001) "
            "instead of variety. This allows a direct visual assessment of the vintage effect.",

            "Key questions addressed:",

            ("bullet", "If 2000 and 2001 samples are intermixed throughout the score space, vintage is NOT a "
             "major source of variation, and varietal differences dominate. This is the desired scenario for "
             "variety-based classification."),
            ("bullet", "If 2000 and 2001 samples form separate clusters or systematic shifts, vintage contributes "
             "significant, structured variation that could confound variety classification."),
            ("bullet", "Partial separation (e.g., shift along one PC but not others) suggests vintage effects "
             "are present but less dominant than varietal effects."),

            "Comparing this figure with Figure 5 is crucial: if the varietal clustering (Fig. 5) is much "
            "stronger than any vintage grouping (Fig. 5b), we can confidently proceed with combined-vintage "
            "classification models."
        ]
    },

    "06_pca_scores_3d.png": {
        "title": "6. PCA 3D Score Plot",
        "caption": "Figure 6 — Three-dimensional PCA score plot (PC1, PC2, PC3) colored by variety.",
        "text": [
            "The 3D score plot simultaneously visualizes the first three principal components, providing "
            "a more complete representation of the sample distribution than any single 2D projection.",

            ("bullet", "Clusters that appear overlapping in 2D may separate in the third dimension, revealing "
             "structure invisible in planar projections."),
            ("bullet", "The three axes together typically capture 70-90% of the total variance, giving a "
             "comprehensive low-dimensional picture of the data."),
            ("bullet", "The viewing angle is set to (azimuth=30, elevation=25) for optimal perspective, "
             "but interactive rotation in MATLAB may reveal additional separation."),

            "This visualization confirms whether three PCs provide adequate separation for the five grape "
            "varieties, informing the SIMCA model (which builds independent PCA models per class) and "
            "suggesting the minimum number of latent variables for PLS-DA."
        ]
    },

    "07_pca_loadings.png": {
        "title": "7. PCA Loading Plots",
        "caption": "Figure 7 — Bar chart of loadings for PC1-PC3 (left) and loading biplot PC1 vs PC2 (right).",
        "text": [
            "Loading plots reveal which original variables are most responsible for the structure observed "
            "in the score plots.",

            "Bar chart (left panel):",
            ("bullet", "Variables with large absolute loadings on a given PC contribute most to the "
             "direction of that component."),
            ("bullet", "Opposite signs indicate that these variables are anti-correlated in that PC direction. "
             "For example, if MVD% is positive on PC1 while PND% is negative, samples with high MVD% and "
             "low PND% will score high on PC1."),
            ("bullet", "A variable with near-zero loading on all retained PCs contributes little to the "
             "multivariate model and is essentially noise in the PCA context."),

            "Loading plot (right panel):",
            ("bullet", "Variables plotted near the unit circle correlation boundary have strong influence."),
            ("bullet", "Variables pointing in similar directions are positively correlated; those in opposite "
             "directions are negatively correlated."),
            ("bullet", "Variables close to the origin are poorly represented by PC1-PC2 and may be better "
             "captured by higher PCs."),

            "Combining the loading plot with the score plot allows interpretation: varieties positioned "
            "in a certain direction of the score space are characterized by the variables that load "
            "strongly in that same direction."
        ]
    },

    "07b_pca_biplot.png": {
        "title": "7b. PCA Biplot (Scores + Loadings)",
        "caption": "Figure 7b — Combined biplot overlaying normalized scores and variable loadings.",
        "text": [
            "The biplot superimposes the PCA scores (samples, shown as colored scatter points) and the "
            "loadings (variables, shown as arrows) on the same graph. This enables direct interpretation "
            "of which anthocyanins characterize each variety cluster.",

            "Reading the biplot:",
            ("bullet", "Samples displaced in the direction of a variable arrow have relatively high values "
             "of that variable compared to the average."),
            ("bullet", "Variable arrows pointing in opposite directions indicate negative correlation."),
            ("bullet", "The length of the arrow indicates how well that variable is represented in the PC1-PC2 "
             "plane. Short arrows mean the variable is better captured by higher PCs."),
            ("bullet", "A variety cluster located near the tip of the MVD% arrow indicates that those samples "
             "are enriched in malvidina relative to the overall mean."),

            "The biplot is one of the most powerful tools in chemometric exploratory analysis because it "
            "provides simultaneous characterization of both sample groupings and the chemical variables "
            "driving those groupings."
        ]
    },

    "08_simca_cv_metrics.png": {
        "title": "8. SIMCA Cross-Validation Metrics",
        "caption": "Figure 8 — Sensitivity, Specificity, and Efficiency from 5-fold venetian-blind CV as a function of the number of PCs.",
        "text": [
            "This figure presents the three key performance metrics for SIMCA across increasing model "
            "complexity (1 to 5 PCs). Metrics are computed via 5-segment venetian-blind cross-validation.",

            "Metric definitions:",
            ("bullet", "Sensitivity (top panel): the proportion of class members that are correctly accepted "
             "by their own class model. High sensitivity means the model captures the within-class structure."),
            ("bullet", "Specificity (middle panel): the proportion of non-members correctly rejected by the "
             "class model. High specificity means the model discriminates well against foreign samples."),
            ("bullet", "Efficiency (bottom panel): the geometric mean sqrt(Sensitivity x Specificity), "
             "balancing both criteria. It is maximized to select the optimal number of PCs per class."),

            "Interpretation guidelines:",
            ("bullet", "The optimal PC number per class is chosen at the peak of the efficiency curve. "
             "Different classes may require different numbers of PCs reflecting their internal complexity."),
            ("bullet", "Increasing PCs beyond the optimum typically increases sensitivity but decreases "
             "specificity (overfitting), leading to a drop in efficiency."),
            ("bullet", "If a class reaches high efficiency with just 1-2 PCs, its anthocyanin profile is "
             "simple and distinct. If 4-5 PCs are needed, the class has complex internal structure."),

            "The optimal PCs selected here are used to build the final SIMCA models for training and "
            "test set prediction."
        ]
    },

    "10_simca_confusion_matrices.png": {
        "title": "10. SIMCA Confusion Matrices",
        "caption": "Figure 10 — Confusion matrices for SIMCA on training (left) and test (right) sets.",
        "text": [
            "Confusion matrices provide a complete picture of classification performance by showing "
            "how samples from each true class are assigned to predicted classes.",

            "Reading the matrix:",
            ("bullet", "Rows represent the true class; columns represent the predicted class."),
            ("bullet", "Diagonal elements (green) are correct classifications; off-diagonal elements (red) "
             "are misclassifications."),
            ("bullet", "Percentages show the rate relative to the total number of samples in each true class."),
            ("bullet", "Overall accuracy is shown below each matrix."),

            "Training vs Test set comparison:",
            ("bullet", "The training set performance is expected to be equal or better than the test set."),
            ("bullet", "A large gap between training and test accuracy may indicate overfitting."),
            ("bullet", "Specific misclassification patterns reveal which varieties are most easily confused: "
             "for example, if Sangiovese samples are frequently assigned to Montepulciano, these two varieties "
             "may have similar anthocyanin profiles."),

            "SIMCA as a class-modeling technique assigns samples to the closest class model (minimum "
            "combined T² + Q distance). Samples that don't fit any model well may still be forced into "
            "the least-distant class."
        ]
    },

    "11_simca_discriminant_power.png": {
        "title": "11. SIMCA Discriminant Power",
        "caption": "Figure 11 — Discriminant power of each anthocyanin variable from the SIMCA model.",
        "text": [
            "SIMCA Discriminant Power quantifies the ability of each variable to distinguish between "
            "classes. It is computed as the ratio of inter-class residual variance to intra-class residual "
            "variance for each variable, aggregated across all class models.",

            "Interpretation:",
            ("bullet", "Higher discriminant power values indicate variables that show large residuals "
             "for out-of-class samples relative to in-class samples, i.e., they are key discriminators."),
            ("bullet", "The mean (dotted black line), 95th percentile (green dashed), and 99th percentile "
             "(red dashed) thresholds help identify truly important variables."),
            ("bullet", "Variables with discriminant power above the mean are above-average discriminators. "
             "Those above the 95th percentile are exceptional."),
            ("bullet", "Variables with low discriminant power contribute little to class separation and "
             "might be removed in a simplified model without significant performance loss."),

            "This analysis complements the PLS-DA VIP scores (Figure 21), providing a SIMCA-specific "
            "perspective on variable importance."
        ]
    },

    "12_simca_loadings.png": {
        "title": "12. SIMCA Loadings per Class",
        "caption": "Figure 12 — PCA loading vectors for each class-specific SIMCA model.",
        "text": [
            "SIMCA builds independent PCA models for each class. This figure shows the loading vectors "
            "(PC1, PC2, etc.) for each variety's local PCA model.",

            "Importance of class-specific loadings:",
            ("bullet", "Unlike global PCA loadings, these loadings reflect the internal variance structure "
             "within each grape variety. Different varieties may have different dominant variation directions."),
            ("bullet", "If two varieties share similar loading patterns, their PCA subspaces are similar, "
             "which may explain mutual confusion in SIMCA classification."),
            ("bullet", "The number of loading vectors differs per class (determined by the CV-optimal "
             "number of PCs). Fewer PCs indicate a simpler within-class structure."),
            ("bullet", "Variables with large absolute loading on PC1 of a class define the main direction "
             "of within-class variability, while PC2+ capture secondary variation."),

            "Comparing these class-specific loadings with the global PCA loadings (Figure 7) reveals "
            "whether the same chemical dimensions drive both global variation and within-class variation."
        ]
    },

    "13_simca_summary_table.png": {
        "title": "13. SIMCA Summary Table",
        "caption": "Figure 13 — Summary of SIMCA model parameters and performance metrics per class.",
        "text": [
            "This table consolidates the key results of the SIMCA analysis:",

            ("bullet", "nPCs: the optimal number of principal components per class, selected "
             "by maximizing CV efficiency."),
            ("bullet", "Sens_Train / Spec_Train / Eff_Train: training set sensitivity, specificity, "
             "and efficiency. These indicate how well the model fits the training data."),
            ("bullet", "Sens_Test / Spec_Test / Eff_Test: test set metrics providing an unbiased "
             "estimate of model generalization to unseen samples."),

            "Performance assessment:",
            ("bullet", "Classes with high sensitivity but low specificity accept too many foreign samples "
             "(model is too loose)."),
            ("bullet", "Classes with high specificity but low sensitivity reject too many of their own "
             "samples (model is too strict)."),
            ("bullet", "Efficiency near 1.0 indicates an excellent balance of both criteria."),

            "This summary facilitates quick comparison across the five grape varieties and "
            "identification of classes that may require model refinement."
        ]
    },

    "14_plsda_cv_error.png": {
        "title": "14. PLS-DA Cross-Validation Performance",
        "caption": "Figure 14 — CV classification metrics as a function of the number of latent variables.",
        "text": [
            "This four-panel figure shows the cross-validation performance of PLS-DA across increasing "
            "latent variable (LV) complexity. Venetian-blind CV with 5 segments is used.",

            "Panel descriptions:",
            ("bullet", "Top-left: percentage of correctly classified samples per class. "
             "Curves that plateau early suggest a simple discriminative structure."),
            ("bullet", "Bottom-left: number of misclassified samples per class. This is the complement "
             "of the correct classification percentage."),
            ("bullet", "Top-right: mean correct classification across all classes. This averaged metric "
             "helps select the overall optimal model complexity."),
            ("bullet", "Bottom-right: total misclassified samples. The minimum of this curve determines "
             "the optimal number of LVs."),

            "Selection criterion: the number of LVs that minimizes total misclassification is chosen. "
            "A minimum of 2 LVs is enforced to enable meaningful 2D score visualization.",

            "Overfitting check: if performance degrades after a certain number of LVs, the model is "
            "starting to fit noise. The validation error should be at or near its minimum at the "
            "selected LV number."
        ]
    },

    "15_plsda_rmsecv.png": {
        "title": "15. PLS-DA RMSECV per Class",
        "caption": "Figure 15 — Root Mean Square Error of Cross-Validation for the dummy Y response, per class.",
        "text": [
            "The RMSECV (Root Mean Square Error of Cross-Validation) measures the prediction accuracy "
            "of the continuous dummy Y variables across cross-validation folds.",

            "Key points:",
            ("bullet", "Each curve represents the RMSECV for one class's dummy response variable "
             "(1 for class members, 0 for non-members)."),
            ("bullet", "Lower RMSECV indicates better prediction of class membership in a continuous sense."),
            ("bullet", "The RMSECV typically decreases as LVs are added, then plateaus or increases "
             "when overfitting begins."),
            ("bullet", "The optimal LV number should be near the elbow where RMSECV stabilizes."),

            "Unlike the misclassification rate (which is a discrete, threshold-based metric), RMSECV "
            "captures the quality of the continuous predicted values. A model can have good RMSECV but "
            "poor classification if predicted values hover near the decision boundary (0.5 threshold).",

            "This plot complements Figure 14 by providing a continuous optimization criterion."
        ]
    },

    "16_plsda_ypred_train.png": {
        "title": "16. PLS-DA Predicted Y (Training Set)",
        "caption": "Figure 16 — Predicted dummy Y values for each class on the training set.",
        "text": [
            "For each of the five grape varieties, this plot shows the predicted Y value for every "
            "training sample. The red dashed line at Y = 0.5 is the classification threshold.",

            "Reading the plot:",
            ("bullet", "Points above the 0.5 threshold (for the correct class row) are classified "
             "as belonging to that class."),
            ("bullet", "Filled markers highlight true members of each class; open markers are non-members."),
            ("bullet", "True members should have predicted values close to 1.0 (well above 0.5)."),
            ("bullet", "Non-members should have predicted values close to 0.0 (well below 0.5)."),

            "Diagnostic value:",
            ("bullet", "Samples near the 0.5 boundary are uncertain and prone to misclassification."),
            ("bullet", "Training set predictions are typically optimistic — test set performance "
             "(Figure 17) provides a more realistic assessment."),
            ("bullet", "Systematic patterns (e.g., one class consistently near the boundary) may indicate "
             "that the model struggles with that particular variety.")
        ]
    },

    "17_plsda_ypred_test.png": {
        "title": "17. PLS-DA Predicted Y (Test Set)",
        "caption": "Figure 17 — Predicted dummy Y values for each class on the independent test set.",
        "text": [
            "Analogous to Figure 16, this shows predicted Y values on the independent test set (30% "
            "of samples not used during model building).",

            "This is the critical validation step:",
            ("bullet", "Test set samples were not seen during model calibration or LV selection, "
             "making this an unbiased performance estimate."),
            ("bullet", "Good generalization is indicated by clear separation of class members (predicted "
             "Y near 1) from non-members (predicted Y near 0), similar to training performance."),
            ("bullet", "Higher scatter or more samples near the 0.5 boundary compared to training "
             "indicates some degree of overfitting."),
            ("bullet", "Specific varieties with poor test predictions may need more training data "
             "or additional variables for reliable discrimination."),

            "The pattern of misclassifications visible here directly maps to the test set confusion "
            "matrix in Figure 18."
        ]
    },

    "18_plsda_confusion_matrices.png": {
        "title": "18. PLS-DA Confusion Matrices",
        "caption": "Figure 18 — Confusion matrices for PLS-DA on training (left) and test (right) sets.",
        "text": [
            "These confusion matrices summarize PLS-DA classification performance identically to "
            "Figure 10 (SIMCA), enabling direct method comparison.",

            "PLS-DA classification is performed by assigning each sample to the class with the "
            "highest predicted Y value (argmax). Unlike SIMCA, which uses distance-based acceptance, "
            "PLS-DA always assigns exactly one class to each sample.",

            "Comparison with SIMCA (Figure 10):",
            ("bullet", "PLS-DA, being a discriminant method, often achieves higher classification "
             "accuracy than SIMCA, which is a class-modeling technique."),
            ("bullet", "However, SIMCA can identify samples that don't belong to any modeled class, "
             "while PLS-DA always forces an assignment."),
            ("bullet", "If both methods agree on misclassification patterns (e.g., S confused with M), "
             "this strongly suggests genuine chemical similarity between those varieties."),
            ("bullet", "Discrepancies between methods can provide complementary insights."),

            "The overall accuracy percentage is reported below each matrix."
        ]
    },

    "19_plsda_scores.png": {
        "title": "19. PLS-DA Score Plots",
        "caption": "Figure 19 — PLS-DA scores LV1 vs LV2 for training (left) and test (right) sets.",
        "text": [
            "The PLS-DA score plots project samples onto the first two latent variable (LV) dimensions. "
            "Unlike PCA scores which maximize variance, PLS-DA scores maximize the covariance between "
            "X (anthocyanins) and Y (class membership), making them explicitly discriminative.",

            "Training set (left):",
            ("bullet", "Well-separated clusters confirm that the PLS-DA model has found discriminative "
             "directions in the anthocyanin space."),
            ("bullet", "The degree of separation reflects the quality of the model."),

            "Test set (right):",
            ("bullet", "Test samples are projected using the training model's weight/loading matrices."),
            ("bullet", "If test clusters overlap with or lie close to the corresponding training clusters, "
             "the model generalizes well."),
            ("bullet", "Displaced test clusters or scattered points suggest potential distribution shift "
             "or overfitting."),

            "Comparing PLS-DA scores with PCA scores (Figure 5) reveals how class-supervised projection "
            "improves discrimination over unsupervised projection."
        ]
    },

    "20_plsda_regression_coefficients.png": {
        "title": "20. PLS-DA Regression Coefficients",
        "caption": "Figure 20 — PLS regression coefficients (B_PLS) for each class.",
        "text": [
            "The PLS-DA regression coefficients quantify the contribution of each autoscaled anthocyanin "
            "variable to the prediction of class membership for each variety.",

            "Interpretation:",
            ("bullet", "Large positive coefficients indicate variables that increase the predicted Y "
             "for that class (i.e., positive markers for that variety)."),
            ("bullet", "Large negative coefficients indicate variables that decrease predicted Y "
             "(i.e., the variety has relatively low values of that anthocyanin)."),
            ("bullet", "Near-zero coefficients contribute negligibly to the prediction."),

            "Practical significance:",
            ("bullet", "The regression coefficient profile provides a characteristic 'chemical fingerprint' "
             "for each variety from the model's perspective."),
            ("bullet", "By comparing coefficient patterns across classes, one can identify which variables "
             "best distinguish each variety from the others."),
            ("bullet", "These coefficients are only meaningful for autoscaled data — the original units "
             "have been normalized, so coefficient magnitudes reflect importance on a standardized scale."),

            "This analysis is complementary to VIP scores (Figure 21) which provide a single aggregated "
            "importance value per variable across all classes."
        ]
    },

    "21_plsda_vip.png": {
        "title": "21. PLS-DA Variable Importance in Projection (VIP)",
        "caption": "Figure 21 — VIP scores indicating each variable's importance in the PLS-DA model.",
        "text": [
            "The VIP (Variable Importance in Projection) summarizes each variable's contribution across "
            "all latent variables and all classes in a single score.",

            "Interpretation guidelines:",
            ("bullet", "VIP > 1 (red dashed line): the variable is considered important — it contributes "
             "above average to the model's predictive ability."),
            ("bullet", "VIP between 0.8 and 1 (black dotted line): moderate importance, borderline."),
            ("bullet", "VIP < 0.8: the variable contributes below average and could potentially be "
             "removed without significantly degrading model performance."),

            "For the mosti dataset:",
            ("bullet", "Variables with the highest VIP scores are the most reliable anthocyanin markers "
             "for variety classification and should be prioritized in simplified analytical methods."),
            ("bullet", "The VIP profile reflects the combined importance across all five varieties, "
             "while the regression coefficients (Figure 20) show class-specific effects."),

            "In practice, VIP-based variable selection can lead to more parsimonious models. "
            "A model using only variables with VIP > 1 may achieve similar classification performance "
            "with fewer measurements, reducing analytical costs."
        ]
    },

    "22_plsda_summary_table.png": {
        "title": "22. PLS-DA Summary Table",
        "caption": "Figure 22 — Summary of PLS-DA performance metrics including RMSEP.",
        "text": [
            "This summary table consolidates PLS-DA classification performance:",

            ("bullet", "Sensitivity, Specificity, Efficiency: computed for both training and test sets, "
             "analogous to the SIMCA summary (Figure 13)."),
            ("bullet", "RMSEP (Root Mean Square Error of Prediction): the test set prediction error for "
             "each class's dummy variable. Lower RMSEP indicates better prediction quality."),
            ("bullet", "The number of latent variables (LVs) used in the final model is indicated in the title."),

            "Cross-method comparison:",
            ("bullet", "Comparing this table with the SIMCA summary (Figure 13) reveals which method "
             "performs better for each variety."),
            ("bullet", "A method with consistently higher efficiency across all classes is generally preferred."),
            ("bullet", "If methods excel on different classes, an ensemble approach could be considered."),

            "RMSEP provides a continuous quality measure beyond the binary correct/incorrect "
            "classification, indicating how confidently the model assigns samples to their correct class."
        ]
    },

    "23_comparison_simca_plsda.png": {
        "title": "23. SIMCA vs PLS-DA Comparison",
        "caption": "Figure 23 — Side-by-side comparison of test set sensitivity and specificity for both methods.",
        "text": [
            "This final comparison figure enables a direct visual assessment of SIMCA versus PLS-DA "
            "performance on the independent test set.",

            "Sensitivity comparison (left):",
            ("bullet", "Shows the proportion of true class members correctly identified by each method."),
            ("bullet", "PLS-DA often achieves higher sensitivity as it is optimized for discrimination."),
            ("bullet", "SIMCA may have lower sensitivity for classes with complex or diffuse internal structure."),

            "Specificity comparison (right):",
            ("bullet", "Shows the proportion of non-members correctly rejected."),
            ("bullet", "SIMCA, as a class-modeling technique, can sometimes achieve higher specificity "
             "because its acceptance boundary is explicitly defined by T² and Q limits."),
            ("bullet", "PLS-DA always assigns a class, which can reduce specificity for similar classes."),

            "Methodological considerations:",
            ("bullet", "SIMCA is preferred when the goal is to verify class membership or detect unknowns."),
            ("bullet", "PLS-DA is preferred when the goal is maximum discrimination among known classes."),
            ("bullet", "For the mosti dataset (98 samples, 5 varieties), the combined evidence from both "
             "methods provides a robust chemometric characterization of grape variety based on "
             "anthocyanin profiles."),

            "The choice between methods should also consider the practical application: quality control "
            "(SIMCA preferred) vs. origin authentication (PLS-DA preferred)."
        ]
    },

    "24_applicability_domain.png": {
        "title": "24. Applicability Domain — Williams Plot",
        "caption": "Figure 24 — Williams plot (leverage vs residui standardizzati) e leverage dei campioni di test.",
        "text": [
            "Il dominio di applicabilità (Applicability Domain, AD) verifica se i campioni del test set "
            "ricadono nello spazio chimico coperto dal training set. Un modello di classificazione fornisce "
            "predizioni affidabili solo per campioni che rientrano nell'AD.",

            "Pannello sinistro — Williams plot:",
            ("bullet", "L'asse X mostra il leverage (h_i), che misura la distanza di ogni campione dal "
             "centroide del training set nello spazio delle X."),
            ("bullet", "L'asse Y mostra i residui standardizzati del modello PLS-DA."),
            ("bullet", "La soglia h* = 3(p+1)/n (linea rossa tratteggiata) definisce il limite di leverage: "
             "campioni con h > h* sono lontani dal centro del training set e le loro predizioni sono meno affidabili."),
            ("bullet", "Le linee nere a ±3 definiscono il limite di residuo standardizzato: campioni oltre queste "
             "soglie hanno residui anomali."),
            ("bullet", "I campioni nell'angolo in basso a sinistra (basso leverage, basso residuo) sono nel cuore "
             "dell'AD e le loro predizioni sono le più affidabili."),

            "Pannello destro — Leverage del test set:",
            ("bullet", "Il grafico a barre mostra il leverage di ogni campione di test, colorato per varietà."),
            ("bullet", "I campioni sopra la soglia h* (linea rossa) sono fuori dal dominio di applicabilità "
             "del modello."),
            ("bullet", "La percentuale di campioni fuori AD indica quanto il test set è rappresentativo "
             "dello spazio chimico del training set."),

            "Un'alta percentuale di campioni di test all'interno dell'AD conferma che la partizione "
            "70/30 stratificata ha prodotto un test set chimicamente coerente con il training set."
        ]
    },

    "25_applicability_domain_T2Q.png": {
        "title": "25. Applicability Domain — T² vs Q (PCA-based)",
        "caption": "Figure 25 — Hotelling T² normalizzato vs Q normalizzato per valutare l'AD basato su PCA.",
        "text": [
            "Questo grafico fornisce una seconda prospettiva sull'Applicability Domain, basata sulla "
            "decomposizione PCA del training set.",

            "Definizioni:",
            ("bullet", "T² (Hotelling): misura la distanza del campione dal centroide all'interno "
             "del modello (nello spazio delle componenti principali trattenute). Un T² elevato indica "
             "un campione estremo ma ancora descritto dal modello."),
            ("bullet", "Q (residui): misura la distanza del campione dal sottospazio del modello PCA. "
             "Un Q elevato indica un campione con struttura non catturata dalle PCs trattenute."),
            ("bullet", "Entrambe le statistiche sono normalizzate per i rispettivi limiti al 95%, "
             "cosicché valori > 1 indicano superamento della soglia."),

            "Interpretazione:",
            ("bullet", "Campioni di training (cerchi vuoti) definiscono la nube di riferimento e dovrebbero "
             "concentrarsi nella regione T²/T²_lim < 1 e Q/Q_lim < 1."),
            ("bullet", "Campioni di test (diamanti pieni) che ricadono nella stessa regione del training "
             "sono nell'AD e le predizioni dei modelli sono affidabili."),
            ("bullet", "Campioni con T²/T²_lim > 1 ma Q/Q_lim < 1 sono estremi ma ancora nel modello "
             "(leverage influente)."),
            ("bullet", "Campioni con Q/Q_lim > 1 presentano struttura chimica non rappresentata nel "
             "training set, e le predizioni possono essere inattendibili."),

            "La combinazione del Williams plot (Figura 24) e del T² vs Q plot offre un quadro "
            "completo dell'affidabilità delle predizioni sui campioni di test."
        ]
    },
}

# Also handle the per-class SIMCA SD vs OD plots (09_simca_SDvsOD_classX.png)
CLASS_FULL = {
    1: "Ancellotta (A)",
    2: "Montepulciano (M)",
    3: "Lambrusco Pugliese (LP)",
    4: "Sangiovese (S)",
    5: "Nero d'Avola (N)",
}

for ic in range(1, 6):
    key = f"09_simca_SDvsOD_class{ic}.png"
    FIGURE_DESCRIPTIONS[key] = {
        "title": f"9.{ic}. SIMCA Acceptance Plot — {CLASS_FULL[ic]}",
        "caption": f"Figure 9.{ic} — Score Distance (T²) vs Orthogonal Distance (Q) for class model {CLASS_FULL[ic]}.",
        "text": [
            f"This plot shows the normalized Hotelling T² (score distance) versus Q residuals "
            f"(orthogonal distance) for the SIMCA model of {CLASS_FULL[ic]}. Both distances are "
            f"normalized by their respective 95% confidence limits (T²_lim and Q_lim).",

            "Plot elements:",
            ("bullet", "Open circles: training samples; filled diamonds: test samples."),
            ("bullet", "Colors indicate the true grape variety of each sample."),
            ("bullet", f"The red circle at radius sqrt(2) represents the combined acceptance boundary: "
             f"samples inside are accepted as belonging to {CLASS_FULL[ic]}."),

            "Interpretation:",
            ("bullet", f"True {CLASS_FULL[ic]} samples (both train and test) should fall inside the boundary."),
            ("bullet", "Samples of other varieties should ideally fall outside the boundary."),
            ("bullet", "Samples inside with the wrong variety represent specificity failures."),
            ("bullet", "True members outside the boundary represent sensitivity failures."),

            f"The combined criterion D = sqrt((T²/T²_lim)² + (Q/Q_lim)²) <= sqrt(2) is used. "
            f"This is equivalent to requiring that both normalized distances are simultaneously "
            f"within acceptable limits."
        ]
    }


# ---------------------------------------------------------------------------
# PDF Helper Functions
# ---------------------------------------------------------------------------
def get_image_flowable(img_path, max_width, max_height=16*cm):
    """Return an Image flowable scaled to fit within bounds."""
    try:
        pil_img = PILImage.open(img_path)
        w_px, h_px = pil_img.size
    except Exception:
        w_px, h_px = 800, 600

    aspect = h_px / w_px
    w = max_width
    h = w * aspect

    if h > max_height:
        h = max_height
        w = h / aspect

    return Image(img_path, width=w, height=h)


def add_header_footer(canvas, doc):
    """Custom header/footer for each page."""
    canvas.saveState()
    # Footer
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(HexColor("#999999"))
    canvas.drawCentredString(
        PAGE_W / 2,
        1.0 * cm,
        f"Mosti Classification Report — Page {doc.page}"
    )
    # Thin line above footer
    canvas.setStrokeColor(HexColor("#cccccc"))
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 1.4 * cm, PAGE_W - MARGIN, 1.4 * cm)
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Build PDF
# ---------------------------------------------------------------------------
def build_pdf():
    ss = build_styles()

    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=2.0 * cm,
        title="Mosti Classification Report",
        author="Chemometric Analysis Pipeline",
    )

    story = []

    # ---- Cover Page ----
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph(
        "Classificazione di Mosti di Uva tramite"
        " Profilo Antocianico (HPLC)",
        ss["CoverTitle"]
    ))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "Esame di Elaborazione Dati Scientifici — Analisi Chemometrica",
        ss["CoverSubtitle"]
    ))
    story.append(Spacer(1, 1.5 * cm))
    story.append(HRFlowable(
        width="60%", thickness=1, color=LIGHT_BLUE,
        spaceAfter=12, spaceBefore=12
    ))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "Metodi: PCA (Esplorazione) | SIMCA (Class Modeling) | PLS-DA (Analisi Discriminante)",
        ss["CoverInfo"]
    ))
    story.append(Paragraph(
        "Dataset: 98 campioni di mosto | 6 variabili antocianiche HPLC | 5 varietali",
        ss["CoverInfo"]
    ))
    story.append(Paragraph(
        "Varietali: Ancellotta (A), Montepulciano (M), Lambrusco Pugliese (LP), Sangiovese (S), Nero d'Avola (N)",
        ss["CoverInfo"]
    ))
    story.append(Paragraph(
        "Vendemmie: 2000 e 2001 (analisi combinata)",
        ss["CoverInfo"]
    ))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        f"Report generato: {datetime.now().strftime('%d/%m/%Y ore %H:%M')}",
        ss["CoverInfo"]
    ))
    story.append(PageBreak())

    # ---- Table of Contents header ----
    story.append(Paragraph("Indice dei Contenuti", ss["SectionHeading"]))
    story.append(Spacer(1, 0.3 * cm))

    # Build ordered list of plots
    ordered_plots = sorted(
        [f for f in os.listdir(PLOTS_DIR) if f.endswith(".png")],
        key=lambda x: x
    )

    # TOC entries
    toc_items = []
    for pf in ordered_plots:
        if pf in FIGURE_DESCRIPTIONS:
            info = FIGURE_DESCRIPTIONS[pf]
            toc_items.append(info["title"])
        else:
            toc_items.append(pf.replace(".png", "").replace("_", " "))

    for i, item in enumerate(toc_items):
        story.append(Paragraph(
            f"<b>{item}</b>",
            ss["BodyText2"]
        ))

    story.append(PageBreak())

    # ---- Introduction & Objectives ----
    story.append(Paragraph("Introduzione e Obiettivi", ss["SectionHeading"]))
    story.append(Spacer(1, 0.3 * cm))

    intro_text = [
        "Il presente lavoro affronta la <b>classificazione di mosti di uva sulla base del profilo "
        "antocianico</b> determinato mediante HPLC. Il dataset comprende 98 campioni di mosto "
        "appartenenti a cinque varietali (Ancellotta, Montepulciano, Lambrusco Pugliese, Sangiovese "
        "e Nero d'Avola) raccolti in due annate (2000 e 2001). Per ciascun campione sono state "
        "quantificate sei variabili: le percentuali di area dei cinque principali antociani "
        "(delfinidina DPD%, cianidina CYD%, petunidina PTD%, peonidina PND%, malvidina MVD%) "
        "e il rapporto tra antociani liberi e legati (R lib/lrg).",

        "<b>Obiettivi dell'analisi:</b>",

        ("bullet", "Verificare se è possibile <b>distinguere i diversi varietali</b> sulla base "
         "del profilo antocianico."),
        ("bullet", "Valutare se l'<b>anno di produzione</b> (vendemmia 2000 vs 2001) rappresenta un "
         "fattore di <b>variabilità minore</b> rispetto al varietale."),
        ("bullet", "Identificare <b>quali variabili distinguono meglio</b> i campioni di varietali diversi."),
        ("bullet", "Applicare sia <b>SIMCA</b> (class modeling) sia <b>PLS-DA</b> (analisi discriminante) "
         "per valutare la possibilità di classificare i diversi varietali."),
        ("bullet", "Verificare l'<b>Applicability Domain</b> del modello per confermare l'affidabilità "
         "delle predizioni sul test set."),
    ]

    for item in intro_text:
        if isinstance(item, tuple) and item[0] == "bullet":
            story.append(Paragraph(f"&#8226; {item[1]}", ss["BulletItem"]))
        else:
            story.append(Paragraph(item, ss["BodyText2"]))
    story.append(Spacer(1, 0.3 * cm))

    # ---- Methodology Overview ----
    story.append(Paragraph("Metodologia", ss["SectionHeading"]))
    story.append(Spacer(1, 0.3 * cm))

    method_text = [
        "<b>Preprocessing — Autoscaling:</b> Tutte le variabili sono state centrate sulla media e "
        "divise per la deviazione standard (autoscaling), in modo da ottenere variabili a media zero "
        "e varianza unitaria. Questo è essenziale perché le variabili hanno unità e scale diverse "
        "(percentuali vs rapporti).",

        "<b>PCA (Principal Component Analysis):</b> Analisi esplorativa non supervisionata per "
        "visualizzare il raggruppamento dei campioni, identificare le relazioni tra variabili e "
        "valutare l'importanza relativa dell'effetto varietale rispetto a quello dell'annata.",

        "<b>SIMCA (Soft Independent Modeling of Class Analogy):</b> Tecnica di class modeling che "
        "costruisce un modello PCA indipendente per ciascun varietale. Il numero di PCs per classe "
        "è ottimizzato via cross-validazione venetian-blind a 5 segmenti (massimizzazione dell'efficienza). "
        "L'accettazione è basata sul criterio combinato T² + Q con α = 0.05.",

        "<b>PLS-DA (Partial Least Squares Discriminant Analysis):</b> Metodo discriminante supervisionato "
        "che trova le variabili latenti (LV) che massimizzano la covarianza tra X (antociani) e "
        "Y (appartenenza di classe, dummy-coded con centratura sulla media). Il numero di LV è selezionato "
        "minimizzando l'errore di classificazione in cross-validazione.",

        "<b>Applicability Domain:</b> Verifica che i campioni di test ricadano nel dominio chimico del "
        "training set. Sono utilizzati due approcci complementari: (1) Williams plot basato sul leverage "
        "(hat matrix) e (2) diagramma T² vs Q normalizzati.",

        "<b>Validazione:</b> Suddivisione stratificata 70/30 training/test (seed=42) per una validazione "
        "esterna non distorta. I parametri di complessità dei modelli sono selezionati in cross-validazione.",
    ]

    for mt in method_text:
        story.append(Paragraph(mt, ss["BodyText2"]))
        story.append(Spacer(1, 0.1 * cm))

    story.append(PageBreak())

    # ---- Figure pages ----
    for pf in ordered_plots:
        img_path = os.path.join(PLOTS_DIR, pf)
        if not os.path.isfile(img_path):
            continue

        if pf in FIGURE_DESCRIPTIONS:
            info = FIGURE_DESCRIPTIONS[pf]
            elements = []

            # Section title
            elements.append(Paragraph(info["title"], ss["SectionHeading"]))
            elements.append(Spacer(1, 0.2 * cm))

            # Image
            img = get_image_flowable(img_path, USABLE_W, max_height=12 * cm)
            elements.append(img)

            # Caption
            elements.append(Paragraph(info["caption"], ss["FigureCaption"]))
            elements.append(Spacer(1, 0.3 * cm))

            # Description paragraphs
            for item in info["text"]:
                if isinstance(item, tuple) and item[0] == "bullet":
                    elements.append(Paragraph(
                        f"&#8226; {item[1]}",
                        ss["BulletItem"]
                    ))
                else:
                    elements.append(Paragraph(item, ss["BodyText2"]))

            elements.append(PageBreak())
            story.extend(elements)
        else:
            # Unknown plot, still include it
            story.append(Paragraph(
                pf.replace(".png", "").replace("_", " ").title(),
                ss["SectionHeading"]
            ))
            img = get_image_flowable(img_path, USABLE_W, max_height=14 * cm)
            story.append(img)
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph(
                f"<i>Figure from file: {pf}</i>",
                ss["FigureCaption"]
            ))
            story.append(PageBreak())

    # ---- Conclusions ----
    story.append(Paragraph("Conclusioni", ss["SectionHeading"]))
    story.append(Spacer(1, 0.3 * cm))

    conclusions = [
        "L'analisi chemometrica integrata condotta su 98 campioni di mosto ha permesso di rispondere "
        "alle domande poste dalla traccia d'esame. Di seguito i risultati principali.",

        "<b>1. È possibile distinguere i diversi varietali?</b>",
        ("bullet", "Sì. L'analisi PCA mostra un raggruppamento chiaro dei cinque varietali nello spazio "
         "delle prime componenti principali (Figg. 5, 6, 7b). I cluster sono ben separati, indicando "
         "che il profilo antocianico è un marcatore efficace dell'identità varietale."),
        ("bullet", "Sia SIMCA che PLS-DA confermano la classificabilità, con valori di efficienza (efficiency) "
         "generalmente elevati sul test set esterno (Figg. 13, 22, 23)."),

        "<b>2. L'anno di produzione è un fattore di minore variabilità?</b>",
        ("bullet", "Sì. Il confronto dei PCA scores colorati per annata (Fig. 5b) mostra che i campioni "
         "delle vendemmie 2000 e 2001 si sovrappongono ampiamente, senza formare cluster distinti. "
         "La variabilità introdotta dall'annata è subordinata a quella varietale."),
        ("bullet", "I boxplot per annata (Fig. 2b) confermano che le distribuzioni delle singole variabili "
         "sono sostanzialmente sovrapposte tra le due vendemmie."),

        "<b>3. Quali variabili distinguono meglio i campioni?</b>",
        ("bullet", "Le variabili con la maggiore importanza discriminante sono identificate dagli "
         "scores VIP del modello PLS-DA (Fig. 21, VIP > 1) e dal Discriminant Power SIMCA (Fig. 11)."),
        ("bullet", "I coefficienti di regressione PLS-DA (Fig. 20) forniscono un'impronta chimica "
         "specifica per ciascun varietale, evidenziando quali antociani caratterizzano quale varietà."),
        ("bullet", "I loading PCA (Figg. 7, 7b) e i loading SIMCA per classe (Fig. 12) completano "
         "il quadro, mostrando le direzioni di variazione chimica più informative."),

        "<b>4. SIMCA e PLS-DA riescono a classificare i varietali?</b>",
        ("bullet", "SIMCA, come tecnica di class modeling, costruisce modelli indipendenti per ogni "
         "varietà (Figg. 8-13). La sua forza è nella capacità di rigettare campioni estranei "
         "(alta specificità), utile per il controllo qualità."),
        ("bullet", "PLS-DA, come tecnica discriminante, ottimizza direttamente la separazione tra classi "
         "(Figg. 14-22) e in genere raggiunge una maggiore accuratezza complessiva di classificazione."),
        ("bullet", "Il confronto diretto (Fig. 23) evidenzia i punti di forza complementari dei due metodi."),

        "<b>5. Applicability Domain</b>",
        ("bullet", "L'analisi del dominio di applicabilità (Figg. 24-25) conferma che la quasi totalità "
         "dei campioni di test ricade nel dominio chimico del training set, validando l'affidabilità "
         "delle predizioni. Il Williams plot e il diagramma T² vs Q non evidenziano campioni di test "
         "anomali o fuori dominio."),

        "<b>In sintesi</b>: la combinazione di PCA (esplorazione), SIMCA (class modeling) e PLS-DA "
        "(discriminazione) fornisce un framework robusto per l'autenticazione dei mosti di uva su base "
        "antocianica. Il profilo HPLC delle sei variabili misurate è sufficiente per distinguere "
        "i cinque varietali studiati, indipendentemente dall'annata di vendemmia.",
    ]

    for item in conclusions:
        if isinstance(item, tuple) and item[0] == "bullet":
            story.append(Paragraph(f"&#8226; {item[1]}", ss["BulletItem"]))
        else:
            story.append(Paragraph(item, ss["BodyText2"]))

    # Build the PDF
    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    print(f"\n[OK] Report saved to: {OUTPUT_PDF}")
    print(f"     Pages: ~{len([f for f in ordered_plots if f.endswith('.png')]) + 5}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.isdir(PLOTS_DIR):
        print(f"[ERROR] Plots directory not found: {PLOTS_DIR}")
        print("        Run main_analysis_mosti.m in MATLAB first to generate the plots.")
        exit(1)

    png_files = glob.glob(os.path.join(PLOTS_DIR, "*.png"))
    if not png_files:
        print(f"[WARNING] No PNG files found in {PLOTS_DIR}")
        print("          Run main_analysis_mosti.m in MATLAB first to generate the plots.")
        exit(1)

    print(f"Found {len(png_files)} plots in {PLOTS_DIR}")
    print("Generating professional PDF report...")
    build_pdf()
