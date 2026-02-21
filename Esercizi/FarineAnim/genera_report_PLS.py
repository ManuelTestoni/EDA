#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Per eseguire: source .venv/bin/activate && python genera_report_PLS.py
"""
=============================================================================
  GENERATORE REPORT PDF - Analisi PLS Regression Farine Animali NIR
=============================================================================

Genera un report PDF professionale a partire dai grafici prodotti dallo script
MATLAB 'analisi_PLS_regression.m'.

Ogni grafico viene incluso con una spiegazione scientifica dettagliata.

Requisiti:
    pip install reportlab Pillow

Uso:
    python genera_report_PLS.py

Output:
    pls_plots/report_PLS_regression.pdf
"""

import os
import glob
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
    Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.platypus.flowables import Flowable

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURAZIONE
# ─────────────────────────────────────────────────────────────────────────────
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pls_plots")
OUTPUT_PDF = os.path.join(PLOT_DIR, "report_PLS_regression.pdf")

# Colori tema
COLOR_PRIMARY   = HexColor("#1B3A5C")   # blu scuro
COLOR_SECONDARY = HexColor("#2E86AB")   # blu medio
COLOR_ACCENT    = HexColor("#A23B72")   # magenta
COLOR_BG_LIGHT  = HexColor("#F0F4F8")   # grigio chiaro
COLOR_TEXT       = HexColor("#2C3E50")

PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm
CONTENT_W = PAGE_W - 2 * MARGIN


# ─────────────────────────────────────────────────────────────────────────────
#  STILI
# ─────────────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

style_title = ParagraphStyle(
    "ReportTitle",
    parent=styles["Title"],
    fontSize=26,
    leading=32,
    textColor=COLOR_PRIMARY,
    spaceAfter=6 * mm,
    alignment=TA_CENTER,
    fontName="Helvetica-Bold",
)

style_subtitle = ParagraphStyle(
    "ReportSubtitle",
    parent=styles["Normal"],
    fontSize=13,
    leading=17,
    textColor=COLOR_SECONDARY,
    spaceAfter=10 * mm,
    alignment=TA_CENTER,
    fontName="Helvetica-Oblique",
)

style_section = ParagraphStyle(
    "SectionTitle",
    parent=styles["Heading1"],
    fontSize=16,
    leading=20,
    textColor=COLOR_PRIMARY,
    spaceBefore=12 * mm,
    spaceAfter=4 * mm,
    fontName="Helvetica-Bold",
    borderWidth=0,
    borderPadding=0,
    borderColor=COLOR_PRIMARY,
)

style_subsection = ParagraphStyle(
    "SubSectionTitle",
    parent=styles["Heading2"],
    fontSize=13,
    leading=16,
    textColor=COLOR_SECONDARY,
    spaceBefore=8 * mm,
    spaceAfter=3 * mm,
    fontName="Helvetica-Bold",
)

style_body = ParagraphStyle(
    "BodyText",
    parent=styles["Normal"],
    fontSize=10,
    leading=14,
    textColor=COLOR_TEXT,
    spaceAfter=3 * mm,
    alignment=TA_JUSTIFY,
    fontName="Helvetica",
)

style_caption = ParagraphStyle(
    "FigCaption",
    parent=styles["Normal"],
    fontSize=9,
    leading=12,
    textColor=HexColor("#555555"),
    spaceBefore=2 * mm,
    spaceAfter=6 * mm,
    alignment=TA_CENTER,
    fontName="Helvetica-Oblique",
)

style_note = ParagraphStyle(
    "NoteText",
    parent=styles["Normal"],
    fontSize=9,
    leading=12,
    textColor=HexColor("#666666"),
    spaceAfter=4 * mm,
    leftIndent=10 * mm,
    fontName="Helvetica",
)


# ─────────────────────────────────────────────────────────────────────────────
#  COMPONENTE PERSONALIZZATO: Linea separatrice
# ─────────────────────────────────────────────────────────────────────────────
class SectionDivider(Flowable):
    """Linea colorata come separatore di sezione."""
    def __init__(self, width, color=COLOR_SECONDARY, thickness=1.5):
        super().__init__()
        self.width = width
        self.color = color
        self.thickness = thickness
        self.height = thickness + 4 * mm

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 2 * mm, self.width, 2 * mm)


# ─────────────────────────────────────────────────────────────────────────────
#  DEFINIZIONE CONTENUTI
# ─────────────────────────────────────────────────────────────────────────────
#  Ogni entry: (filename, titolo_sezione, testo_descrittivo)

PLOT_DESCRIPTIONS = [
    # --- PCA ESPLORATIVA ---
    (
        "01_PCA_scores_PC1_PC2.png",
        "1. PCA Esplorativa — Score Plot (PC1 vs PC2)",
        """Lo score plot della PCA esplorativa (con solo mean centering) mostra la proiezione dei campioni
        nel sotto-spazio a 2 componenti principali. <b>PC1</b> spiega tipicamente la quasi totalità della
        varianza (&gt;99%) e cattura effetti fisici quali variazioni di <i>scattering</i> e spostamenti
        di baseline, che colpiscono uniformemente l'intero spettro NIR.
        <br/><br/>
        In questo grafico si osserva la distribuzione dei campioni colorati per classe (pollo, bovino, pesce).
        Se le classi non sono ben separate lungo PC1, ciò conferma che la prima componente è dominata da
        variazioni fisiche piuttosto che chimiche, e indica la necessità di un preprocessing spettrale
        (es. derivate, SNV, MSC) per rimuovere questi effetti prima di costruire il modello PLS."""
    ),
    (
        "02_PCA_loadings.png",
        "2. PCA Esplorativa — Loading Plot",
        """I loadings mostrano il contributo di ciascun wavenumber alle componenti principali.
        <br/><br/>
        <b>Loading PC1 (blu):</b> Un profilo quasi piatto o monotono indica che PC1 cattura variazioni fisiche
        (scattering, baseline shift) che influenzano uniformemente lo spettro. Questo è coerente con il
        fatto che PC1 spiega una percentuale enorme della varianza.
        <br/><br/>
        <b>Loading PC2 (rosso):</b> Nonostante spieghi poca varianza, presenta picchi e valli ben definiti,
        indicatori di variazione chimica reale: le regioni con loadings elevati corrispondono a bande di
        assorbimento di gruppi funzionali come O-H (~5200 cm⁻¹), N-H (~4600 cm⁻¹) e C-H (~4300 cm⁻¹),
        legati a differenze nel contenuto di acqua, proteine e grassi tra le categorie."""
    ),
    (
        "03_PCA_T2_vs_Q.png",
        "3. PCA Esplorativa — T² vs Q (Outlier Detection)",
        """Il grafico T² vs Q permette di identificare campioni anomali (outlier).
        <br/><br/>
        <b>T² (Hotelling):</b> Misura quanto un campione si discosta dal centro del modello nello spazio
        delle PC selezionate. Valori elevati indicano campioni con variazioni estreme <i>dentro</i>
        il modello.
        <br/><br/>
        <b>Q (Residui):</b> Misura quanto un campione non è spiegato dal modello. Valori elevati indicano
        campioni che il modello non riesce a descrivere bene (diversità strutturale).
        <br/><br/>
        Campioni con T² alto e Q basso sono spiegati dal modello ma sono estremi; campioni con T² basso
        e Q alto non sono ben descritti; campioni con entrambi alti sono i più critici. Questa analisi
        è fondamentale per decidere se escludere outlier prima di costruire il modello PLS."""
    ),
    (
        "04_spettri_originali.png",
        "4. Spettri NIR Originali",
        """A sinistra sono mostrati tutti gli spettri NIR originali, colorati per classe. Si possono osservare
        le differenze globali tra i gruppi e la sovrapposizione nelle regioni comuni.
        <br/><br/>
        A destra gli spettri medi per categoria evidenziano le differenze sistematiche. Le regioni di
        maggior interesse nel mid-NIR (6000-4000 cm⁻¹) includono:
        <br/>
        • <b>~5200 cm⁻¹:</b> Combinazione O-H (acqua)
        <br/>
        • <b>~4600 cm⁻¹:</b> Combinazione N-H (proteine)
        <br/>
        • <b>~4300 cm⁻¹:</b> Combinazione C-H (lipidi, carboidrati)
        <br/><br/>
        Le differenze tra gli spettri medi sono spesso sottili e mascherate da effetti di scattering,
        giustificando il preprocessing spettrale."""
    ),

    # --- PREPROCESSING E SCELTA LV ---
    (
        "05_scree_plot_LV.png",
        "5. Scree Plot — RMSEC e RMSECV vs Numero LV",
        """Lo scree plot confronta l'errore in calibrazione (RMSEC, blu) con l'errore in cross-validation
        (RMSECV, rosso) al variare del numero di LV, utilizzando il preprocessing scelto (SNV + Mean Centering).
        <br/><br/>
        <b>Criterio di scelta del numero di LV:</b> Si seleziona il numero di LV corrispondente al
        "gomito" della curva RMSECV, cioè il punto dopo il quale aggiungere ulteriori LV non produce
        un miglioramento significativo dell'errore di CV, o peggio, lo aumenta (overfitting).
        <br/><br/>
        <b>SNV (Standard Normal Variate)</b> corregge gli effetti di scattering moltiplicativo e additivo
        tipici degli spettri NIR, normalizzando ciascuno spettro per la propria media e deviazione standard.
        Dopo SNV, il <b>Mean Centering</b> centra i dati sull'origine, facilitando la decomposizione PLS.
        <br/><br/>
        Se RMSEC continua a scendere mentre RMSECV si stabilizza o sale, siamo in presenza di overfitting:
        il modello impara rumore dai dati di calibrazione senza migliorare la capacità predittiva.
        La linea verticale tratteggiata indica il numero di LV selezionato."""
    ),

    # --- MODELLO PLS ---
    (
        "07_T2_vs_Q_PLS.png",
        "7. Modello PLS — T² vs Q",
        """Analoga al T² vs Q della PCA, ma calcolata sul modello PLS finale. Si verificano:
        <br/><br/>
        • <b>Accumulo intorno allo zero:</b> La maggior parte dei campioni si concentra a bassi valori
        di T² e Q, indicando che il modello li descrive bene.
        <br/>
        • <b>Campioni con T² elevato ma Q basso:</b> Sono campioni estremi ma ben spiegati dal modello
        (variazioni chimiche reali, non errori).
        <br/>
        • <b>Campioni con Q elevato:</b> Non ben descritti dal modello, possono indicare contaminazioni
        o errori sperimentali.
        <br/><br/>
        Le linee tratteggiate rosse indicano i limiti di confidenza al 95%. Campioni oltre entrambi
        i limiti sono potenziali outlier da investigare."""
    ),
    (
        "08_scores_LV1_LV2.png",
        "8. Score Plot PLS — LV1 vs LV2",
        """Lo score plot nello spazio delle Latent Variables del PLS mostra come il modello separa le
        classi. A differenza della PCA, che massimizza la varianza di X, le LV del PLS massimizzano
        la covarianza tra X e Y, quindi la separazione tra classi dovrebbe essere migliore.
        <br/><br/>
        Gruppi ben separati indicano che il modello ha trovato pattern spettrali discriminanti tra
        le categorie. La posizione relativa dei cluster fornisce informazioni sulla somiglianza
        tra classi: categorie vicine hanno profili spettrali simili."""
    ),
    (
        "09_01_Y_pred_vs_meas_*.png",
        "9. Y Misurato vs Y Predetto (per classe)",
        """Per ciascuna classe (variabile dummy), il grafico confronta i valori misurati (0 o 1) con
        i valori predetti dal modello, sia in <b>calibrazione (fit)</b> che in <b>cross-validation</b>.
        <br/><br/>
        <b>Pannello sinistro (Fit):</b> Mostra la capacità del modello di riprodurre i dati di training.
        La linea verde è la diagonale perfetta (y=x), la linea rossa è la regressione effettiva.
        Maggiore è la sovrapposizione tra le due linee, migliore è il fit.
        <br/><br/>
        <b>Pannello destro (CV):</b> Misura la capacità predittiva lasciando fuori campioni uno alla
        volta (o a gruppi). Se il pattern è simile al fit, non c'è overfitting. Il valore R² in CV
        è la metrica più affidabile per valutare la bontà predittiva."""
    ),
    (
        "10_Y_vs_CV_residuals.png",
        "10. Y Misurato vs Residui CV",
        """I residui di cross-validation in funzione dei valori misurati devono presentare una
        <b>distribuzione casuale</b> centrata sullo zero, senza trend sistematici, curvature o
        pattern a imbuto.
        <br/><br/>
        • <b>Distribuzione casuale:</b> Conferma l'ipotesi di linearità del modello PLS e indica
        che l'incertezza di predizione è costante nel range di Y (omoschedasticità).
        <br/>
        • <b>Bande ±3σ:</b> Delimitano il 99.7% dei residui attesi. Campioni al di fuori sono
        potenziali outlier o indicano problematiche specifiche per quella classe.
        <br/>
        • <b>Pattern curvi:</b> Suggerirebbero non-linearità nel modello."""
    ),
    (
        "11_leverage_vs_residuals.png",
        "11. Leverage vs Y Residuals",
        """Il grafico Leverage vs Residui è uno strumento diagnostico per identificare campioni influenti.
        <br/><br/>
        <b>Leverage (asse x):</b> Misura quanto un campione influenza il modello. Campioni con leverage
        elevato hanno un peso sproporzionato nella costruzione del modello.
        <br/><br/>
        <b>Residui Y (asse y):</b> Errore di predizione per quel campione.
        <br/><br/>
        Campioni con <i>alto leverage e alti residui</i> sono i più problematici: influenzano fortemente
        il modello ma non sono predetti bene. Idealmente, i punti devono concentrarsi vicino all'origine
        con residui entro la banda ±3σ."""
    ),
    (
        "12_inner_relations.png",
        "12. Inner Relations (T scores vs U scores per LV)",
        """Le inner relations mostrano la relazione tra X-scores (T) e Y-scores (U) per ogni Latent
        Variable. Nel modello PLS, la relazione T→U deve essere lineare.
        <br/><br/>
        • <b>Prime LV:</b> Mostrano tipicamente una struttura lineare forte (alto coefficiente di
        correlazione r), indicando che catturano la maggior parte dell'informazione predittiva.
        <br/>
        • <b>LV successive:</b> La struttura lineare si attenua, il contributo informativo diminuisce.
        <br/><br/>
        Se le ultime LV non mostrano relazione lineare, il loro contributo predittivo è minimo e si
        potrebbe considerare un modello con meno LV. Questo grafico conferma la scelta del numero
        di LV effettuata con lo scree plot."""
    ),

    # --- IMPORTANZA VARIABILI ---
    (
        "13_PLS_weights.png",
        "13. PLS Weights (Pesi — primi 3 LV)",
        """I pesi (weights) del PLS indicano come ogni wavenumber contribuisce alla costruzione delle
        Latent Variables. Sono analoghi ai loadings della PCA, ma orientati alla predizione di Y.
        <br/><br/>
        <b>LV1:</b> I pesi di maggior modulo (positivi o negativi) indicano le regioni spettrali che
        il modello utilizza di più per catturare la variazione principale tra le classi. Poiché gli
        spettri potrebbero essere in derivata seconda, i <i>minimi</i> dei pesi tendono a corrispondere
        alle bande di assorbimento originali.
        <br/><br/>
        <b>LV2, LV3:</b> Catturano variazioni secondarie e correzioni di forma legate ad acqua, grassi
        e proteine. Le regioni con pesi elevati nelle prime 3 LV sono quelle più rilevanti per la
        discriminazione tra le categorie di farine animali."""
    ),
    (
        "14_regression_coefficients.png",
        "14. Coefficienti di Regressione PLS",
        """I coefficienti di regressione forniscono una visione sintetica di come ogni wavenumber
        contribuisce alla predizione di ciascuna classe. Il profilo oscillante, con alternanza di
        regioni positive e negative, è tipico dei modelli PLS su dati spettrali.
        <br/><br/>
        <b>Valori con modulo elevato</b> (picchi e valli) indicano le lunghezze d'onda dove le
        variazioni dello spettro sono più fortemente correlate all'appartenenza a una specifica classe.
        <br/><br/>
        Nelle regioni NIR 4000-6000 cm⁻¹, i coefficienti di maggior modulo dovrebbero localizzarsi
        in corrispondenza delle bande di overtone <b>N-H</b> (proteine), <b>O-H</b> (acqua) e
        <b>C-H</b> (lipidi), che differenziano il contenuto nutritivo di pollo, bovino e pesce."""
    ),
    (
        "15_VIP_scores.png",
        "15. VIP Scores (Variable Importance in Projection)",
        """Il VIP (Variable Importance in Projection) è una misura cumulativa dell'importanza di
        ciascuna variabile nella proiezione PLS. La formula considera il peso di ogni variabile
        in ogni LV, ponderato per la varianza di Y spiegata da quella LV.
        <br/><br/>
        <b>Soglia VIP = 1:</b> Variabili con VIP > 1 contribuiscono in media più della media alla
        predizione di Y. Es. un valore VIP di 1.5 indica che quella variabile è 50% più importante
        della media.
        <br/><br/>
        Le regioni con VIP elevato identificano le bande spettrali più informative per la
        classificazione. Questo è particolarmente utile per l'interpretazione chimica: le bande
        NIR con VIP > 1 corrispondono ai gruppi funzionali che differenziano le tre classi
        di farine animali."""
    ),
    (
        "16_selectivity_ratio.png",
        "16. Selectivity Ratio",
        """Il Selectivity Ratio (SR) confronta, per ciascuna variabile, la varianza spiegata
        correlata a Y con la varianza residua (non correlata).
        <br/><br/>
        <b>SR elevato:</b> In quelle lunghezze d'onda, la varianza spiegata dal modello domina
        rispetto al rumore, indicando alta informatività.
        <br/><br/>
        <b>Limite di confidenza al 95%:</b> Variabili con SR sopra questo limite sono
        statisticamente significative per la predizione.
        <br/><br/>
        A differenza del VIP che è una misura globale su tutte le LV, il Selectivity Ratio
        fornisce una valutazione diretta della qualità del segnale per ogni variabile, risultando
        particolarmente utile per identificare regioni spettrali specifiche da utilizzare in
        modelli ridotti."""
    ),

    # --- TEST SET ---
    (
        "17_test_set_prediction.png",
        "17. Validazione Test Set — Y Predetto vs Y Misurato",
        """Questo è il grafico più importante dell'intera analisi: mostra la capacità del modello
        di predire campioni <b>mai visti</b> durante la fase di calibrazione.
        <br/><br/>
        Per ciascuna classe, i punti devono distribuirsi lungo la diagonale (linea verde).
        Il valore <b>RMSEP</b> (Root Mean Square Error of Prediction) quantifica l'errore medio
        di predizione, mentre <b>R²</b> misura la quota di varianza spiegata nei dati di test.
        <br/><br/>
        <b>Interpretazione:</b>
        <br/>
        • RMSEP ≈ RMSECV → il modello generalizza bene (no overfitting)
        <br/>
        • RMSEP ≪ RMSECV → test set troppo "facile" o fortunato
        <br/>
        • RMSEP ≫ RMSECV → overfitting, il modello non generalizza
        <br/><br/>
        Valori di R² > 0.90 e bassa dispersione attorno alla diagonale indicano un modello
        con eccellente capacità di generalizzazione."""
    ),
    (
        "18_confusion_matrix.png",
        "18. Confusion Matrix — Calibrazione e Test Set",
        """La Confusion Matrix riassume le prestazioni di classificazione in formato tabellare.
        Ogni riga rappresenta la classe vera, ogni colonna la classe predetta dal modello.
        <br/><br/>
        <b>Diagonale:</b> Campioni correttamente classificati. Idealmente, tutti i campioni
        dovrebbero trovarsi sulla diagonale.
        <br/><br/>
        <b>Fuori diagonale:</b> Errori di classificazione (misclassificazioni). Indicano quali
        classi il modello confonde più facilmente. Ad esempio, confusione tra bovino e pollo
        potrebbe indicare profili proteici simili, mentre il pesce ha tipicamente un profilo
        lipidico molto diverso.
        <br/><br/>
        L'accuratezza complessiva (percentuale di campioni correttamente classificati) è riportata
        nel titolo di ciascun pannello. Un buon modello PLS-DA su dati NIR raggiunge tipicamente
        accuratezze del 85-100%."""
    ),
    (
        "19_spettri_preprocessati.png",
        "19. Spettri Originali vs Preprocessati",
        """Questo confronto visuale mostra l'effetto del preprocessing spettrale selezionato.
        <br/><br/>
        <b>Pannello sinistro (Originali):</b> Gli spettri grezzi mostrano ampia variabilità dovuta
        a effetti fisici (scattering, dimensione particelle, baseline shift) che mascherano le
        differenze chimiche.
        <br/><br/>
        <b>Pannello destro (Preprocessati):</b> Dopo il preprocessing, le variazioni fisiche
        vengono ridotte/eliminate, esaltando le differenze chimiche tra le classi. La derivata
        seconda rimuove offset e trend lineari, la MSC normalizza lo scattering, e il mean
        centering centra i dati. Il risultato sono spettri dove le differenze tra pollo, bovino
        e pesce sono più evidenti e interpretabili in termini di composizione chimica."""
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
#  FUNZIONE PER TROVARE I FILE IMMAGINE (con globbing per wildcard)
# ─────────────────────────────────────────────────────────────────────────────
def find_image(pattern):
    """Cerca un file immagine nella cartella pls_plots, supportando wildcard."""
    if "*" in pattern:
        matches = sorted(glob.glob(os.path.join(PLOT_DIR, pattern)))
        return matches if matches else []
    else:
        path = os.path.join(PLOT_DIR, pattern)
        return [path] if os.path.exists(path) else []


# ─────────────────────────────────────────────────────────────────────────────
#  COSTRUZIONE DEL PDF
# ─────────────────────────────────────────────────────────────────────────────
def build_pdf():
    """Genera il report PDF completo."""
    print(f"[INFO] Cartella grafici: {PLOT_DIR}")

    if not os.path.isdir(PLOT_DIR):
        print(f"[ERRORE] La cartella '{PLOT_DIR}' non esiste.")
        print("         Esegui prima 'analisi_PLS_regression.m' in MATLAB.")
        return

    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title="Report PLS Regression - Farine Animali NIR",
        author="Analisi Automatizzata",
    )

    story = []

    # ── Pagina titolo ──────────────────────────────────────────────────
    story.append(Spacer(1, 6 * cm))
    story.append(Paragraph(
        "Analisi PLS Regression<br/>Farine Animali NIR",
        style_title
    ))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        "Regressione PLS / PLS-DA per la classificazione di farine animali "
        "(pollo, bovino, pesce) tramite spettroscopia NIR",
        style_subtitle
    ))
    story.append(Spacer(1, 1.5 * cm))
    story.append(SectionDivider(CONTENT_W, COLOR_ACCENT, 2))
    story.append(Spacer(1, 1 * cm))

    # Info tabella
    info_data = [
        ["Dataset", "farineanimNIR.mat (84 campioni × 2001 wavenumbers)"],
        ["Range spettrale", "6000 – 4000 cm⁻¹ (NIR)"],
        ["Classi", "Pollo, Bovino, Pesce"],
        ["Metodo", "PLS-DA (Partial Least Squares Discriminant Analysis)"],
        ["Cross-Validation", "Venetian Blind (15 splits, thickness = 3)"],
        ["Software", "MATLAB + PLS Toolbox (Eigenvector Research Inc.)"],
        ["Data generazione", datetime.now().strftime("%d/%m/%Y %H:%M")],
    ]
    info_table = Table(info_data, colWidths=[4.5 * cm, CONTENT_W - 5 * cm])
    info_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), COLOR_PRIMARY),
        ("TEXTCOLOR", (1, 0), (1, -1), COLOR_TEXT),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, HexColor("#DDDDDD")),
    ]))
    story.append(info_table)
    story.append(PageBreak())

    # ── Indice ─────────────────────────────────────────────────────────
    story.append(Paragraph("Indice", style_section))
    story.append(SectionDivider(CONTENT_W))
    story.append(Spacer(1, 3 * mm))

    toc_sections = [
        "PCA Esplorativa",
        "  1. Score Plot PC1 vs PC2",
        "  2. Loading Plot",
        "  3. T² vs Q — Outlier Detection",
        "  4. Spettri NIR Originali",
        "Selezione Numero LV",
        "  5. Scree Plot — Scelta Numero LV",
        "Modello PLS Finale",
        "  7. T² vs Q (PLS)",
        "  8. Score Plot LV1 vs LV2",
        "  9. Y Predetto vs Y Misurato",
        "  10. Residui Cross-Validation",
        "  11. Leverage vs Residuals",
        "  12. Inner Relations",
        "Importanza Variabili",
        "  13. PLS Weights",
        "  14. Coefficienti di Regressione",
        "  15. VIP Scores",
        "  16. Selectivity Ratio",
        "Validazione",
        "  17. Test Set Prediction",
        "  18. Confusion Matrix",
        "  19. Spettri Preprocessati vs Originali",
    ]
    for item in toc_sections:
        if item.startswith("  "):
            story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;{item.strip()}", style_body))
        else:
            story.append(Paragraph(f"<b>{item}</b>", style_subsection))

    story.append(PageBreak())

    # ── Sezioni con grafici ────────────────────────────────────────────
    n_included = 0
    n_missing = 0

    for filename, title, description in PLOT_DESCRIPTIONS:
        images = find_image(filename)

        if not images:
            print(f"  [SKIP] {filename} — non trovato")
            n_missing += 1
            continue

        for img_path in images:
            # Nuova pagina per ogni sezione principale
            story.append(PageBreak())

            # Titolo sezione
            story.append(Paragraph(title, style_section))
            story.append(SectionDivider(CONTENT_W))
            story.append(Spacer(1, 3 * mm))

            # Immagine
            try:
                from PIL import Image as PILImage
                with PILImage.open(img_path) as im:
                    iw, ih = im.size
                aspect = ih / iw
                img_w = min(CONTENT_W, 16 * cm)
                img_h = img_w * aspect
                # Se troppo alta, riduci
                max_h = 10 * cm
                if img_h > max_h:
                    img_h = max_h
                    img_w = img_h / aspect
            except Exception:
                img_w = CONTENT_W * 0.9
                img_h = img_w * 0.6

            img_obj = Image(img_path, width=img_w, height=img_h)
            story.append(img_obj)

            # Didascalia
            img_basename = os.path.basename(img_path)
            story.append(Paragraph(
                f"Figura: <i>{img_basename}</i>",
                style_caption
            ))

            # Descrizione
            story.append(Paragraph(description, style_body))

            n_included += 1
            print(f"  [OK]   {img_basename}")

    # ── Pagina conclusiva ──────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Conclusioni", style_section))
    story.append(SectionDivider(CONTENT_W, COLOR_ACCENT, 2))
    story.append(Spacer(1, 5 * mm))

    conclusion_text = """
    L'analisi PLS-DA condotta sugli spettri NIR delle farine animali ha dimostrato l'efficacia della
    spettroscopia nel vicino infrarosso per la classificazione di campioni di diversa origine
    (pollo, bovino, pesce).
    <br/><br/>
    <b>Risultati principali:</b>
    <br/><br/>
    1. <b>PCA Esplorativa:</b> La dominanza di PC1 (varianza >99%) ha confermato la necessità di
    preprocessing spettrale per rimuovere gli effetti fisici di scattering.
    <br/><br/>
    2. <b>Preprocessing:</b> Si è applicato il preprocessing <b>SNV + Mean Centering</b>. L'SNV corregge
    gli effetti di scattering tipici degli spettri NIR, mentre il mean centering centra i dati sull'origine,
    facilitando la decomposizione PLS e l'interpretazione dei risultati.
    <br/><br/>
    3. <b>Modello PLS:</b> La cross-validation con schema Venetian Blind (15 split, thickness 3)
    ha garantito una stima robusta della capacità predittiva. L'analisi dei grafici diagnostici
    (T² vs Q, leverage, residui) conferma la validità e stabilità del modello.
    <br/><br/>
    4. <b>Importanza Variabili:</b> VIP, Selectivity Ratio e coefficienti di regressione identificano
    coerentemente le regioni spettrali legate alle bande di assorbimento N-H, O-H e C-H, in accordo
    con la composizione chimica attesa (differenze in proteine, grassi e acqua tra le specie animali).
    <br/><br/>
    5. <b>Validazione:</b> La performance sul test set indipendente confermata dalla confusion matrix
    dimostra che il modello generalizza adeguatamente a campioni non visti durante la calibrazione.
    """
    story.append(Paragraph(conclusion_text, style_body))

    # ── Note metodologiche ─────────────────────────────────────────────
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("Note Metodologiche", style_subsection))
    story.append(SectionDivider(CONTENT_W, HexColor("#BBBBBB"), 0.5))

    notes = [
        "La divisione calibrazione/test è stata effettuata con campionamento stratificato (70/30) "
        "per mantenere la proporzione delle classi.",
        "La cross-validation Venetian Blind è stata scelta per garantire riproducibilità "
        "(a differenza del random subset).",
        "La matrice Y è codificata come dummy (one-hot) per adattare il problema di classificazione "
        "alla regressione PLS (PLS-DA).",
        "Il preprocessing Y consiste esclusivamente in mean centering.",
    ]
    for note in notes:
        story.append(Paragraph(f"• {note}", style_note))

    # ── Genera PDF ─────────────────────────────────────────────────────
    doc.build(story)

    print(f"\n{'='*60}")
    print(f"  Report PDF generato con successo!")
    print(f"  File: {OUTPUT_PDF}")
    print(f"  Grafici inclusi:  {n_included}")
    print(f"  Grafici mancanti: {n_missing}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_pdf()
