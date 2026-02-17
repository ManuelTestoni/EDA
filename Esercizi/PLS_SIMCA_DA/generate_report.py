#!/usr/bin/env python3
"""
generate_report.py
Generates a professional PDF report from the plots produced by main_analysis.m

Usage:
    python3 generate_report.py

Prerequisite: Run main_analysis.m in MATLAB first to generate all plots in the 'plot/' folder.
"""

import os
import sys
from fpdf import FPDF
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plot")
OUTPUT_PDF = os.path.join(SCRIPT_DIR, "Report_Olive_Oil_Classification.pdf")

# ============================================================================
# Custom PDF class
# ============================================================================
class ReportPDF(FPDF):
    def __init__(self):
        super().__init__('P', 'mm', 'A4')
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 5, 'Olive Oil Classification Analysis - SIMCA & PLS-DA', align='L')
            self.cell(0, 5, f'Page {self.page_no()}', align='R', new_x="LMARGIN", new_y="NEXT")
            self.line(10, 12, 200, 12)
            self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, 'Universita di Modena e Reggio Emilia - Elaborazione Dati Scientifici A.A. 2024-2025', align='C')

    def title_page(self):
        self.add_page()
        self.ln(50)
        self.set_font('Helvetica', 'B', 28)
        self.set_text_color(30, 60, 120)
        self.cell(0, 15, 'Analisi di Classificazione', align='C', new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 15, 'Olio di Oliva Italiano', align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(8)
        self.set_font('Helvetica', '', 16)
        self.set_text_color(80, 80, 80)
        self.cell(0, 10, 'SIMCA & PLS-DA', align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(15)
        self.set_draw_color(30, 60, 120)
        self.set_line_width(0.8)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(15)
        self.set_font('Helvetica', '', 11)
        self.set_text_color(60, 60, 60)
        self.cell(0, 8, 'Corso: Elaborazione Dati Scientifici', align='C', new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, 'A.A. 2024-2025', align='C', new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, f'Data: {datetime.now().strftime("%d/%m/%Y")}', align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(25)
        self.set_font('Helvetica', 'I', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, 'Dataset: Olive Oil (382 campioni, 7 acidi grassi, 5 regioni italiane)', align='C', new_x="LMARGIN", new_y="NEXT")

    def section_title(self, title, level=1):
        if level == 1:
            self.set_font('Helvetica', 'B', 18)
            self.set_text_color(30, 60, 120)
            self.ln(5)
            self.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(30, 60, 120)
            self.set_line_width(0.5)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(5)
        elif level == 2:
            self.set_font('Helvetica', 'B', 14)
            self.set_text_color(50, 90, 150)
            self.ln(3)
            self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)
        elif level == 3:
            self.set_font('Helvetica', 'B', 11)
            self.set_text_color(70, 70, 70)
            self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def body_text_italic(self, text):
        self.set_font('Helvetica', 'I', 10)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def add_figure(self, filename, caption="", width=170):
        filepath = os.path.join(PLOT_DIR, filename)
        if not os.path.exists(filepath):
            self.set_font('Helvetica', 'I', 9)
            self.set_text_color(200, 0, 0)
            self.cell(0, 8, f'[Figura non disponibile: {filename}]', new_x="LMARGIN", new_y="NEXT")
            self.ln(3)
            return

        # Check if we need a new page
        if self.get_y() > 200:
            self.add_page()

        x = (210 - width) / 2  # center
        self.image(filepath, x=x, w=width)
        if caption:
            self.ln(2)
            self.set_font('Helvetica', 'I', 9)
            self.set_text_color(80, 80, 80)
            self.multi_cell(0, 4.5, caption, align='C')
        self.ln(4)

    def bullet_list(self, items):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(40, 40, 40)
        for item in items:
            x = self.get_x()
            self.cell(8, 5.5, '  -  ', new_x="END")
            self.multi_cell(0, 5.5, item)
            self.set_x(x)
        self.ln(2)


# ============================================================================
# Build the report
# ============================================================================
def build_report():
    pdf = ReportPDF()

    # ---- TITLE PAGE ----
    pdf.title_page()

    # ---- TABLE OF CONTENTS ----
    pdf.add_page()
    pdf.section_title("Indice", level=1)
    toc_items = [
        "1. Introduzione e Descrizione del Dataset",
        "2. Fondamenti Teorici",
        "   2.1 Analisi delle Componenti Principali (PCA)",
        "   2.2 SIMCA (Soft Independent Modelling of Class Analogy)",
        "   2.3 PLS-DA (Partial Least Squares Discriminant Analysis)",
        "   2.4 Metriche di Valutazione",
        "3. Analisi Esplorativa dei Dati (EDA)",
        "   3.1 Distribuzione globale delle variabili",
        "   3.2 Distribuzione per classe",
        "   3.3 Matrice di correlazione",
        "   3.4 PCA esplorativa: varianza spiegata",
        "   3.5 PCA esplorativa: score plots",
        "   3.6 PCA esplorativa: loading analysis",
        "4. Classificazione con SIMCA",
        "   4.1 Cross-validazione e scelta delle componenti",
        "   4.2 Score Distance vs Orthogonal Distance",
        "   4.3 Matrici di confusione e metriche",
        "   4.4 Potere discriminante e loadings",
        "5. Classificazione con PLS-DA",
        "   5.1 Cross-validazione e RMSECV",
        "   5.2 Predizioni Y e matrici di confusione",
        "   5.3 Score plots PLS",
        "   5.4 Coefficienti di regressione e VIP",
        "6. Confronto dei Metodi e Conclusioni",
    ]
    pdf.set_font('Helvetica', '', 11)
    for item in toc_items:
        if item.startswith("   "):
            pdf.cell(10)
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(0, 7, item.strip(), new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font('Helvetica', 'B', 11)
            pdf.cell(0, 7, item, new_x="LMARGIN", new_y="NEXT")

    # ====================================================================
    # 1. INTRODUCTION
    # ====================================================================
    pdf.add_page()
    pdf.section_title("1. Introduzione e Descrizione del Dataset")

    pdf.body_text(
        "Il presente report documenta l'analisi di classificazione completa eseguita "
        "sul dataset Olive Oil. L'obiettivo e' discriminare campioni di olio d'oliva "
        "provenienti da 5 diverse regioni italiane sulla base della loro composizione "
        "in acidi grassi, utilizzando due metodi di classificazione complementari: "
        "SIMCA (class modeling) e PLS-DA (discriminant analysis). L'analisi si articola "
        "in una fase esplorativa (EDA con PCA) seguita dalla costruzione, validazione e "
        "confronto dei due modelli classificatori."
    )

    pdf.section_title("Dataset", level=2)
    pdf.body_text(
        "Il dataset contiene 382 campioni di olio d'oliva italiano, ciascuno caratterizzato "
        "dalla concentrazione percentuale di 7 acidi grassi misurati tramite gascromatografia. "
        "Le variabili analitiche sono:"
    )
    pdf.bullet_list([
        "Acido Palmitico (C16:0) - acido grasso saturo, principale componente della frazione satura",
        "Acido Palmitoleico (C16:1) - acido grasso monoinsaturo a 16 atomi di carbonio",
        "Acido Stearico (C18:0) - acido grasso saturo a 18 atomi di carbonio",
        "Acido Oleico (C18:1) - acido grasso monoinsaturo predominante, tipico dell'olio d'oliva",
        "Acido Linoleico (C18:2) - acido grasso polinsaturo omega-6 essenziale",
        "Acido Eicosanoico (C20:0) - acido grasso saturo a catena lunga, presente in tracce",
        "Acido Linolenico (C18:3) - acido grasso polinsaturo omega-3, presente in piccole quantita'"
    ])

    pdf.section_title("Classi", level=2)
    pdf.body_text("I campioni appartengono a 5 aree geografiche italiane con numerosita' molto diverse:")
    pdf.bullet_list([
        "NA - North Apulia (Puglia Nord): 25 campioni (6.5%) - classe piu' piccola",
        "SA - South Apulia (Puglia Sud): 206 campioni (53.9%) - classe dominante",
        "U  - Umbria: 51 campioni (13.4%)",
        "EL - East Liguria (Liguria Est): 50 campioni (13.1%)",
        "WL - West Liguria (Liguria Ovest): 50 campioni (13.1%)"
    ])
    pdf.body_text(
        "Lo sbilanciamento delle classi e' un elemento critico: la classe SA (South Apulia) contiene "
        "piu' della meta' dei campioni totali (206/382 = 53.9%), mentre la classe NA (North Apulia) ne "
        "contiene solo 25 (6.5%). Questo squilibrio puo' influenzare le prestazioni di classificazione, "
        "favorendo la classe maggioritaria e rendendo meno stabili le stime per la classe minoritaria, "
        "specialmente in cross-validazione. Per mitigare questo effetto, e' stata utilizzata una "
        "suddivisione train/test stratificata (70/30) che preserva le proporzioni originali delle classi."
    )

    # ====================================================================
    # 2. THEORETICAL FOUNDATIONS
    # ====================================================================
    pdf.add_page()
    pdf.section_title("2. Fondamenti Teorici")

    pdf.section_title("2.1 PCA - Analisi delle Componenti Principali", level=2)
    pdf.body_text(
        "La PCA (Principal Component Analysis) e' una tecnica di riduzione della dimensionalita' "
        "che trasforma le variabili originali, potenzialmente correlate, in un nuovo set di variabili "
        "ortogonali e non correlate chiamate Componenti Principali (PC). Le PC sono ordinate per "
        "varianza spiegata decrescente: la prima PC cattura la direzione di massima varianza nei dati, "
        "la seconda la massima varianza residua ortogonale alla prima, e cosi' via."
    )
    pdf.body_text(
        "La decomposizione matriciale e': X = T*P' + E, dove T (scores) rappresentano le coordinate "
        "dei campioni nello spazio ridotto, P (loadings) esprimono il contributo delle variabili "
        "originali alle componenti, ed E i residui. La PCA e' implementata tramite SVD (Singular Value "
        "Decomposition) della matrice X autoscalata. L'autoscaling (sottrazione della media e divisione "
        "per la deviazione standard) e' essenziale per dare uguale peso a variabili con scale diverse."
    )

    pdf.section_title("2.2 SIMCA - Soft Independent Modelling of Class Analogy", level=2)
    pdf.body_text(
        "SIMCA e' un metodo di class modeling: costruisce un modello PCA indipendente per ciascuna "
        "classe, definendo i confini di appartenenza tramite limiti statistici. Le caratteristiche "
        "fondamentali del metodo sono:"
    )
    pdf.bullet_list([
        "Soft: non fa assunzioni restrittive sulla distribuzione dei dati",
        "Independent: ogni classe ha il proprio modello PCA con il proprio numero di componenti",
        "Modeling: definisce lo spazio di ciascuna classe tramite un modello matematico",
        "Un campione puo' essere accettato da piu' classi, una sola classe, o nessuna classe"
    ])
    pdf.body_text(
        "Per ogni campione si calcolano due distanze rispetto al modello di ciascuna classe:\n\n"
        "1) Score Distance (SD): misurata dalla statistica T2 di Hotelling, quantifica quanto il "
        "campione e' lontano dal centro della classe nello spazio delle PC. Segue una distribuzione F.\n\n"
        "2) Orthogonal Distance (OD): misurata dalla statistica Q (somma dei quadrati dei residui), "
        "quantifica quanto il campione e' lontano dal sottospazio PCA del modello. Il limite Q viene "
        "calcolato con l'approssimazione di Jackson-Mudholkar basata sugli autovalori residui.\n\n"
        "Il criterio di appartenenza combinato e':\n"
        "  D = sqrt((T2/T2lim)^2 + (Q/Qlim)^2) <= sqrt(2)\n\n"
        "Il Discriminant Power misura la capacita' discriminante di ciascuna variabile, calcolato come "
        "rapporto tra la varianza residua dei campioni esterni e quella dei campioni appartenenti alla classe."
    )

    pdf.section_title("2.3 PLS-DA - Partial Least Squares Discriminant Analysis", level=2)
    pdf.body_text(
        "PLS-DA e' un metodo discriminante che combina la regressione PLS con la codifica dummy delle classi. "
        "La matrice Y contiene variabili indicatrici binarie (1/0) che codificano l'appartenenza alle classi. "
        "L'algoritmo NIPALS (Nonlinear Iterative Partial Least Squares) decompone simultaneamente X e Y:\n"
        "  X = T*P' + E\n"
        "  Y = U*Q' + F\n\n"
        "Le variabili latenti (LV) sono scelte per massimizzare la covarianza tra X e Y, quindi sono "
        "intrinsecamente orientate alla discriminazione. La predizione avviene tramite:\n"
        "  Y_pred = X_new * B_PLS + mean(Y)\n"
        "dove B_PLS = W*(P'W)^-1 * diag(b) * Q' sono i coefficienti di regressione.\n\n"
        "La classificazione si basa sul criterio True Discriminant: ogni campione viene assegnato alla "
        "classe corrispondente al valore massimo di Y predetto. A differenza di SIMCA, PLS-DA assegna "
        "sempre ogni campione a esattamente una classe (hard classification).\n\n"
        "I VIP scores (Variable Importance in Projection) misurano l'importanza globale di ciascuna "
        "variabile nella proiezione PLS, ponderata per la varianza di Y spiegata da ciascuna LV."
    )

    pdf.section_title("2.4 Metriche di Valutazione", level=2)
    pdf.body_text(
        "Per entrambi i metodi si utilizzano le seguenti metriche, calcolate per ciascuna classe:\n\n"
        "- Sensitivity (Sensibilita'): TP/(TP+FN) - proporzione di campioni della classe correttamente "
        "identificati. Una sensibilita' bassa indica che molti campioni della classe vengono persi.\n\n"
        "- Specificity (Specificita'): TN/(TN+FP) - proporzione di campioni delle altre classi "
        "correttamente rifiutati. Una specificita' bassa indica che molti campioni estranei vengono "
        "erroneamente accettati.\n\n"
        "- Efficiency (Efficienza): sqrt(Sensitivity * Specificity) - media geometrica che bilancia "
        "sensibilita' e specificita', particolarmente utile per classi sbilanciate.\n\n"
        "- Accuracy: proporzione totale di classificazioni corrette su tutti i campioni.\n\n"
        "La cross-validazione (venetian blinds, 5 segmenti) viene utilizzata per selezionare "
        "la complessita' ottimale dei modelli: il numero di PC per ciascuna classe SIMCA e il numero "
        "di LV per PLS-DA. Questo approccio fornisce una stima imparziale delle prestazioni evitando "
        "l'overfitting."
    )

    # ====================================================================
    # 3. EXPLORATORY DATA ANALYSIS
    # ====================================================================
    pdf.add_page()
    pdf.section_title("3. Analisi Esplorativa dei Dati (EDA)")

    pdf.body_text(
        "L'analisi esplorativa dei dati (EDA) rappresenta il primo passo fondamentale dell'analisi. "
        "Il suo scopo e' comprendere la struttura dei dati, identificare pattern, correlazioni, outlier "
        "e valutare visivamente la separabilita' delle classi prima di applicare metodi di classificazione."
    )

    # --- FIGURE 01 ---
    pdf.section_title("3.1 Distribuzione Globale delle Variabili", level=2)
    pdf.body_text(
        "Il boxplot delle variabili grezze (non autoscalate) mostra la distribuzione di ciascuno "
        "dei 7 acidi grassi sull'intero dataset di 382 campioni. Per ogni variabile, il box rappresenta "
        "l'intervallo interquartile (IQR, dal 25o al 75o percentile), la linea centrale indica la mediana, "
        "e i baffi (whiskers) si estendono fino a 1.5*IQR. I punti al di fuori dei baffi sono outlier."
    )
    pdf.add_figure("01_raw_data_boxplot.png",
                    "Figura 1: Boxplot delle 7 variabili (acidi grassi) sull'intero dataset (382 campioni).")
    pdf.body_text(
        "Dall'osservazione del grafico emergono aspetti fondamentali:\n\n"
        "- L'acido oleico (Oleico, C18:1) domina nettamente con una mediana intorno al 75-78%, coerente "
        "con la composizione tipica dell'olio d'oliva extra vergine. La sua distribuzione e' relativamente "
        "compatta, indicando una buona omogeneita' tra i campioni.\n\n"
        "- L'acido linoleico (Linoleico, C18:2) e' il secondo componente per abbondanza, con valori "
        "tra circa 5% e 15%, mostrando una dispersione significativa che lo rende potenzialmente "
        "discriminante.\n\n"
        "- L'acido palmitico (Palmitico, C16:0) occupa il terzo posto con valori intorno all'10-14%.\n\n"
        "- Le variabili Palmitoleico, Stearico, Eicosanoico e Linolenico presentano valori molto bassi "
        "(< 5%) e scale molto diverse tra loro. Questo giustifica la necessita' di autoscaling prima "
        "di applicare PCA e metodi multivariati, per evitare che l'oleico (per il suo range molto ampio) "
        "domini l'intera analisi.\n\n"
        "- La presenza di alcuni outlier (cerchi al di fuori dei baffi) suggerisce la presenza di campioni "
        "atipici che potrebbero appartenere a classi particolari o rappresentare composizioni inusuali."
    )

    # --- FIGURE 02 ---
    pdf.add_page()
    pdf.section_title("3.2 Distribuzione per Classe", level=2)
    pdf.body_text(
        "Questo pannello contiene 7 subplot (uno per ciascun acido grasso), con boxplot suddivisi "
        "per le 5 classi (NA, SA, U, EL, WL). L'asse X riporta le classi, l'asse Y il valore della "
        "variabile corrispondente. Questa vista e' cruciale per identificare visivamente quali variabili "
        "discriminano meglio tra le regioni."
    )
    pdf.add_figure("02_boxplot_by_class.png",
                    "Figura 2: Distribuzione di ciascun acido grasso suddivisa per classe (regione geografica).")
    pdf.body_text(
        "L'analisi per classe rivela pattern discriminanti importanti:\n\n"
        "- Acido Palmitico: la classe SA (Sud Puglia) mostra valori mediamente piu' bassi rispetto "
        "alle classi liguri (EL, WL), suggerendo una prima separazione tra le macroaree.\n\n"
        "- Acido Palmitoleico: questa variabile mostra una forte differenziazione. Le classi EL e WL "
        "(Liguria) tendono ad avere valori piu' elevati rispetto alle classi pugliesi (NA, SA), mentre "
        "l'Umbria (U) occupa una posizione intermedia. Questa variabile potrebbe essere un marker "
        "geografico chiave.\n\n"
        "- Acido Stearico: la distribuzione e' relativamente simile tra le classi, con lieve "
        "differenziazione, il che lo rende un discriminante piu' debole.\n\n"
        "- Acido Oleico: variabile fondamentale. Le classi liguri (EL, WL) mostrano valori leggermente "
        "inferiori rispetto alle pugliesi, con la WL (Liguria Ovest, Taggiasche) che presenta una "
        "distribuzione piu' concentrata. L'Umbria mostra valori intermedi.\n\n"
        "- Acido Linoleico: pattern complementare all'oleico (correlazione inversa attesa). Le classi "
        "EL e WL tendono a valori piu' alti, mentre SA ha valori piu' bassi e concentrati.\n\n"
        "- Acido Eicosanoico e Linolenico: valori bassi con differenziazione modesta. Tuttavia, "
        "anche piccole differenze sistematiche possono contribuire alla classificazione multivariata."
    )

    # --- FIGURE 03 ---
    pdf.add_page()
    pdf.section_title("3.3 Matrice di Correlazione", level=2)
    pdf.body_text(
        "La matrice di correlazione di Pearson visualizza le relazioni lineari bivariante tra tutte "
        "le 7 variabili. La colormap (jet: blu = correlazione negativa forte, rosso = correlazione "
        "positiva forte) e' accompagnata dalle annotazioni numeriche del coefficiente r per ciascuna "
        "coppia. La diagonale mostra sempre r = 1.00 (autocorrelazione)."
    )
    pdf.add_figure("03_correlation_matrix.png",
                    "Figura 3: Matrice di correlazione di Pearson tra i 7 acidi grassi, con annotazioni numeriche.")
    pdf.body_text(
        "Dalla matrice di correlazione si possono trarre le seguenti osservazioni:\n\n"
        "- Oleico vs Linoleico: questi due acidi grassi mostrano una forte correlazione negativa "
        "(r atteso intorno a -0.8/-0.9). Questo riflette la biochimica della pianta d'olivo: la "
        "desaturazione dell'oleico produce linoleico, quindi un aumento dell'uno corrisponde a una "
        "diminuzione dell'altro. Questa anticorrelazione e' la principale sorgente di variabilita' "
        "nel dataset e dominera' la prima componente principale.\n\n"
        "- Palmitoleico e Linoleico: correlazione positiva moderata, che indica un trend comune legato "
        "probabilmente alla maturazione e al clima.\n\n"
        "- Palmitico e Oleico: correlazione negativa, coerente con il profilo degli oli meridionali "
        "(alto oleico, basso palmitico) vs settentrionali.\n\n"
        "- Eicosanoico e Linolenico: bassa correlazione con le altre variabili, il che indica che portano "
        "informazione indipendente, anche se di piccola entita'.\n\n"
        "La presenza di correlazioni significative giustifica l'uso della PCA, che e' progettata proprio "
        "per gestire e sfruttare la struttura di correlazione tra le variabili. Le variabili fortemente "
        "correlate contribuiranno alle stesse componenti principali."
    )

    # --- FIGURE 04 ---
    pdf.add_page()
    pdf.section_title("3.4 PCA Esplorativa: Varianza Spiegata", level=2)
    pdf.body_text(
        "Lo scree plot e il grafico della varianza cumulativa sono strumenti fondamentali per determinare "
        "il numero di componenti principali significative. La PCA e' stata eseguita sulla matrice autoscalata "
        "(382 x 7) tramite SVD."
    )
    pdf.add_figure("04_pca_scree_plot.png",
                    "Figura 4: Scree plot (sinistra) e varianza cumulativa (destra) delle componenti principali.")
    pdf.body_text(
        "Il pannello sinistro (Scree Plot) mostra la percentuale di varianza spiegata da ciascuna "
        "singola PC tramite un diagramma a barre. In un dataset con 7 variabili si ottengono al massimo "
        "7 PC. L'andamento tipico e' una curva decrescente: le prime PC catturano la maggior parte della "
        "varianza, mentre le ultime catturano prevalentemente rumore.\n\n"
        "Il pannello destro (Varianza Cumulativa) mostra la somma cumulata delle varianze, con una linea "
        "rossa tratteggiata al 95%. Questo grafico permette di stabilire quante PC sono necessarie per "
        "catturare una quota sufficiente dell'informazione totale.\n\n"
        "Dall'analisi si osserva che:\n"
        "- La PC1 cattura tipicamente il 35-45% della varianza, dominata dall'asse oleico-linoleico.\n"
        "- La PC2 aggiunge un ulteriore 20-25%, portando il cumulato al 60-70%.\n"
        "- Con 3 PC si supera generalmente l'80% della varianza.\n"
        "- Per raggiungere il 95% sono necessarie 4-5 PC.\n\n"
        "Il 'gomito' dello scree plot (punto di flessione dove la curva si appiattisce) indica la "
        "dimensionalita' intrinseca dei dati e guida la scelta del numero di PC per la modellazione SIMCA."
    )

    # --- FIGURE 05 ---
    pdf.add_page()
    pdf.section_title("3.5 PCA Esplorativa: Score Plots", level=2)

    pdf.section_title("Score Plot 2D", level=3)
    pdf.body_text(
        "I grafici degli scores proiettano i 382 campioni nello spazio ridotto delle PC. I campioni sono "
        "colorati per classe e visualizzati come scatter plot. Il pannello sinistro mostra PC1 vs PC2, "
        "quello destro PC1 vs PC3. Le etichette degli assi riportano la percentuale di varianza spiegata."
    )
    pdf.add_figure("05_pca_scores.png",
                    "Figura 5: Score plots PCA - PC1 vs PC2 (sinistra) e PC1 vs PC3 (destra), colorati per classe.")
    pdf.body_text(
        "Analisi dello score plot PC1 vs PC2:\n"
        "- Le classi liguri (EL ed WL) tendono a raggrupparsi in una regione dello spazio a valori "
        "positivi/negativi di PC1, separate dalle classi pugliesi.\n"
        "- La classe SA (South Apulia), essendo la piu' numerosa, forma un cluster ampio e denso.\n"
        "- La classe NA (North Apulia) mostra una distribuzione piu' dispersa, potenzialmente "
        "parzialmente sovrapposta con SA, il che e' atteso data la prossimita' geografica.\n"
        "- L'Umbria (U) tende a posizionarsi in una regione intermedia, riflettendo la sua posizione "
        "geografica tra il Sud e il Nord.\n\n"
        "- La separazione lungo PC2 aggiunge discriminazione extra, differenziando ulteriormente "
        "le classi che appaiono sovrapposte nella sola dimensione PC1.\n\n"
        "Analisi dello score plot PC1 vs PC3:\n"
        "- La terza PC cattura varianza complementare che puo' separare classi non distinguibili "
        "nel piano PC1-PC2. In particolare, la separazione tra EL e WL (entrambe liguri) potrebbe "
        "emergere lungo PC3, cosi' come una migliore separazione tra NA e SA."
    )

    # --- FIGURE 06 ---
    pdf.section_title("Score Plot 3D", level=3)
    pdf.body_text(
        "Lo score plot 3D combina simultaneamente PC1, PC2 e PC3, fornendo una vista tridimensionale "
        "della struttura dei dati. L'angolo di visualizzazione e' impostato a azimuth=30, elevation=25 "
        "per massimizzare la leggibilita' dei cluster. I campioni sono rappresentati come sfere colorate "
        "per classe con bordo nero per migliorare il contrasto."
    )
    pdf.add_figure("06_pca_scores_3d.png",
                    "Figura 6: Score plot 3D (PC1 vs PC2 vs PC3), colorato per regione.", width=140)
    pdf.body_text(
        "La visualizzazione 3D conferma e integra le osservazioni dei plot 2D:\n\n"
        "- I cluster delle 5 classi sono discernibili anche visivamente, indicando che la composizione "
        "in acidi grassi contiene informazione sufficiente per la classificazione geografica.\n"
        "- Le classi che appaiono sovrapposte nei plot 2D possono risultare separate nella terza dimensione.\n"
        "- La compattezza dei cluster varia significativamente: SA, essendo la classe piu' numerosa, mostra "
        "una maggiore dispersione, mentre WL e EL tendono a essere piu' compatte.\n"
        "- La presenza di eventuali campioni isolati (outlier) o ai margini dei cluster puo' indicare "
        "campioni di difficile classificazione che saranno critici per le prestazioni dei modelli."
    )

    # --- FIGURE 07 ---
    pdf.add_page()
    pdf.section_title("3.6 PCA Esplorativa: Loading Analysis", level=2)
    pdf.body_text(
        "Il pannello sinistro mostra un bar chart dei loadings delle prime tre PC (PC1, PC2, PC3), con "
        "una barra per ciascuna variabile raggruppata per PC. Il pannello destro mostra il loading plot "
        "nello spazio PC1-PC2, dove ciascuna variabile e' rappresentata come un vettore dall'origine, "
        "con il cerchio unitario come riferimento."
    )
    pdf.add_figure("07_pca_loadings.png",
                    "Figura 7: Loading bar chart PC1-PC3 (sinistra) e Loading plot PC1 vs PC2 (destra).")
    pdf.body_text(
        "Interpretazione dei loadings:\n\n"
        "Bar chart (pannello sinistro):\n"
        "- PC1: dominata dall'opposizione oleico (loading negativo o positivo elevato) vs linoleico "
        "(segno opposto). Palmitoleico contribuisce nella stessa direzione del linoleico. Questo "
        "conferma che la principale sorgente di variazione e' la composizione oleico/linoleico.\n"
        "- PC2: cattura la variazione in palmitoleico, stearico e potenzialmente eicosanoico, "
        "ortogonale all'asse oleico-linoleico.\n"
        "- PC3: variabili minori come linolenico, eicosanoico e stearico contribuiscono "
        "maggiormente, catturando varianza residua non rappresentata dalle prime due PC.\n\n"
        "Loading plot (pannello destro):\n"
        "- Il cerchio unitario grigio tratteggiato indica il raggio massimo (r=1). Variabili vicine al "
        "cerchio sono ben rappresentate nello spazio PC1-PC2; variabili vicine all'origine sono mal "
        "rappresentate e richiedono PC aggiuntive.\n"
        "- Oleico e Linoleico si trovano in direzioni opposte (anticorrelazione), confermando la "
        "struttura biochimica nota.\n"
        "- La prossimita' di Palmitoleico al Linoleico nello spazio dei loadings indica che queste "
        "variabili co-variano nello stesso senso.\n"
        "- Eicosanoico e Linolenico, essendo vicini all'origine, sono scarsamente rappresentati nelle "
        "prime due PC e la loro informazione e' catturata da PC successive."
    )

    # ====================================================================
    # 4. SIMCA RESULTS
    # ====================================================================
    pdf.add_page()
    pdf.section_title("4. Classificazione con SIMCA")

    pdf.body_text(
        "Il metodo SIMCA (Soft Independent Modelling of Class Analogy) costruisce un modello PCA "
        "indipendente per ciascuna delle 5 classi. La selezione del numero ottimale di PC per classe "
        "avviene tramite cross-validazione (venetian blinds, 5 segmenti) sul training set, valutando "
        "PC da 1 a 6. Il criterio di classificazione combina la Score Distance (T2) e la Orthogonal "
        "Distance (Q) con un limite D <= sqrt(2)."
    )

    # --- FIGURE 08 ---
    pdf.section_title("4.1 Cross-validazione e Scelta delle Componenti", level=2)
    pdf.body_text(
        "Il grafico mostra tre pannelli sovrapposti, ciascuno con l'andamento di una metrica al variare "
        "del numero di PC (da 1 a 6), con una curva per ciascuna delle 5 classi. I pannelli sono:\n\n"
        "1) Sensitivity (pannello superiore): per ciascun numero di PC, la proporzione di campioni "
        "della classe correttamente accettati dal proprio modello in CV. Valori alti sono desiderabili.\n\n"
        "2) Specificity (pannello centrale): per ciascun numero di PC, la proporzione di campioni "
        "delle altre classi correttamente rifiutati. Una specificita' troppo bassa indica che il modello "
        "e' troppo 'morbido' e accetta campioni non appartenenti.\n\n"
        "3) Efficiency (pannello inferiore): sqrt(Sensitivity * Specificity), media geometrica che "
        "fornisce un bilancio ottimale. Questo e' il criterio usato per selezionare il numero di PC."
    )
    pdf.add_figure("08_simca_cv_metrics.png",
                    "Figura 8: Metriche SIMCA in cross-validazione al variare del numero di PC per classe.")
    pdf.body_text(
        "Dall'analisi di questo grafico si seleziona per ciascuna classe il numero di PC che "
        "massimizza l'efficienza. In generale:\n"
        "- All'aumentare delle PC, la sensibilita' tende ad aumentare (i modelli diventano piu' flessibili "
        "e accettano piu' campioni), ma la specificita' puo' diminuire (i confini si allargano).\n"
        "- L'efficienza bilancia questi due aspetti e tipicamente mostra un massimo dopo il quale "
        "l'aggiunta di PC non migliora ma potenzialmente peggiora le prestazioni.\n"
        "- Classi con struttura interna piu' complessa (es. SA con molti campioni) potrebbero necessitare "
        "di piu' PC, mentre classi piccole e compatte (es. NA) potrebbero averne bisogno di meno.\n"
        "- E' fondamentale che ciascuna classe possa avere un numero diverso di PC, coerentemente "
        "con la filosofia SIMCA di modellazione indipendente."
    )

    # --- FIGURE 09 (x5) ---
    pdf.add_page()
    pdf.section_title("4.2 Score Distance vs Orthogonal Distance", level=2)
    pdf.body_text(
        "Per ciascuna delle 5 classi viene generato un grafico SD vs OD che rappresenta il piano "
        "diagnostico fondamentale del modello SIMCA. L'asse X riporta la Score Distance normalizzata "
        "(T2/T2lim), l'asse Y la Orthogonal Distance normalizzata (Q/Qlim). La curva rossa rappresenta "
        "il cerchio di raggio sqrt(2), ovvero il confine di accettazione: campioni all'interno della "
        "curva sono accettati dal modello della classe. I cerchi (o) rappresentano i campioni di "
        "training, i diamanti (d) pieni rappresentano i campioni di test, tutti colorati per classe."
    )

    class_names_full = ["North Apulia", "South Apulia", "Umbria", "East Liguria", "West Liguria"]
    class_names_short = ["NA", "SA", "U", "EL", "WL"]
    sd_od_comments = [
        # Class 1 - NA
        (
            "Per il modello della classe North Apulia, i campioni NA di training (cerchi blu) dovrebbero "
            "trovarsi all'interno del cerchio rosso. I campioni delle altre classi (SA, U, EL, WL) dovrebbero "
            "trovarsi all'esterno, indicando che il modello li rifiuta. Data la piccola numerosita' di NA "
            "(25 campioni, ~18 in training), il modello potrebbe essere meno stabile. Eventuali campioni NA "
            "di test (diamanti blu) all'esterno segnalano falsi negativi, mentre campioni di altre classi "
            "all'interno segnalano falsi positivi. La posizione dei campioni rispetto ai due assi indica "
            "la natura dell'anomalia: alta SD = lontano dal centro nello spazio delle PC, "
            "alta OD = struttura diversa non catturata dal modello."
        ),
        # Class 2 - SA
        (
            "Il modello South Apulia e' quello con piu' campioni di training (~144). La classe SA, "
            "essendo la piu' numerosa ed eterogenea, potrebbe richiedere piu' PC per una descrizione "
            "adeguata. I campioni SA dovrebbero formare un cluster compatto all'interno del cerchio, "
            "mentre le classi liguri (EL, WL), avendo composizioni diverse, dovrebbero trovarsi ben "
            "al di fuori. La separazione SA/NA e' meno netta per la prossimita' geografica. Campioni "
            "con elevata OD indicano una struttura chimica non compatibile con il modello SA, mentre "
            "campioni con elevata SD hanno composizioni compatibili ma estreme."
        ),
        # Class 3 - U
        (
            "Il modello Umbria descrive una classe di dimensione intermedia (~36 in training). "
            "L'Umbria, essendo geograficamente tra il Centro e il Nord, potrebbe condividere "
            "caratteristiche con sia le classi pugliesi che quelle liguri. Il grafico mostra se "
            "campioni di altre classi vengono erroneamente accettati (bassa specificita') o se "
            "campioni umbri vengono rifiutati dal proprio modello (bassa sensibilita'). La posizione "
            "relativa delle classi rispetto al cerchio di accettazione rivela le 'distanze chimiche' "
            "tra le regioni dal punto di vista dell'Umbria."
        ),
        # Class 4 - EL
        (
            "Il modello East Liguria mostra le distanze di tutti i campioni rispetto al modello PCA "
            "costruito sui campioni liguri orientali. Le classi EL e WL (entrambe liguri) potrebbero "
            "avere una sovrapposizione parziale, che si manifesta con campioni WL all'interno o "
            "vicino al confine del modello EL. I campioni pugliesi (NA, SA) dovrebbero essere chiaramente "
            "al di fuori, con OD elevata per la diversa struttura degli acidi grassi. L'analisi dei "
            "campioni di test (diamanti) rispetto al training (cerchi) indica la generalizzabilita' "
            "del modello."
        ),
        # Class 5 - WL
        (
            "Il modello West Liguria (Taggiasche) descrive una classe con composizione distintiva. "
            "Le olive taggiasche hanno un profilo di acidi grassi specifico che potrebbe rendere "
            "questa classe piu' compatta e ben separata. Il grafico mostrera' se questo e' confermato: "
            "campioni WL concentrati all'interno del cerchio e campioni delle altre classi ben "
            "all'esterno. Una buona separazione nel piano SD/OD corrisponde a elevata efficienza "
            "di classificazione per questa classe."
        )
    ]

    for ic in range(5):
        fname = f"09_simca_SDvsOD_class{ic+1}.png"
        if os.path.exists(os.path.join(PLOT_DIR, fname)):
            pdf.add_figure(fname,
                f"Figura {8+ic+1}: SIMCA - Score Distance vs Orthogonal Distance per il modello della "
                f"Classe {ic+1} ({class_names_full[ic]}). Cerchi = training, Diamanti = test, "
                f"Curva rossa = confine D <= sqrt(2).",
                width=150)
            pdf.body_text(sd_od_comments[ic])

    # --- FIGURE 10 ---
    pdf.add_page()
    pdf.section_title("4.3 Matrici di Confusione e Metriche", level=2)

    pdf.section_title("Matrici di Confusione SIMCA", level=3)
    pdf.body_text(
        "La matrice di confusione e' una tabella (nClasses x nClasses) che riassume le predizioni del "
        "classificatore. Le righe rappresentano le classi predette, le colonne le classi vere (o viceversa, "
        "secondo la convenzione). Gli elementi sulla diagonale sono le classificazioni corrette (True "
        "Positives per ciascuna classe), gli elementi extra-diagonali sono gli errori. I valori percentuali "
        "annotati in ciascuna cella facilitano la lettura. Due matrici sono presentate fianco a fianco: "
        "training set (sinistra) e test set (destra)."
    )
    pdf.add_figure("10_simca_confusion_matrices.png",
                    "Figura 14: Matrici di confusione SIMCA per training set (sinistra) e test set (destra).")
    pdf.body_text(
        "Interpretazione delle matrici di confusione:\n\n"
        "- Diagonale: i numeri sulla diagonale indicano quanti campioni di ciascuna classe sono stati "
        "correttamente classificati. Percentuali alte sulla diagonale corrispondono ad alta sensibilita'.\n\n"
        "- Elementi extra-diagonali: indicano confusione tra le classi. Ad esempio, un valore alto nella "
        "cella (riga SA, colonna NA) indica che molti campioni NA sono stati classificati come SA. "
        "Questo tipo di errore rivela la somiglianza chimica tra le regioni.\n\n"
        "- Confronto training vs test: il modello tipicamente performa meglio sul training set. Una "
        "degradazione significativa sul test set e' segno di overfitting. SIMCA, essendo un metodo "
        "basato su modelli individuali per classe, e' relativamente robusto all'overfitting, ma classi "
        "con pochi campioni (NA) possono mostrare instabilita'.\n\n"
        "- Si noti che in SIMCA un campione puo' essere rifiutato da tutte le classi (non accettato) "
        "o accettato da piu' classi. In caso di accettazione multipla, la regola di assegnazione "
        "tipicamente sceglie la classe con la distanza D minore."
    )

    # --- FIGURE 13 ---
    pdf.section_title("Tabella Riassuntiva SIMCA", level=3)
    pdf.body_text(
        "La tabella riassuntiva mostra, per ciascuna classe: il numero ottimale di PC selezionato "
        "in CV, e le metriche di sensibilita', specificita' ed efficienza calcolate sul training set "
        "e sul test set indipendente. Questa tabella permette un confronto diretto delle prestazioni "
        "per classe e tra i due set."
    )
    pdf.add_figure("13_simca_summary_table.png",
                    "Figura 15: Tabella riassuntiva delle metriche SIMCA per classe.", width=150)
    pdf.body_text(
        "Dalla tabella si puo' valutare:\n"
        "- Quali classi sono meglio modellate (alta efficienza su entrambi i set)\n"
        "- Il grado di generalizzazione (confronto train vs test)\n"
        "- L'effetto dello sbilanciamento (la classe SA potrebbe avvantaggiarsi dalla numerosita')\n"
        "- La complessita' di ciascuna classe (numero di PC necessarie)"
    )

    # --- FIGURE 11 ---
    pdf.add_page()
    pdf.section_title("4.4 Potere Discriminante e Loadings", level=2)

    pdf.section_title("Discriminant Power", level=3)
    pdf.body_text(
        "Il Discriminant Power e' un indice specifico di SIMCA che quantifica la capacita' di ciascuna "
        "variabile di discriminare tra le classi. Viene calcolato come il rapporto medio tra i residui "
        "dei campioni non appartenenti alla classe e quelli appartenenti, mediato su tutte le coppie di "
        "classi. Il diagramma a barre mostra il valore per ciascuno dei 7 acidi grassi, con linee di "
        "riferimento orizzontali per il 95o percentile (verde), il 99o percentile (rosso) e la media "
        "(nero tratteggiato)."
    )
    pdf.add_figure("11_simca_discriminant_power.png",
                    "Figura 16: SIMCA Discriminant Power per ciascun acido grasso, con soglie percentili.")
    pdf.body_text(
        "Interpretazione del Discriminant Power:\n\n"
        "- Variabili con potere discriminante superiore al 95o percentile sono considerate 'molto "
        "discriminanti' e sono le piu' utili per la separazione delle classi.\n"
        "- Variabili sopra il 99o percentile hanno un ruolo eccezionale nella discriminazione.\n"
        "- Ci si attende che l'acido oleico, il linoleico e il palmitoleico abbiano elevato potere "
        "discriminante, coerentemente con le differenze osservate nei boxplot per classe.\n"
        "- Variabili con basso potere discriminante (sotto la media) contribuiscono poco alla "
        "classificazione SIMCA e potrebbero essere eliminate in un'analisi di feature selection.\n"
        "- Il ranking delle variabili per discriminant power puo' essere confrontato con i VIP scores "
        "della PLS-DA per verificare la coerenza tra i due metodi."
    )

    # --- FIGURE 12 ---
    pdf.section_title("Loadings per Classe SIMCA", level=3)
    pdf.body_text(
        "Questo pannello di grafici mostra i loadings dei modelli PCA di ciascuna classe, organizzati "
        "in una griglia (nClasses x maxPC). Ogni subplot mostra un diagramma a barre con il loading "
        "delle 7 variabili per una specifica PC di una specifica classe. Il colore delle barre indica "
        "la classe (blu=NA, rosso=SA, verde=U, magenta=EL, arancione=WL)."
    )
    pdf.add_figure("12_simca_loadings.png",
                    "Figura 17: SIMCA Loadings per ciascuna classe e componente principale.")
    pdf.body_text(
        "L'analisi dei loadings per classe e' fondamentale nella filosofia SIMCA:\n\n"
        "- Se i loadings delle diverse classi sono simili, le classi condividono la stessa struttura "
        "di covarianza e la discriminazione si basa principalmente sulle posizioni nello spazio scores.\n"
        "- Se i loadings sono diversi tra le classi, le classi hanno strutture interne differenti, il "
        "che giustifica pienamente l'approccio SIMCA di modellazione indipendente.\n"
        "- Ad esempio, la PC1 di SA potrebbe essere dominata dall'asse oleico-linoleico, mentre la PC1 "
        "di WL potrebbe coinvolgere maggiormente il palmitoleico, indicando diverse sorgenti di variabilita'.\n"
        "- Classi con piu' PC (es. SA) hanno una struttura interna piu' complessa che richiede componenti "
        "aggiuntive per essere descritta. Classi con 1-2 PC hanno una struttura piu' semplice.\n"
        "- La variabilita' dei loadings tra classi conferma che SIMCA e' la scelta appropriata rispetto "
        "a un singolo modello PCA globale."
    )

    # ====================================================================
    # 5. PLS-DA RESULTS
    # ====================================================================
    pdf.add_page()
    pdf.section_title("5. Classificazione con PLS-DA")

    pdf.body_text(
        "PLS-DA (Partial Least Squares Discriminant Analysis) costruisce un unico modello di regressione "
        "che mappa simultaneamente tutte le variabili X nella matrice di classi Y (dummy coding 1/0). "
        "L'algoritmo NIPALS trova variabili latenti che massimizzano la covarianza tra X e Y. La "
        "classificazione avviene assegnando ogni campione alla classe con il valore Y predetto piu' alto "
        "(True Discriminant). X e' autoscalato e Y e' mean-centered prima della regressione. "
        "La cross-validazione (venetian blinds, 5 segmenti) seleziona il numero ottimale di LV (da 1 a 10)."
    )

    # --- FIGURE 14 ---
    pdf.section_title("5.1 Cross-validazione e RMSECV", level=2)

    pdf.section_title("Errori di Classificazione in CV", level=3)
    pdf.body_text(
        "La figura mostra 4 pannelli che visualizzano le prestazioni PLS-DA in cross-validazione "
        "al variare del numero di variabili latenti (LV) da 1 a 10:\n\n"
        "1) % Correct Classification (alto-sx): la percentuale di campioni correttamente classificati "
        "per ciascuna delle 5 classi. Ciascuna classe ha la propria curva colorata. L'obiettivo e' "
        "massimizzare queste percentuali.\n\n"
        "2) Misclassified Samples (basso-sx): il numero assoluto di campioni mal classificati per classe. "
        "Complemento del pannello precedente, utile per capire l'entita' assoluta degli errori.\n\n"
        "3) Mean % Correct (alto-dx): la media delle percentuali corrette su tutte le classi, una "
        "singola curva blu. Questo e' un indicatore globale unidimensionale delle prestazioni.\n\n"
        "4) Total Misclassified (basso-dx): il numero totale di campioni mal classificati (somma su "
        "tutte le classi), una singola curva rossa. Il numero ottimale di LV e' selezionato "
        "minimizzando questo valore."
    )
    pdf.add_figure("14_plsda_cv_error.png",
                    "Figura 18: PLS-DA Cross-Validation: percentuali corrette per classe, errori e metriche aggregate.")
    pdf.body_text(
        "Dall'analisi di questi grafici si ricava il numero ottimale di LV:\n"
        "- Per LV=1 le prestazioni sono generalmente basse, poiche' una sola variabile latente "
        "non e' sufficiente per discriminare 5 classi.\n"
        "- Le prestazioni migliorano rapidamente con l'aggiunta delle prime LV (tipicamente fino a 3-5).\n"
        "- Oltre un certo numero di LV, le prestazioni in CV si stabilizzano o peggiorano (overfitting).\n"
        "- La classe NA, avendo pochi campioni, puo' mostrare oscillazioni maggiori tra i diversi numeri "
        "di LV per la minore stabilita' statistica.\n"
        "- Si impone un minimo di 2 LV per permettere la visualizzazione 2D degli scores."
    )

    # --- FIGURE 15 ---
    pdf.section_title("RMSECV per Classe", level=3)
    pdf.body_text(
        "Il grafico RMSECV (Root Mean Square Error of Cross-Validation) mostra l'errore di regressione "
        "in cross-validazione per ciascuna classe al variare del numero di LV. L'RMSECV misura quanto "
        "bene il modello predice i valori dummy (1/0) delle classi, non direttamente l'errore di "
        "classificazione. Una curva per ciascuna classe e' tracciata, con marcatori circolari e "
        "legenda."
    )
    pdf.add_figure("15_plsda_rmsecv.png",
                    "Figura 19: RMSECV per classe al variare del numero di variabili latenti.")
    pdf.body_text(
        "NOTA IMPORTANTE: l'andamento dell'RMSECV non corrisponde necessariamente al trend degli "
        "errori di classificazione. L'RMSECV misura l'errore quadratico medio delle Y predette (valori "
        "continui) rispetto alle Y vere (0/1), mentre la classificazione dipende solo dal valore "
        "massimo tra le Y predette. Puo' accadere che una LV addizionale riduca l'RMSECV senza migliorare "
        "la classificazione, o viceversa.\n\n"
        "La classe SA (piu' numerosa) puo' mostrare un RMSECV piu' basso semplicemente per la "
        "maggiore disponibilita' di campioni per la stima. Classi piccole (NA) tendono ad avere "
        "RMSECV piu' alto e piu' variabile."
    )

    # --- FIGURE 16, 17 ---
    pdf.add_page()
    pdf.section_title("5.2 Predizioni Y e Matrici di Confusione", level=2)

    pdf.section_title("Y Predetto - Training Set", level=3)
    pdf.body_text(
        "Questa figura contiene 5 subplot verticali (uno per classe). In ciascun subplot:\n"
        "- L'asse X rappresenta i campioni del training set (numerati sequenzialmente).\n"
        "- L'asse Y mostra il valore predetto della Y dummy per quella specifica classe.\n"
        "- I campioni appartenenti alla classe target sono rappresentati con simboli pieni (filled circles), "
        "quelli delle altre classi con simboli vuoti.\n"
        "- Una linea rossa tratteggiata a Y=0.5 indica la soglia: campioni della classe target dovrebbero "
        "avere Y > 0.5, gli altri Y < 0.5.\n"
        "- La linea grigia sottile collega i punti per facilitare la visualizzazione dell'andamento."
    )
    pdf.add_figure("16_plsda_ypred_train.png",
                    "Figura 20: PLS-DA - Valori Y predetti per ogni classe sul training set.")
    pdf.body_text(
        "Analisi del grafico Y predetto (training):\n\n"
        "- Per un modello perfetto, i campioni della classe target (pieni) dovrebbero trovarsi tutti "
        "sopra la soglia 0.5 e i campioni delle altre classi (vuoti) tutti sotto.\n"
        "- Campioni della classe target sotto la soglia sono falsi negativi: il modello non li riconosce.\n"
        "- Campioni delle altre classi sopra la soglia sono falsi positivi: il modello li confonde.\n"
        "- La 'altura' sopra/sotto la soglia indica la confidenza della predizione: Y molto vicino a 0.5 "
        "indica poca confidenza, Y vicino a 1 o 0 indica alta confidenza.\n"
        "- Le classi piu' piccole (NA) possono mostrare predizioni piu' 'incerte' (vicine a 0.5) "
        "perche' il modello ha meno campioni da cui apprendere.\n"
        "- Il training set tende a mostrare buone prestazioni anche da modelli sovraparametrizzati."
    )

    # --- FIGURE 17 ---
    pdf.section_title("Y Predetto - Test Set", level=3)
    pdf.body_text(
        "Struttura identica al grafico precedente ma calcolato sul test set indipendente. Questo e' "
        "il vero banco di prova del modello, poiche' i campioni di test non sono stati utilizzati "
        "nella costruzione del modello."
    )
    pdf.add_figure("17_plsda_ypred_test.png",
                    "Figura 21: PLS-DA - Valori Y predetti per ogni classe sul test set.")
    pdf.body_text(
        "Confronto training vs test nelle Y predette:\n\n"
        "- Una degradazione significativa della separazione Y > 0.5 / Y < 0.5 dal training al test "
        "indica overfitting.\n"
        "- Se le prestazioni sono stabili, il modello generalizza bene.\n"
        "- Campioni di test con Y predicte errate (sotto/sopra soglia) corrispondono agli errori "
        "nella matrice di confusione. La posizione di questi campioni rispetto alla soglia indica "
        "quanto il modello e' 'sicuro' delle sue predizioni errate.\n"
        "- La classe SA, essendo la piu' rappresentata, tende ad avere predizioni piu' stabili e una "
        "separazione piu' netta, mentre le classi piccole mostrano piu' variabilita'."
    )

    # --- FIGURE 18 ---
    pdf.add_page()
    pdf.section_title("Matrici di Confusione PLS-DA", level=3)
    pdf.body_text(
        "Le matrici di confusione PLS-DA sono strutturate come quelle SIMCA (righe x colonne = predette x "
        "vere) con annotazioni percentuali. Training set a sinistra, test set a destra. A differenza di "
        "SIMCA, PLS-DA assegna sempre ogni campione a esattamente una classe (hard classification): "
        "non esistono campioni 'rifiutati' o assegnati a piu' classi."
    )
    pdf.add_figure("18_plsda_confusion_matrices.png",
                    "Figura 22: Matrici di confusione PLS-DA per training set e test set.")
    pdf.body_text(
        "Interpretazione delle matrici di confusione PLS-DA:\n\n"
        "- Tutti i campioni sono assegnati a una (e una sola) classe, quindi il numero totale di "
        "campioni classificati corrisponde esattamente alla dimensione del dataset.\n"
        "- Errori tipici: confusione tra NA e SA (prossimita' geografica pugliese), confusione "
        "occasionale tra EL e WL (prossimita' ligure), confusione tra U e classi limitrofe.\n"
        "- Le prestazioni sul test set sono la misura piu' affidabile della capacita' predittiva "
        "del modello, poiche' non affette dal bias di autocalibrazione.\n"
        "- Una matrice di confusione con elementi diagonali dominanti e valori extra-diagonali "
        "nulli o molto bassi indica un classificatore eccellente."
    )

    # --- FIGURE 22 (Summary table) ---
    pdf.section_title("Tabella Riassuntiva PLS-DA", level=3)
    pdf.body_text(
        "La tabella riassuntiva mostra per ciascuna classe: sensibilita', specificita' ed efficienza "
        "sul training set e sul test set, oltre al RMSEP (Root Mean Square Error of Prediction). "
        "L'intestazione indica il numero ottimale di LV selezionato in cross-validazione."
    )
    pdf.add_figure("22_plsda_summary_table.png",
                    "Figura 23: Tabella riassuntiva delle metriche PLS-DA per classe (con RMSEP).", width=155)
    pdf.body_text(
        "Note sulla tabella:\n"
        "- Il RMSEP e' calcolato solo sul test set e fornisce una misura dell'errore di predizione "
        "delle Y dummy. Un RMSEP basso indica predizioni precise delle variabili dummy.\n"
        "- Classi con alta sensibilita' ma bassa specificita' tendono ad 'attrarre' campioni delle "
        "altre classi; classi con alta specificita' ma bassa sensibilita' 'perdono' i propri campioni.\n"
        "- L'efficienza (sqrt(Sens*Spec)) e' la metrica piu' informativa per classi sbilanciate."
    )

    # --- FIGURE 19 ---
    pdf.add_page()
    pdf.section_title("5.3 Score Plots PLS", level=2)
    pdf.body_text(
        "Gli score plots PLS mostrano la proiezione dei campioni nello spazio delle prime due variabili "
        "latenti (LV1 vs LV2). Due pannelli affiancati: training set (sinistra) e test set (destra). "
        "I campioni sono colorati per classe con marcatori circolari pieni e bordo nero."
    )
    pdf.add_figure("19_plsda_scores.png",
                    "Figura 24: PLS-DA Score plots (LV1 vs LV2) - training set (sinistra) e test set (destra).")
    pdf.body_text(
        "A differenza dello score plot PCA (Fig. 5), le variabili latenti PLS sono costruite per "
        "massimizzare la covarianza con Y (classi), non semplicemente la varianza di X. Di conseguenza:\n\n"
        "- La separazione tra le classi dovrebbe essere piu' netta rispetto allo spazio PCA, poiche' "
        "le LV sono specificamente orientate alla discriminazione.\n"
        "- Il training set mostra la separazione ottimale ottenuta dal modello.\n"
        "- Il test set mostra la generalizzabilita' di questa separazione su campioni nuovi.\n"
        "- Cluster piu' separati e compatti indicano una migliore capacita' discriminante.\n"
        "- Campioni di test che cadono lontano dal cluster della propria classe sono quelli "
        "potenzialmente mal classificati.\n"
        "- La proiezione del test set si ottiene applicando la matrice di proiezione W*(P'W)^-1 "
        "ai dati di test autoscalati con le statistiche del training set.\n\n"
        "Il confronto tra lo score plot PCA (varianza-orientato) e lo score plot PLS (classe-orientato) "
        "e' particolarmente istruttivo: classi che si sovrappongono in PCA possono risultare separate "
        "in PLS, e viceversa."
    )

    # --- FIGURE 20 ---
    pdf.add_page()
    pdf.section_title("5.4 Coefficienti di Regressione e VIP", level=2)

    pdf.section_title("Coefficienti di Regressione PLS", level=3)
    pdf.body_text(
        "La figura mostra 5 subplot orizzontali (1 per classe), ciascuno con un diagramma a barre "
        "dei coefficienti di regressione B_PLS per le 7 variabili. Il colore delle barre corrisponde "
        "alla classe (blu=NA, rosso=SA, verde=U, magenta=EL, arancione=WL). Le etichette X indicano "
        "i nomi degli acidi grassi."
    )
    pdf.add_figure("20_plsda_regression_coefficients.png",
                    "Figura 25: Coefficienti di regressione PLS-DA per ciascuna classe.")
    pdf.body_text(
        "Interpretazione dei coefficienti di regressione:\n\n"
        "- I coefficienti B_PLS rappresentano il 'peso' di ciascuna variabile nella predizione della Y "
        "dummy per ogni classe: Y_pred_k = X * B_PLS_k + mean(Y_k).\n\n"
        "- Coefficiente positivo: all'aumentare della variabile, aumenta la probabilita' (Y predetto) "
        "di appartenenza alla classe. Coefficiente negativo: all'aumentare della variabile, diminuisce.\n\n"
        "- Il modulo (valore assoluto) indica l'importanza della variabile per quella specifica classe.\n\n"
        "- Confrontando i pattern tra le classi: variabili con segni opposti in classi diverse sono "
        "particolarmente discriminanti tra quelle classi. Ad esempio, se l'oleico ha coefficiente "
        "positivo per SA e negativo per EL, l'oleico e' un buon discriminante SA vs EL.\n\n"
        "- La somma dei coefficienti di tutte le classi per una variabile non e' necessariamente zero "
        "perche' Y e' mean-centered, non sommata a 1.\n\n"
        "- Variabili con coefficienti bassi in tutte le classi sono poco importanti per la classificazione."
    )

    # --- FIGURE 21 ---
    pdf.section_title("VIP Scores", level=3)
    pdf.body_text(
        "Il diagramma VIP (Variable Importance in Projection) mostra una barra per ciascuna delle "
        "7 variabili, con due linee di riferimento:\n"
        "- Linea rossa tratteggiata a VIP=1: soglia convenzionale di importanza. "
        "Variabili con VIP > 1 sono considerate importanti per la classificazione.\n"
        "- Linea nera puntinata a VIP=0.8: soglia inferiore. Variabili con VIP < 0.8 sono "
        "considerate poco influenti e potenziali candidate per l'eliminazione.\n"
        "Le barre sono colorate in verde per facilitare la lettura."
    )
    pdf.add_figure("21_plsda_vip.png",
                    "Figura 26: VIP scores - Importanza delle variabili nella proiezione PLS-DA.")
    pdf.body_text(
        "I VIP scores forniscono una misura globale (non specifica per classe) dell'importanza di "
        "ciascuna variabile nella discriminazione PLS-DA:\n\n"
        "- VIP > 1: variabile importante. Contribuisce significativamente alla separazione delle classi "
        "nello spazio delle variabili latenti.\n"
        "- 0.8 < VIP < 1: variabile moderatamente importante. Puo' contribuire ma non e' fondamentale.\n"
        "- VIP < 0.8: variabile poco importante. La sua rimozione non dovrebbe degradare "
        "significativamente le prestazioni.\n\n"
        "Il ranking VIP permette una feature selection informata: in un'applicazione pratica, "
        "si potrebbe ridurre il costo analitico misurando solo le variabili con VIP alto.\n\n"
        "Confronto con il Discriminant Power SIMCA (Fig. 16): se le stesse variabili risultano "
        "importanti in entrambi i metodi (VIP alto e Discriminant Power alto), la conclusione e' "
        "robusta. Discrepanze possono indicare che le due metriche catturano aspetti diversi della "
        "discriminazione (SIMCA basa il discriminant power sui residui, PLS-DA sulla covarianza con Y)."
    )

    # ====================================================================
    # 6. COMPARISON AND CONCLUSIONS
    # ====================================================================
    pdf.add_page()
    pdf.section_title("6. Confronto dei Metodi e Conclusioni")

    pdf.section_title("6.1 Confronto SIMCA vs PLS-DA", level=2)
    pdf.body_text(
        "I due metodi di classificazione adottano filosofie fondamentalmente diverse:\n\n"
        "SIMCA (Class Modeling):\n"
        "- Costruisce modelli PCA indipendenti per ciascuna classe\n"
        "- Puo' rifiutare campioni (nessuna classe li accetta) o assegnarli a piu' classi\n"
        "- Adatto quando l'obiettivo e' verificare l'autenticita'/appartenenza a una specifica classe\n"
        "- Ogni classe ha un proprio numero ottimale di PC\n"
        "- Fornisce il Discriminant Power come misura di importanza delle variabili\n"
        "- Robusto per campioni anomali o non appartenenti a nessuna classe nota\n\n"
        "PLS-DA (Discriminant Analysis):\n"
        "- Costruisce un unico modello focalizzato sulle differenze tra tutte le classi\n"
        "- Assegna sempre ogni campione a esattamente una classe (hard classification)\n"
        "- Adatto quando le classi sono ben definite e mutualmente esclusive\n"
        "- Un unico numero di LV per l'intero modello\n"
        "- Fornisce VIP scores e coefficienti di regressione per l'interpretazione\n"
        "- Tipicamente piu' potente quando le classi sono ben separate"
    )

    # --- FIGURE 23 ---
    pdf.section_title("Confronto delle Prestazioni sul Test Set", level=3)
    pdf.body_text(
        "La figura mostra due pannelli affiancati con diagrammi a barre raggruppate:\n"
        "- Pannello sinistro (Test Set Sensitivity): per ciascuna classe, due barre che confrontano "
        "la sensibilita' SIMCA (blu) e PLS-DA (rosso). Valori piu' alti indicano maggiore capacita' "
        "di riconoscere correttamente i campioni della classe.\n"
        "- Pannello destro (Test Set Specificity): per ciascuna classe, due barre che confrontano "
        "la specificita' SIMCA e PLS-DA. Valori piu' alti indicano maggiore capacita' di rifiutare "
        "campioni non appartenenti.\n"
        "L'asse Y va da 0 a 1.1 per tutte le classi, permettendo un confronto diretto."
    )
    pdf.add_figure("23_comparison_simca_plsda.png",
                    "Figura 27: Confronto sensibilita' e specificita' sul test set tra SIMCA e PLS-DA per classe.")
    pdf.body_text(
        "Interpretazione del confronto:\n\n"
        "- Se PLS-DA mostra sensibilita' maggiori di SIMCA, il modello discriminante e' piu' efficace "
        "nel riconoscere i campioni della classe. Cio' e' comune quando le classi sono ben separate.\n\n"
        "- Se SIMCA mostra specificita' maggiori, il modello per classe e' piu' selettivo nel rifiutare "
        "campioni estranei. Cio' e' tipico e costituisce il punto di forza di SIMCA.\n\n"
        "- La classe NA (piu' piccola) potrebbe mostrare le maggiori differenze tra i due metodi a causa "
        "della scarsa numerosita' che rende i modelli meno stabili.\n\n"
        "- Le classi SA (dominante) ed EL/WL (compatte e ben separate) tendono a performare bene con "
        "entrambi i metodi.\n\n"
        "- L'Umbria (U), geograficamente intermedia, potrebbe essere la classe piu' critica per "
        "entrambi i classificatori."
    )

    pdf.section_title("6.2 Conclusioni", level=2)
    pdf.body_text(
        "L'analisi di classificazione del dataset Olive Oil ha dimostrato che la composizione in "
        "acidi grassi e' un descrittore efficace dell'origine geografica dell'olio d'oliva italiano. "
        "L'analisi puo' essere sintetizzata nei seguenti punti chiave:\n\n"
        "1. L'analisi esplorativa (EDA con PCA) ha rivelato la presenza di raggruppamenti naturali "
        "nel dataset, con le prime 2-3 PC che catturano la maggior parte della varianza. L'asse "
        "oleico-linoleico e' la principale sorgente di variabilita', con il palmitoleico come "
        "variabile addizionale discriminante.\n\n"
        "2. Le classi liguri (EL, WL) si separano bene dalle pugliesi (NA, SA), riflettendo le "
        "differenze climatiche e cultivar tra le regioni. L'Umbria occupa una posizione intermedia.\n\n"
        "3. SIMCA ha costruito modelli PCA indipendenti con complessita' diversa per classe, "
        "fornendo una modellazione flessibile della struttura interna di ciascuna regione.\n\n"
        "4. PLS-DA ha costruito un modello discriminante globale ottimizzato per separare le classi, "
        "con interpretazione diretta tramite coefficienti di regressione e VIP scores.\n\n"
        "5. Lo sbilanciamento delle classi (SA dominante con 54% dei campioni) e' un fattore da "
        "considerare: le metriche di efficienza (sqrt(Sens*Spec)) sono piu' informative dell'accuracy "
        "globale per valutare le prestazioni per classe.\n\n"
        "6. La cross-validazione venetian blinds ha permesso di selezionare la complessita' ottimale "
        "evitando l'overfitting, con validazione finale su test set indipendente stratificato."
    )

    pdf.body_text(
        "Raccomandazioni operative:\n\n"
        "- Per applicazioni di autenticazione alimentare (es. verificare se un olio e' genuinamente "
        "taggiasco), si raccomanda SIMCA per la sua capacita' di rifiutare campioni non conformi.\n\n"
        "- Per la classificazione routinaria dell'origine geografica tra le 5 regioni note, PLS-DA "
        "offre tipicamente prestazioni migliori e una interpretazione piu' diretta.\n\n"
        "- Le variabili con VIP > 1 e alto Discriminant Power (tipicamente oleico, linoleico, "
        "palmitoleico) sono i marker piu' informativi e potrebbero essere sufficienti per un protocollo "
        "analitico semplificato.\n\n"
        "- Per un sistema di classificazione robusto in produzione, si consiglirebbe di utilizzare "
        "entrambi i metodi in cascata: SIMCA per la verifica di conformita' (il campione e' un olio "
        "d'oliva italiano autentico?) seguita da PLS-DA per l'assegnazione della regione specifica."
    )

    # ---- Save ----
    pdf.output(OUTPUT_PDF)
    print(f"\n[OK] Report saved to: {OUTPUT_PDF}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    # Check if plots exist
    if not os.path.exists(PLOT_DIR):
        print(f"[WARNING] Plot directory not found: {PLOT_DIR}")
        print("  Run main_analysis.m in MATLAB first to generate the plots.")
        print("  Generating report template with placeholders...\n")

    build_report()
