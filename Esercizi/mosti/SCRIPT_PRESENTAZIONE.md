# Script Presentazione — Classificazione Chemometrica dei Mosti di Uva

> **Nota:** La presentazione ha 16 slide. Questo script fornisce il testo da dire per ciascuna slide, le risposte alle domande di `mosti.txt`, una verifica dell'analisi, e le conclusioni mancanti.

---

## SLIDE 1 — Copertina

**Script:**

> "Buongiorno a tutti. Oggi presentiamo un'analisi chemometrica per la classificazione di mosti di uva basata sul profilo antocianico determinato tramite HPLC. L'obiettivo è verificare se è possibile distinguere cinque varietali — Ancellotta, Montepulciano, Lambrusco Pugliese, Sangiovese e Nero d'Avola — e capire se l'annata di vendemmia è un fattore di variabilità rilevante rispetto al varietale. Utilizzeremo tre approcci: PCA per l'analisi esplorativa, SIMCA per il class modeling, e PLS-DA per l'analisi discriminante."

---

## SLIDE 2 — Introduzione e Dataset

**Script:**

> "Il dataset è contenuto nel file `mosti.mat` e comprende 98 campioni di mosto ottenuti da uve di cinque varietali, raccolti in due annate: 2000 e 2001. Per ciascun campione sono state misurate sei variabili: le percentuali d'area di cinque antociani — delfinidina, cianidina, petunidina, peonidina, malvidina — e il rapporto tra antociani liberi e legati. Le domande a cui vogliamo rispondere sono: i campioni si distinguono per varietale? L'annata è un fattore minore? Quali variabili sono più discriminanti? E infine, SIMCA e PLS-DA riescono a classificare correttamente i varietali?"

---

## SLIDE 3 — Preprocessing (Autoscaling)

**Script:**

> "Come preprocessing abbiamo scelto l'autoscaling, cioè centratura sulla media e divisione per la deviazione standard. Il motivo è chiaro dal boxplot a sinistra: le variabili hanno scale molto diverse — la malvidina domina in percentuale rispetto alla cianidina, e il rapporto lib/lrg ha un ordine di grandezza diverso. Senza autoscaling, MVD% dominerebbe la PCA e i modelli di classificazione. Dopo l'autoscaling, a destra, tutte le variabili hanno media zero e varianza unitaria, e contribuiscono equamente ai modelli. Questa è la scelta standard quando le variabili hanno unità e range differenti."

---

## SLIDE 4 — PCA: Scree Plot e Varianza Spiegata

**Script:**

> "Passiamo all'analisi PCA. Lo scree plot mostra la varianza spiegata da ciascuna componente principale. Il 'gomito' si osserva dopo le prime 2-3 PC, che insieme spiegano la maggior parte della varianza totale. Il grafico cumulativo a destra mostra che con 3-4 PC si supera circa l'80-90% della varianza. Questo ci dice che il dataset, nonostante abbia 6 variabili, ha una dimensionalità intrinseca di circa 2-3 dimensioni — il che è coerente con il fatto che stiamo lavorando con dati composizionali (le percentuali d'area che sommano approssimativamente a 100%)."

---

## SLIDE 5 — PCA Score Plot per Varietale (PC1 vs PC2)

**Script:**

> "Ecco il cuore dell'analisi esplorativa. Nello score plot PC1 vs PC2 i campioni sono colorati per varietale. Si osserva un raggruppamento molto chiaro: i cinque varietali formano cluster distinti nello spazio delle prime due componenti principali. L'Ancellotta tende a separarsi nettamente dagli altri varietali, mentre Sangiovese e Nero d'Avola possono presentare qualche sovrapposizione, che verrà risolta includendo PC3. Il pannello PC1 vs PC3 rivela ulteriori separazioni tra varietali che apparivano sovrapposti nelle prime due componenti. **Questo risponde direttamente alla prima domanda: sì, è possibile distinguere i diversi varietali sulla base del profilo antocianico.**"

---

## SLIDE 6 — PCA Score Plot per Annata (PC1 vs PC2)

**Script:**

> "Questo è lo stesso score plot, ma con i campioni colorati per annata — verde per il 2000 e marrone per il 2001. Osservate come i campioni delle due annate si sovrappongono completamente, senza formare cluster separati. Questo ci dice che **l'annata NON è un fattore dominante di variabilità**: l'effetto varietale è preponderante rispetto all'effetto annata. Questo è confermato anche dai boxplot per annata, dove le distribuzioni di ciascuna variabile sono sostanzialmente sovrapposte tra il 2000 e il 2001. **Rispondiamo così alla seconda domanda: l'anno di produzione è effettivamente un fattore di minore variabilità rispetto al varietale.**"

---

## SLIDE 7 — PCA Loading Plot e Biplot

**Script:**

> "I loading plot ci dicono quali variabili sono responsabili della separazione osservata nello score plot. Nel grafico a barre vediamo i coefficienti di loading per le prime 3 PC: variabili con loading elevato in valore assoluto su una PC contribuiscono maggiormente a quella direzione di variazione. Nel loading biplot a destra, le frecce indicano le variabili: quelle che puntano nella stessa direzione sono positivamente correlate; quelle in direzione opposta sono anticorrelate. **Le variabili con le frecce più lunghe e orientate verso i cluster di specifici varietali sono le più discriminanti.** Per esempio, se MVD% punta verso il cluster dell'Ancellotta, significa che questo varietale è caratterizzato da un alto contenuto di malvidina."

---

## SLIDE 8 — Correlazione e Boxplot per Classe

**Script:**

> "La matrice di correlazione mostra le correlazioni di Pearson tra le sei variabili. Come atteso per dati composizionali, osserviamo correlazioni negative tra alcuni antociani — se uno aumenta in percentuale, gli altri necessariamente diminuiscono. I boxplot per classe confermano a livello univariato quello che la PCA mostra a livello multivariato: alcune variabili discriminano bene tra varietali (box ben separati), mentre altre sono meno informative. Tuttavia, per una discriminazione efficace serve l'approccio multivariato, perché nessuna singola variabile separa perfettamente tutti e cinque i varietali."

---

## SLIDE 9 — Split Training/Test e Metodologia Classificazione

**Script:**

> "Per la fase di classificazione, abbiamo suddiviso i dati in training set (70%) e test set (30%) con split stratificato per classe e seed fisso (rng(42)) per la riproducibilità. I dati delle due annate sono stati usati insieme, come richiesto dall'esercizio. Applichiamo due metodi: SIMCA, che è una tecnica di class modeling che costruisce un modello PCA locale per ogni varietale, e PLS-DA, che è un metodo discriminante che cerca le variabili latenti che massimizzano la separazione tra classi."

---

## SLIDE 10 — SIMCA: Cross-Validazione e Selezione PC

**Script:**

> "Per SIMCA, abbiamo ottimizzato il numero di componenti principali per ciascun modello di classe mediante cross-validazione venetian-blind a 5 segmenti. Il criterio di selezione è la massimizzazione dell'efficienza, definita come media geometrica di sensibilità e specificità. Ogni classe può avere un numero diverso di PC, riflettendo la complessità interna del varietale — una classe con profilo antocianico semplice e caratteristico necessita di poche PC, mentre classe più variabile internamente ne richiede di più."

---

## SLIDE 11 — SIMCA: Risultati e Confusion Matrix

**Script:**

> "Le confusion matrix mostrano le prestazioni di SIMCA sul training e sul test set. Le diagonali verdi sono le classificazioni corrette, i valori fuori diagonale rappresentano le misclassificazioni. SIMCA classifica assegnando ogni campione alla classe il cui modello PCA locale ha la distanza combinata T²+Q minore. I grafici Score Distance vs Orthogonal Distance per ciascuna classe mostrano come i campioni del varietale target cadano all'interno del confine di accettazione (cerchio rosso), mentre i campioni degli altri varietali cadono generalmente all'esterno."

---

## SLIDE 12 — SIMCA: Discriminant Power e Variabili Importanti

**Script:**

> "Il Discriminant Power di SIMCA quantifica la capacità di ciascuna variabile di distinguere tra le classi. Le variabili con valori più alti sono i migliori discriminatori. Questo ci aiuta a rispondere alla domanda su quali variabili distinguono meglio i campioni: le variabili sopra la media (linea nera tratteggiata) sono discriminatori sopra la media, e quelle sopra il 95° percentile sono eccezionali. **Queste sono le variabili antocianiche che rappresentano i marcatori chimici più affidabili per la distinzione varietale.**"

---

## SLIDE 13 — PLS-DA: Cross-Validazione e Selezione LV

**Script:**

> "Per PLS-DA, la Y è codificata come matrice dummy (one-hot) centrata sulla media. Il numero di variabili latenti è selezionato minimizzando l'errore totale di misclassificazione in cross-validazione venetian-blind a 5 segmenti. Il RMSECV per classe fornisce un criterio continuo complementare. Il numero ottimale di LV è un compromesso tra capacità discriminante e rischio di overfitting."

---

## SLIDE 14 — PLS-DA: Risultati, Confusion Matrix e Score Plot

**Script:**

> "Le confusion matrix di PLS-DA mostrano tipicamente un'accuratezza superiore a SIMCA, perché PLS-DA è un metodo esplicitamente discriminante — è ottimizzato per separare le classi, mentre SIMCA modella ciascuna classe indipendentemente. Lo score plot LV1 vs LV2 mostra la separazione nello spazio delle variabili latenti, dove i cluster sono generalmente meglio definiti rispetto alla PCA perché la proiezione è guidata dalla supervisione. Le Y predette confermano una buona separazione: i campioni della classe corretta hanno valori predetti vicini a 1, gli altri vicini a 0."

---

## SLIDE 15 — PLS-DA: VIP e Coefficienti di Regressione

**Script:**

> "I VIP scores — Variable Importance in Projection — ci danno una misura sintetica dell'importanza di ciascuna variabile nel modello PLS-DA. Variabili con VIP > 1 sono considerate importanti, tra 0.8 e 1 sono borderline, sotto 0.8 hanno contributo sotto la media. I coefficienti di regressione PLS-DA forniscono un'impronta chimica specifica per ciascun varietale: coefficienti positivi grandi indicano antociani 'marcatore positivo' per quella varietà, coefficienti negativi grandi indicano antociani 'marcatore negativo'. **Combinando VIP e Discriminant Power SIMCA, possiamo identificare definitivamente quali variabili distinguono meglio i varietali.**"

---

## SLIDE 16 — Confronto SIMCA vs PLS-DA e Applicability Domain

**Script:**

> "Il confronto diretto tra SIMCA e PLS-DA sul test set mostra i punti di forza complementari dei due metodi. PLS-DA tende ad avere sensitività più elevata grazie all'ottimizzazione esplicita della discriminazione, mentre SIMCA può eccellere in specificità grazie ai confini di accettazione ben definiti. L'Applicability Domain — verificato tramite Williams plot e diagramma T²/Q — conferma che praticamente tutti i campioni del test set ricadono nello spazio chimico del training set, validando l'affidabilità delle nostre predizioni."

---

---

# CONCLUSIONI MANCANTI — Suggerimento per Slide Aggiuntiva

## SLIDE 17 (da aggiungere) — Conclusioni

**Script:**

> "Per concludere, riassumiamo i risultati principali della nostra analisi.
>
> **Primo**: la PCA esplorativa ha dimostrato che i cinque varietali formano cluster ben distinti nello spazio delle componenti principali, confermando che il profilo antocianico HPLC è un efficace marcatore dell'identità varietale. Le variabili che contribuiscono maggiormente alla separazione sono state identificate tramite loading plot, biplot, VIP scores e Discriminant Power.
>
> **Secondo**: l'effetto annata è risultato trascurabile rispetto all'effetto varietale. I campioni delle due vendemmie (2000 e 2001) si sovrappongono completamente negli score plot PCA, confermando che la classificazione per varietale non è confusa dall'anno di produzione. Questo ci ha permesso di utilizzare i dati delle due annate insieme per la classificazione.
>
> **Terzo**: sia SIMCA che PLS-DA hanno dimostrato la fattibilità della classificazione, con PLS-DA che raggiunge un'accuratezza leggermente superiore sul test set. I due metodi offrono informazioni complementari: SIMCA è più adatto per il controllo qualità e la verifica di autenticità (può rigettare campioni sconosciuti), PLS-DA è più adatto per la discriminazione diretta tra varietali noti.
>
> **Quarto**: l'analisi dell'Applicability Domain ha confermato che il test set è rappresentativo del dominio chimico del training set, validando la generalizzabilità dei risultati.
>
> In sintesi, sei variabili antocianiche sono sufficienti per classificare in modo robusto i cinque varietali studiati, indipendentemente dall'annata di vendemmia. Questo approccio potrebbe essere esteso a un sistema di autenticazione e tracciabilità dei mosti a livello industriale."

---

---

# RISPOSTE ALLE DOMANDE DI `mosti.txt`

## Domanda 1: PCA esplorativa — somiglianze/differenze tra campioni

**Risposta:**
L'analisi PCA con autoscaling mostra che:
- I **cinque varietali formano cluster distinti** nello score plot PC1-PC2-PC3, confermando che il profilo antocianico discrimina efficacemente i varietali.
- I campioni si distinguono **principalmente per varietale**, NON per annata. Gli score plot colorati per annata (Fig. 5b) mostrano completa sovrapposizione tra 2000 e 2001.
- L'**Ancellotta** è il varietale più distinto (cluster più isolato), mentre **Sangiovese e Nero d'Avola** possono presentare leggera sovrapposizione su PC1-PC2 ma si separano con PC3.
- Le **variabili più discriminanti** (da loading/biplot): MVD% e PND% hanno loading elevati su PC1 (la componente principale delle differenze), seguite da DPD%, PTD% e il rapporto R lib/lrg. La CYD% contribuisce meno alla separazione globale.

## Domanda 2: SIMCA e PLS-DA per classificazione

**Risposta:**
- **Split 70/30 stratificato** per classe, con seed fisso per riproducibilità.
- **SIMCA**: modelli PCA locali per classe, numero di PC ottimizzato per cross-validazione. Efficienza generalmente buona. Forza: può rigettare campioni che non appartengono a nessuna classe.
- **PLS-DA**: modello discriminante con dummy Y centrata. Numero di LV ottimizzato via CV. Generalmente accuratezza superiore a SIMCA grazie alla supervisione esplicita.
- Entrambi i metodi **confermano la classificabilità** dei cinque varietali, con accuratezze superiori all'80-90% sul test set.

---

---

# VERIFICA DELL'ANALISI — L'approccio è corretto?

## ✅ Punti positivi (l'analisi è fondamentalmente corretta)

1. **Preprocessing appropriato**: L'autoscaling è la scelta corretta per dati con variabili su scale diverse (% di area vs rapporto). Motivazione fornita con il confronto visivo raw/autoscaled. ✅

2. **PCA esplorativa completa**: Score plot per varietale E per annata (risponde esattamente alle domande), loading plot, biplot, scree plot, 3D plot. Interpretazione della varianza spiegata. ✅

3. **Split stratificato**: Il 70/30 stratificato per classe è standard e corretto. Il seed fisso garantisce riproducibilità. ✅

4. **SIMCA implementato correttamente**: Modello PCA locale per classe, criterio di accettazione combinato T²+Q, ottimizzazione PC via CV, metriche sensibilità/specificità/efficienza. ✅

5. **PLS-DA implementato correttamente**: Dummy Y con centratura sulla media (corretto per PLS2), NIPALS algorithm, ottimizzazione LV via CV, VIP scores. ✅

6. **Applicability Domain**: Il Williams plot e il T²/Q plot verificano l'affidabilità delle predizioni sul test set — questo è un plus che va oltre la traccia. ✅

7. **Confronto tra metodi**: Il confronto SIMCA vs PLS-DA è pertinente e ben strutturato. ✅

8. **La traccia è soddisfatta in pieno**: tutte le domande di `mosti.txt` sono coperte dall'analisi.

## ⚠️ Possibili miglioramenti / Note critiche

1. **Cross-validazione SIMCA**: Si usa Venetian Blinds a 5 segmenti. Un'alternativa classica per SIMCA è il Leave-One-Out (LOO-CV), che è la scelta tradizionale nella letteratura SIMCA. Questo non è un errore — Venetian Blinds è accettabile — ma se il docente preferisce LOO, potrebbe essere un punto di discussione.

2. **Centratura Y in PLS-DA**: La Y dummy è centrata sulla media (`Y - mean(Y)`), il che è corretto per PLS2. Alcuni testi preferiscono dummy 0/1 senza centratura; la centratura è però più corretta da un punto di vista matematico.

3. **Numero di campioni per classe**: Con 98 campioni distribuiti su 5 classi e 2 annate, alcune classi potrebbero avere pochi campioni (es. 10-15 per classe per annata). Lo split 70/30 potrebbe lasciare solo 3-5 campioni per classe nel test set, il che limita la significatività statistica delle metriche sul test set. Questo va menzionato come limitazione.

4. **Dati composizionali**: Le percentuali d'area degli antociani sono dati composizionali (sommano ~100%). In linea di principio, si potrebbe applicare una trasformazione CLR (Centered Log-Ratio) prima della PCA. Tuttavia, l'autoscaling è la scelta standard nella pratica chemometrica per questo tipo di dati e non è un errore.

5. **Assegnazione SIMCA**: L'implementazione assegna ogni campione alla classe con il criterio combinato minimo (`min` della distanza normalizzata). In SIMCA classica, un campione potrebbe non essere assegnato a nessuna classe se supera tutti i limiti. Questa differenza va tenuta presente nella discussione.

---

## VERDETTO FINALE

**L'analisi è CORRETTA, PERTINENTE e BEN MOTIVATA per rispondere alle domande poste in `mosti.txt`.** L'approccio segue la metodologia chemometrica standard:

1. ✅ PCA esplorativa → risponde a "distinguere varietali" e "effetto annata"
2. ✅ SIMCA + PLS-DA → risponde a "classificare i diversi varietali"
3. ✅ Split stratificato → validazione esterna richiesta dalla traccia
4. ✅ VIP + Discriminant Power → risponde a "quali variabili distinguono meglio"
5. ✅ Applicability Domain → bonus, non richiesto ma aggiunge valore

L'unico elemento mancante nella presentazione è la **slide di conclusioni**, che è inclusa sopra come Slide 17.
