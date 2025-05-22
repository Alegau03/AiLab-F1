# 🏎️ F1 AI Predictor: Predizione Avanzata dei Risultati di Gara in Formula 1

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Un progetto accademico che esplora l'uso del Machine Learning per predire i risultati dei Gran Premi di Formula 1, sfruttando dati telemetrici, cronometrici e contestuali. Il sistema implementa due modelli predittivi distinti e un'interfaccia utente interattiva.

**Autori:** Alessandro Gautieri (2041850), Giovanni Cinieri (2054772)
**Corso:** AI Lab
**Anno Accademico:** 2024/2025

---

## 📝 Panoramica del Progetto

L'analisi predittiva nel motorsport, e in particolare nella Formula 1, rappresenta una sfida affascinante data la natura dinamica e multifattoriale delle competizioni. Questo progetto si propone di sviluppare un sistema innovativo per la predizione della posizione finale di ciascun pilota.

Il progetto si articola in due approcci principali:

1.  **Modello 1 (Predizione Durante la Gara):** Utilizza dati telemetrici, cronometrici e contestuali (come i rating Elo di piloti e team) relativi ai primi 30 giri di gara per predire la classifica finale. Si basa su un sofisticato **ensemble ponderato posizionalmente** di algoritmi di Gradient Boosting Decision Trees (GBDT): **LightGBM** e **CatBoost**.
2.  **Modello 2 (Predizione Pre-Gara):** Fornisce una stima della classifica finale prima dell'inizio della competizione, utilizzando informazioni quali la griglia di partenza fornita dall'utente, i rating Elo più recenti e le performance storiche dei piloti e dei team su specifici circuiti. Anche questo modello si avvale di **CatBoost**.

Entrambi i sistemi sono stati integrati in un'**interfaccia utente interattiva (GUI) realizzata con Gradio**, permettendo l'esplorazione delle predizioni per gare concluse (stagioni 2023, 2024 e gare 2025 disputate) e per gare future della stagione 2025.

➡️ **Per una descrizione dettagliata del progetto, delle metodologie, dei risultati e delle analisi, consulta il nostro report completo: `REPORT_PROGETTO_F1_AI.pdf` (incluso in questo repository).**

---

## ✨ Funzionalità Chiave

*   **Acquisizione Dati Avanzata:** Utilizzo della libreria `FastF1` per raccogliere dati dettagliati dalle sessioni di F1.
*   **Feature Engineering Sofisticato:**
    *   Creazione di feature telemetriche sintetiche.
    *   Integrazione di metriche relative al leader e alla competizione (differenziali, percentili).
    *   Implementazione di un sistema di **Rating Elo dinamico** per piloti e team.
    *   Calcolo di statistiche storiche di performance su specifici circuiti per il Modello 2.
*   **Modellazione Robusta:**
    *   **Modello 1:** Ensemble ottimizzato di LightGBM (con hyperparameter tuning via Optuna) e CatBoost, con una strategia di ponderazione posizionale per le predizioni.
    *   **Modello 2:** CatBoostRegressor addestrato su feature pre-gara.
*   **Due Scenari di Predizione:**
    *   Analisi di gare concluse con predizioni basate sui dati dei primi 30 giri.
    *   Predizioni pre-gara per eventi futuri basate sulla griglia di partenza fornita dall'utente.
*   **Interfaccia Utente Interattiva:** Una GUI sviluppata con `Gradio` per un' facile esplorazione e interazione con i modelli.
*   **Pipeline Dati Flessibile:** Script separati per la costruzione del dataset storico (`build_dataset.py`), per l'aggiornamento con i dati della stagione corrente (`build_dataset_current_season.py`), e per la creazione del dataset pre-gara (`build_dataset_pre_race.py`).

---

## 🛠️ Struttura del Progetto e Tecnologie Utilizzate

*   **Linguaggio:** Python
*   **Acquisizione Dati:** `FastF1`
*   **Manipolazione Dati:** `pandas`, `numpy`
*   **Modellazione:**
    *   `scikit-learn` (per pipeline, preprocessing, metriche)
    *   `LightGBM`
    *   `CatBoost`
    *   `Optuna` (per hyperparameter tuning)
    *   `joblib` (per serializzazione modelli)
*   **Interfaccia Utente:** `Gradio`
*   **Visualizzazione (per analisi e report):** `matplotlib`, `seaborn`, `Graphviz`



## 🚀 Come Eseguire

**Prerequisiti:**

*   Python 3.9+
*   Aver installato le dipendenze elencate in `requirements.txt`. Si consiglia un ambiente virtuale.
    ```bash
    pip install -r requirements.txt
    ```
*   Per il supporto GPU (LightGBM, CatBoost), assicurarsi che i driver NVIDIA e il CUDA Toolkit siano configurati correttamente sul sistema. Consultare la documentazione ufficiale delle rispettive librerie per i dettagli sull'installazione con supporto GPU.

**Passaggi:**

1.  **Preparazione dei Dataset:**
    *   Assicurarsi che `data/gare_per_anno.csv` e `data/gare_2025.csv` siano presenti e aggiornati.
    *   Popolare le cartelle `data/GareYYYY/` con i file CSV sorgente per ogni GP (se si vogliono rigenerare i dataset da zero). Questi file non sono inclusi nel repository per motivi di dimensione.
    *   Eseguire gli script di build per generare i dataset principali:
        ```bash
        python build_dataset.py 
        python build_dataset_current_season.py # Per i dati 2025 disputati
        python build_dataset_pre_race.py       # Per il dataset del Modello 2
        ```
    *   *Nota: Se i file `training_dataset_COMPLETO.csv`, `training_dataset_2025.csv`, e `pre_race_prediction_dataset.csv` sono già presenti e corretti, questo passaggio può essere saltato.*

2.  **Training dei Modelli:**
    *   Eseguire gli script di training. I modelli verranno salvati nella cartella `models/`.
        ```bash
        python train_model.py   # Addestra Modello 1 (LGBM + CatBoost ensemble)
        python train_model2.py  # Addestra Modello 2 (CatBoost pre-gara)
        ```
    *   *Nota: Gli script sono progettati per caricare i modelli se già addestrati e presenti nella cartella `models/`, saltando il retraining. Per forzare il retraining, eliminare i file `.joblib` corrispondenti dalla cartella `models/`.*

3.  **Avvio dell'Interfaccia Gradio:**
    *   Una volta che i dataset sono pronti e i modelli sono stati addestrati (o se si usano modelli pre-addestrati forniti), avviare l'applicazione:
        ```bash
        python interfaccia_finale.py
        ```
    *   Aprire il browser all'indirizzo locale fornito (solitamente `http://127.0.0.1:7860`).

4.  **(Opzionale) Generazione Grafici Diagnostici:**
    *   Per generare i grafici di analisi delle performance dei modelli (come quelli inclusi nella relazione):
        ```bash
        python plot_model_diagnostics.py
        ```
    *   I grafici verranno salvati nella cartella `diagrams/`.

---

## 📊 Risultati Chiave

Le metriche di valutazione dettagliate per entrambi i modelli sono disponibili nel report (`REPORT_PROGETTO_F1_AI.pdf`). In sintesi:

*   **Modello 1 (Durante la Gara):** Ha dimostrato una buona capacità predittiva, con un MAE per l'ensemble di circa **2.28 - 2.38 posizioni** e un R² di circa **0.63 - 0.69** sul set di test (i valori esatti possono variare leggermente in base all'ultima esecuzione di training sul dataset combinato).
*   **Modello 2 (Pre-Gara):** Come atteso, presenta una precisione inferiore (MAE ~2.7, R² ~0.56), ma riesce comunque a catturare segnali predittivi significativi basati sulle informazioni disponibili prima della partenza.

---

## 💡 Sviluppi Futuri

*   Tuning più approfondito degli iperparametri del Modello 2.
*   Esplorazione di feature alternative o aggiuntive (es. condizioni meteo, aggiornamenti vetture).
*   Estensione del Modello 1 a un numero maggiore di giri o a fasi specifiche della gara.
*   Adattamento del sistema per l'utilizzo con flussi di dati in tempo reale (se disponibili).
*   Esplorazione di tecniche di ensemble più sofisticate (es. stacking).

---

## 📄 Licenza

Questo progetto è rilasciato sotto la Licenza MIT. Vedi il file `LICENSE` per maggiori dettagli (se decidi di aggiungerne uno).

---
