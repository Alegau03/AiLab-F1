# Progetto AI Lab: Predizione Risultati Gare di Formula 1

**Autori:**
*   Alessandro Gautieri (Matricola: 2041850)
*   Giovanni Cinieri (Matricola: 2054772)

**Anno Accademico:** 2024/2025 (AI Lab)

---

## Panoramica del Progetto

Questo progetto è stato sviluppato con l'obiettivo di creare e addestrare sistemi di Machine Learning capaci di predire i risultati finali delle gare di Formula 1. Sono stati implementati due modelli distinti per affrontare differenti scenari predittivi:

1.  **Modello 1:** Predice la classifica finale di una gara basandosi sui dati telemetrici, cronometrici e contestuali dei primi 30 giri (per gare già disputate).
2.  **Modello 2:** Effettua una predizione pre-gara della classifica finale, utilizzando la griglia di partenza fornita dall'utente, i rating Elo di piloti e team, e le performance storiche sul circuito specifico.

Entrambi i modelli e le loro funzionalità sono accessibili tramite un'interfaccia utente interattiva.

Una **relazione di progetto dettagliata (`AiLab report.pdf`)** è inclusa in questa cartella e fornisce un'analisi approfondita delle metodologie, dei risultati e delle discussioni.

---

## Esecuzione del Progetto

Per interagire con il sistema predittivo e testarne il funzionamento, è sufficiente eseguire lo script principale dell'interfaccia utente:

```bash
python interfaccia_finale.py
```

## Struttura e Organizzazione dei File del Progetto
Di seguito una descrizione dell'organizzazione delle cartelle e dei file principali:
### 📁 `code/`
In questa cartella si trova tutto il codice sorgente sviluppato
È strutturato come segue:

*   **`Modello1/`**:
    *   `datasetBuild/`: Contiene gli script per la creazione dei dataset per il Modello 1:
        *   `build_dataset.py`: Per il dataset storico (2023-2024).
        *   `build_dataset_current_season.py`: Per il dataset della stagione corrente (es. 2025, gare disputate).
    *   `next/`: Contiene lo script `train_model.py` per l'addestramento e la valutazione del Modello 1 (ensemble LightGBM + CatBoost).
*   **`Modello2/`**:
    *   `datasetBuild/`: Contiene lo script `build_dataset_pre_race.py` per la creazione del dataset specifico per le predizioni pre-gara del Modello 2.
    *   `next/`: Contiene lo script `train_model2.py` per l'addestramento e la valutazione del Modello 2 (CatBoost pre-gara).
*   **`varie/`**:
    *   Contiene script ausiliari utilizzati durante lo sviluppo, come quelli per il raccoglimento iniziale dei dati da `FastF1` (se separati dalla logica di `build_dataset`), script di validazione dei dataset (`validate_dataset.py`, `validate_pre_race_dataset.py`), e lo script per la generazione dei grafici di analisi (`plot_model_diagnostics.py`).

### 📁 utils/
`elo.py`: Contiene la logica per il calcolo e la gestione dei rating Elo dinamici per piloti e team, inclusa la funzione per recuperare lo storico degli Elo.

### 📁 models/
Questa cartella è destinata a contenere i modelli addestrati e serializzati (.joblib) pronti per essere caricati dall'interfaccia o da altri script.
Include:
- lgbm_pipeline.joblib (Modello 1)
- catboost_pipeline.joblib (Modello 1)
- pre_race_catboost_pipeline.joblib (Modello 2)
- positional_weights.json (Per l'ensamble pesato)
### 📁 catboost_info/ (generata automaticamente da CatBoost)
Contiene log e informazioni ausiliarie generate durante il training dei modelli CatBoost.
### 📁 lightning_logs/ 
Contiene i log e i checkpoint generati da PyTorch Lightning.
### 📁 data/
Questa cartella contiene tutti i file di dati utilizzati e generati dal progetto:
Gare2023/, Gare2024/, Gare2025/:
Queste sottocartelle contengono i file CSV sorgente (data_NOMEGP_ANNO.csv) per ogni Gran Premio, utilizzati dagli script `build_dataset.py` e `build_dataset_current_season.py`. Questi file rappresentano l'input grezzo per la creazione dei dataset di training per il Modello 1.
training_dataset_COMPLETO.csv:
Il dataset principale per il Modello 1, contenente i dati elaborati per le stagioni 2023-2024.
training_dataset_2025.csv:
Dataset per il Modello 1, contenente i dati elaborati per le gare della stagione 2025 già disputate. Viene concatenato a quello storico per il training.
pre_race_prediction_dataset.csv:
Il dataset specifico per l'addestramento del Modello 2 (predizioni pre-gara).
gare_per_anno.csv:
File di calendario contenente informazioni su tutte le gare delle stagioni considerate (anno, round, nome GP, ecc.), utilizzato principalmente per il calcolo dell'Elo e per mappare i nomi dei GP.
gare_2025.csv:
File di calendario specifico per la stagione 2025, utilizzato dall'interfaccia per popolare l'elenco completo delle gare del 2025.
### 📁 diagrams/
Contiene le immagini e i diagrammi generati per l'analisi e inclusi nella relazione di progetto.
### 📁 cache/
Utilizzata dalla libreria FastF1 per memorizzare nella cache i dati scaricati dalle API della F1, al fine di velocizzare i caricamenti successivi.
Speriamo questa guida fornisca una chiara comprensione della struttura e dell'organizzazione del progetto.

# NOTA IMPORTANTE
Troverà alcuni parte di codice (commenti o print) in inglese mentre altri in italiano, semplicemente perchè avevamo iniziato a scrivere in inglese ma poi abbiamo saputo che non era obbligatorio e ci recava solo una perdita di tempo e abbiamo inziato a scrivere in italiano.