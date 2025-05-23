import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Per uno stile migliore
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path 


# --- CONFIGURAZIONE ---
PRE_RACE_DATASET_PATH = "data/Modello2/pre_race_prediction_dataset.csv" # Assicurati che il path sia corretto
# Piloti/Team da visualizzare (scegline alcuni significativi)
PILOTI_DA_VISUALIZZARE = ["VER", "HAM", "LEC", "NOR"] 
TEAM_DA_VISUALIZZARE = ["Red Bull Racing", "Mercedes", "Ferrari"] 
OUTPUT_ELO_PLOT_PATH = "diagrams/elo_evolution_plot.png" # Path per salvare il grafico
output_path=OUTPUT_ELO_PLOT_PATH
# --- FINE CONFIGURAZIONE ---

def plot_elo_evolution(df_pre_race, piloti_target, team_target, output_path):
    """
    Genera e salva un grafico dell'evoluzione dell'Elo per piloti e team selezionati.
    """
    if df_pre_race.empty:
        print("Dataset pre-gara vuoto, impossibile generare grafico Elo.")
        return

    # Assicurati che le colonne necessarie esistano
    req_cols_driver = ['anno', 'round', 'driver', 'elo_driver_pre_race']
    req_cols_team = ['anno', 'round', 'team', 'elo_team_pre_race']
    if not all(col in df_pre_race.columns for col in req_cols_driver) or \
       not all(col in df_pre_race.columns for col in req_cols_team):
        print("Colonne necessarie per l'Elo mancanti nel dataset.")
        return

    # Crea una colonna 'event_order' per l'asse X, combinando anno e round
    # per un ordinamento cronologico corretto
    df_pre_race = df_pre_race.sort_values(by=['anno', 'round']).copy()
    df_pre_race['event_str'] = df_pre_race['anno'].astype(str) + "-R" + df_pre_race['round'].astype(str).str.zfill(2)
    
    # Usa Seaborn per uno stile migliore
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=False) # Due subplot, uno per piloti, uno per team
    
    # Grafico Elo Piloti
    piloti_plot_df = df_pre_race[df_pre_race['driver'].isin(piloti_target)]
    if not piloti_plot_df.empty:
        sns.lineplot(data=piloti_plot_df, x='event_str', y='elo_driver_pre_race', hue='driver', ax=axes[0], marker='o', linewidth=2)
        axes[0].set_title('Evoluzione Rating Elo Piloti (Pre-Gara)', fontsize=16)
        axes[0].set_ylabel('Elo Pilota', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45, labelsize=8)
        axes[0].legend(title='Pilota')
    else:
        axes[0].text(0.5, 0.5, "Nessun dato Elo per i piloti selezionati.", horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
        axes[0].set_title('Evoluzione Rating Elo Piloti (Pre-Gara)', fontsize=16)


    # Grafico Elo Team
    team_plot_df = df_pre_race[df_pre_race['team'].isin(team_target)]
    # Rimuovi duplicati per team/evento se un team ha più piloti (prendi la media o il primo)
    team_plot_df = team_plot_df.drop_duplicates(subset=['event_str', 'team'])
    
    if not team_plot_df.empty:
        sns.lineplot(data=team_plot_df, x='event_str', y='elo_team_pre_race', hue='team', ax=axes[1], marker='o', linewidth=2)
        axes[1].set_title('Evoluzione Rating Elo Team (Pre-Gara)', fontsize=16)
        axes[1].set_xlabel('Evento Gara (Anno-Round)', fontsize=12)
        axes[1].set_ylabel('Elo Team', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45, labelsize=8) # Applica rotazione e dimensione
        axes[1].legend(title='Team')
    else:
        axes[1].text(0.5, 0.5, "Nessun dato Elo per i team selezionati.", horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)
        axes[1].set_title('Evoluzione Rating Elo Team (Pre-Gara)', fontsize=16)

    # Migliora la leggibilità degli X ticks se ci sono troppi eventi
    # Mostra solo ogni N-esimo tick per l'asse X se ci sono molti eventi
    if not piloti_plot_df.empty: # Usa il df dei piloti per determinare i ticks
        num_events = len(piloti_plot_df['event_str'].unique())
        if num_events > 30: # Se ci sono più di 30 eventi, dirada i tick
            tick_spacing = max(1, num_events // 15) # Mostra circa 15 tick
            for ax in axes:
                ax.set_xticks(ax.get_xticks()[::tick_spacing])


    plt.tight_layout()
    try:
        # Crea la directory se non esiste
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Grafico Elo salvato in: {output_path}")
        plt.close(fig) # Chiudi la figura per liberare memoria
    except Exception as e:
        print(f"Errore durante il salvataggio del grafico Elo: {e}")


if __name__ == "__main__":
    try:
        df_pre_race_data = pd.read_csv(PRE_RACE_DATASET_PATH)
        if not df_pre_race_data.empty:
            plot_elo_evolution(df_pre_race_data, PILOTI_DA_VISUALIZZARE, TEAM_DA_VISUALIZZARE, OUTPUT_ELO_PLOT_PATH)
        else:
            print(f"Il file {PRE_RACE_DATASET_PATH} è vuoto. Impossibile generare il grafico Elo.")
    except FileNotFoundError:
        print(f"File {PRE_RACE_DATASET_PATH} non trovato. Assicurati di aver eseguito build_dataset_pre_race.py.")
    except Exception as e:
        print(f"Errore generale nella generazione del grafico Elo: {e}")