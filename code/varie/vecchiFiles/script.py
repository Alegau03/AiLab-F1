from graphviz import Digraph
from pathlib import Path 
# --- Codice per generare il diagramma dell'Ensemble Pesato ---

dot = Digraph(comment='Ensemble Pesato Posizionale', format='png')
dot.attr(rankdir='TB', size='8,6', label="Architettura dell'Ensemble Pesato Posizionale (Modello 1)", fontsize='20', labelloc="t")
dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightyellow', fontname='Helvetica')
dot.attr('edge', fontname='Helvetica', fontsize='10')

# Input Nodi
dot.node('X_test', 'Dati di Test\n(Giro N)')

# Modelli Base
with dot.subgraph(name='cluster_models') as models_cluster:
    models_cluster.attr(label='Modelli Base Addestrati', style='filled', color='lightgrey', fontname='Helvetica')
    models_cluster.node('LGBM', 'LightGBM Pipeline\n(Modello 1 - LGBM)')
    models_cluster.node('CB', 'CatBoost Pipeline\n(Modello 1 - CatBoost)')

# Predizioni dai Modelli Base
dot.node('Pred_LGBM', 'Predizioni Grezze LGBM\n(valori float)')
dot.node('Pred_CB', 'Predizioni Grezze CatBoost\n(valori float)')

# Logica di Ponderazione
dot.node('Avg_Pred_Pos', 'Stima Posizione Media\n(basata su Pred_LGBM, Pred_CB)')
dot.node('Weights_DB', 'Database Pesi Posizionali\n(positional_weights.json)\n(calcolato su dati di training)')
dot.node('Get_Weights', 'Recupero Pesi Specifici\nper Posizione Media Stimata')

# Applicazione dei Pesi e Combinazione
dot.node('Weighted_Sum', 'Somma Ponderata delle Predizioni\n(per ogni campione)')
dot.node('Normalization', 'Normalizzazione (se somma pesi != 1)') # Opzionale ma buona pratica
dot.node('Final_Raw_Pred', 'Predizione Ensemble Grezza\n(valore float)')

# Ranking Finale
dot.node('Ranking', 'Applicazione Ranking\n(conversione in posizioni 1°, 2°, ...)')
dot.node('Final_Output', 'Predizione Finale Ensemble\n(Posizioni Classifica)')


# Connessioni
dot.edge('X_test', 'LGBM')
dot.edge('X_test', 'CB')

dot.edge('LGBM', 'Pred_LGBM')
dot.edge('CB', 'Pred_CB')

dot.edge('Pred_LGBM', 'Avg_Pred_Pos')
dot.edge('Pred_CB', 'Avg_Pred_Pos')

dot.edge('Avg_Pred_Pos', 'Get_Weights')
dot.edge('Weights_DB', 'Get_Weights')

dot.edge('Get_Weights', 'Weighted_Sum')
dot.edge('Pred_LGBM', 'Weighted_Sum')
dot.edge('Pred_CB', 'Weighted_Sum')

dot.edge('Weighted_Sum', 'Normalization')
dot.edge('Normalization', 'Final_Raw_Pred')
dot.edge('Final_Raw_Pred', 'Ranking')
dot.edge('Ranking', 'Final_Output')


# Salva e visualizza
try:
    # Crea la directory se non esiste
    output_dir = Path("diagrams")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / 'ensemble_pesato_posizionale'
    
    dot.render(str(file_path), view=False) # Salva come .png e .dot
    print(f"Diagramma '{file_path}.png' generato.")
    
    # Se vuoi anche SVG per qualità migliore:
    dot_svg = dot.copy()
    dot_svg.format = 'svg'
    dot_svg.render(str(file_path), view=False)
    print(f"Diagramma '{file_path}.svg' generato.")

except Exception as e:
    print(f"Errore durante la generazione del diagramma Graphviz: {e}")
    print("Assicurati di avere Graphviz installato e nel PATH di sistema.")