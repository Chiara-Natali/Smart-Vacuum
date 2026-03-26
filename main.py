import tensorflow as tf
from search import astar_search, breadth_first_graph_search
from classifier import classify_grid 
from logic import SmartVacuum, smart_vacuum_h

def main():
    MODEL_PATH = "model.h5"
    IMAGE_PATH = "grid1.jpg"
    classes = ['V', 'D', 'C', 'X', 'S', 'F']

    print("--- 1. Inizializzazione ---")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Modello '{MODEL_PATH}' caricato correttamente.")
    except Exception as e:
        print(f"Errore nel caricamento del modello: {e}")
        return

    # 2. FASE DI VISIONE
    print("Analisi dell'immagine in corso...")
    mappa_letta = classify_grid(IMAGE_PATH, model, classes) 
    
    if len(mappa_letta) != 9:
        print(f"Errore: Rilevate {len(mappa_letta)} celle invece di 9.")
        return

    # 3. COSTRUZIONE MATRICE 3x3
    grid = [mappa_letta[i:i + 3] for i in range(0, 9, 3)]
    
    # 4. RICERCA START E GOAL
    start_pos = None
    goal_pos = None
    for r in range(3):
        for c in range(3):
            if grid[r][c] == 'S': start_pos = (r, c)
            if grid[r][c] == 'F': goal_pos = (r, c)

    if not start_pos or not goal_pos:
        print("Errore: 'S' o 'F' non trovate.")
        return

    # 5. LOGICA E RISOLUZIONE
    problem = SmartVacuum(grid, start_pos, goal_pos)
    
    print("\n--- 2. Ricerca Percorso ---")
    scelta = input("\n Eseguire: \n[1] A* \n[2] BFS")
    if scelta == '1':
        # Esecuzione A*
        node_astar = astar_search(problem, h=smart_vacuum_h)
    elif scelta == '2':
        # Esecuzione BFS
        node_bfs = breadth_first_graph_search(problem)

    if node_astar:
        print("Soluzione A* trovata!")
        print(f"Azioni: {node_astar.solution()}")
        stampa_simulazione(node_astar)
    elif node_bfs:
        print("Soluzione BFS trovata!")
        print(f"Azioni: {node_bfs.solution()}")
        stampa_simulazione(node_bfs)
    else:
        print("Nessuna soluzione trovata.")

def stampa_simulazione(node_finale):
    percorso = node_finale.path()
    print(f"--- INIZIO SIMULAZIONE ({len(percorso)-1} azioni totali) ---")
    print(f" Posizione Robot: {posizione_robot}")
    
    for i, nodo in enumerate(percorso):
        posizione_robot, griglia = nodo.state
        azione_fatta = nodo.action # Sarà None per il primo nodo (lo start)
        
        print(f"\nPasso {i}:")
        if azione_fatta:
            print(f" -> Azione eseguita: {azione_fatta}")
        else:
            print(" -> Stato Iniziale")
            
        for r in range(3):
            riga_testo = " | ".join(griglia[r])
            print(f" [ {riga_testo} ]")

if __name__ == "__main__":
    main()
