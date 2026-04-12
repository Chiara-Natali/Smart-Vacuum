import tensorflow as tf

MODEL_PATH = "/content/models/model.h5"
IMAGE_PATH = "/content/grid1.jpeg"
# classes = ['V', 'D', 'C', 'X', 'S', 'F']

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Modello '{MODEL_PATH}' caricato correttamente.")
except Exception as e:
    print(f"Errore nel caricamento del modello: {e}")
    sys.exit()

# 2. FASE DI VISIONE
print("Analisi dell'immagine in corso...")
map_classes = classify_grid(MODEL_PATH, IMAGE_PATH)

if len(map_classes) != 9:
    print(f"Errore: Rilevate {len(map_classes)} celle invece di 9.")
    sys.exit()

# 3. COSTRUZIONE MATRICE 3x3
grid = [map_classes[i:i + 3] for i in range(0, 9, 3)]

# 4. RICERCA START E GOAL
start_pos = None
goal_pos = None
for r in range(3):
    for c in range(3):
        if grid[r][c] == 'S': start_pos = (r, c)
        if grid[r][c] == 'F': goal_pos = (r, c)

if not start_pos or not goal_pos:
    print("Errore: 'S' o 'F' non trovate.")
    sys.exit()

# 5. LOGICA E RISOLUZIONE
problem = SmartVacuum(grid, start_pos, goal_pos)

print("\n--- 2. Ricerca Percorso ---")
choice = input("\n Eseguire:\n[1] A* \n[2] BFS\n")
if choice == '1':
    node_astar = astar_search(problem, h=problem.h)
    if node_astar:
      print("Soluzione A* trovata!")
      print(f"Azioni: {node_astar.solution()}")
    else:
      print("Nessuna soluzione trovata")
elif choice == '2':
    node_bfs = breadth_first_graph_search(problem)
    if node_bfs:
        print("Soluzione BFS trovata!")
        print(f"Azioni: {node_bfs.solution()}")
    else:
      print("Nessuna soluziona trovata")
