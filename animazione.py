import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np

# Reconstruct the sequence of states from the problem's solution
states = [problem.initial]
current_state = problem.initial
for action in node_astar.solution():
    current_state = problem.result(current_state, action)
    states.append(current_state)

all_actions = ["Start"] + node_astar.solution()

cell_colors_map = {
    'S': 0,
    'F': 1,
    'V': 2,
    'D': 3,
    'C': 4,
    'X': 5
}

cmap_list = ['#2ca02c', '#808080', '#520052', '#A300A3', '#ffffff', '#ff0000']
cmap = plt.cm.colors.ListedColormap(cmap_list)

bounds = np.array(list(cell_colors_map.values())) - 0.5
bounds = np.append(bounds, max(cell_colors_map.values()) + 0.5)
norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

# Figure con due sotto-assi affiancati: griglia a sinistra, testo a destra
fig, (ax, ax_text) = plt.subplots(
    1, 2,
    figsize=(9, 6),
    gridspec_kw={'width_ratios': [4, 2]}
)

# Configurazione asse griglia
ax.set_xticks(np.arange(problem.cols + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(problem.rows + 1) - 0.5, minor=True)
ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
ax.set_xticks([])
ax.set_yticks([])

img_plot = ax.imshow(np.zeros((problem.rows, problem.cols)), cmap=cmap, norm=norm,
                     origin='upper', extent=[-0.5, problem.cols - 0.5, problem.rows - 0.5, -0.5])

# Pallino GIALLO
robot_marker, = ax.plot([], [], 'o', color='yellow', markersize=20, zorder=5)

cell_char_texts = [
    [ax.text(c, r, '', ha='center', va='center', color='black', fontsize=10) for c in range(problem.cols)]
    for r in range(problem.rows)
]

# Asse testo a destra: invisibile, scritte una sotto l'altra
ax_text.axis('off')
action_text = ax_text.text(0.05, 0.75, '', transform=ax_text.transAxes,
                            color='black', fontsize=12, verticalalignment='top')
status_text = ax_text.text(0.05, 0.55, '', transform=ax_text.transAxes,
                            color='black', fontsize=12, verticalalignment='top')

plt.tight_layout()

def update(frame):
    current_pos, current_grid_tuple = states[frame]
    current_action = all_actions[frame]

    grid_numeric = np.zeros((problem.rows, problem.cols))
    for r in range(problem.rows):
        for c in range(problem.cols):
            cell_val = current_grid_tuple[r][c]

            if (r, c) == start_pos and cell_val == 'C':
                grid_numeric[r, c] = cell_colors_map['S']
            elif (r, c) == goal_pos and cell_val == 'C':
                grid_numeric[r, c] = cell_colors_map['F']
            else:
                grid_numeric[r, c] = cell_colors_map.get(cell_val, cell_colors_map['C'])

            char_to_display = current_grid_tuple[r][c]
            if (r, c) == start_pos and char_to_display == 'C':
                char_to_display = 'S(C)'
            elif (r, c) == goal_pos and char_to_display == 'C':
                char_to_display = 'F(C)'

            cell_char_texts[r][c].set_position((c, r))
            cell_char_texts[r][c].set_text(char_to_display)

            if grid_numeric[r, c] in [cell_colors_map['X'], cell_colors_map['V']]:
                cell_char_texts[r][c].set_color('white')
            else:
                cell_char_texts[r][c].set_color('black')

    img_plot.set_array(grid_numeric)
    robot_marker.set_data([current_pos[1]], [current_pos[0]])

    action_text.set_text(f'Action: {current_action}')
    status_text.set_text(
        f'Robot at: ({current_pos[0]}, {current_pos[1]})\n'
        f'Cell status: {current_grid_tuple[current_pos[0]][current_pos[1]]}'
    )

    return [img_plot, robot_marker, action_text, status_text] + \
           [text for row_texts in cell_char_texts for text in row_texts]

anim = animation.FuncAnimation(fig, update, frames=len(states), interval=1000, blit=True, repeat=False)

plt.close(fig)
HTML(anim.to_jshtml())
