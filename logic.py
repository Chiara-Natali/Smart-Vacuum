!git clone https://github.com/aimacode/aima-python.git

import sys
sys.path.append("/content/aima-python") # Use absolute path for robustness
from search import *

class SmartVacuum(Problem):
    def __init__(self, initial_grid, start_pos, goal_pos):
        # Lo stato è: (posizione_robot, griglia_immutabile)
        self.initial = (start_pos, tuple(tuple(row) for row in initial_grid))
        self.goal_pos = goal_pos
        self.rows = len(initial_grid)
        self.cols = len(initial_grid[0])

    def actions(self, state):
        curr_pos, grid = state
        row, col = curr_pos
        curr_cell_status = grid[row][col]
        possible_actions = []

        # Azione: Pulire
        if curr_cell_status in ('D', 'V'):
            possible_actions.append("Suck")

        # Azione: Movimento
        for delta_row, delta_col, move in [(-1, 0, 'Up'), (1, 0, 'Down'), (0, -1, 'Left'), (0, 1, 'Right')]:
            next_row, next_col = row + delta_row, col + delta_col # Corrected line
            if 0 <= next_row < self.rows and 0 <= next_col < self.cols:
                if grid[next_row][next_col] != 'X':
                    possible_actions.append(move)
        return possible_actions

    def result(self, state, action):
        curr_pos, grid = state
        r, c = curr_pos
        # Convertiamo la tupla di tuple in lista di liste per modificarla
        new_grid = [list(row) for row in grid]
        new_pos = curr_pos

        if action == "Suck":
            if new_grid[r][c] == 'V':
                new_grid[r][c] = 'D'
            elif new_grid[r][c] == 'D':
                new_grid[r][c] = 'C'
        else:
            moves = {'Up': (-1, 0), 'Down': (1, 0), 'Left': (0, -1), 'Right': (0, 1)}
            dr, dc = moves[action]
            new_pos = (r + dr, c + dc)

        return (new_pos, tuple(tuple(row) for row in new_grid))

    def goal_test(self, state):
        curr_pos, grid = state
        # 1. Deve essere in posizione finale
        if curr_pos != self.goal_pos:
            return False
        # 2. Tutto deve essere pulito (niente D o V)
        for row in grid:
            if 'D' in row or 'V' in row:
                return False
        return True

    def path_cost(self, c, state1, action, state2):
        return c + 1

    def h(self, node):
        curr_pos, grid = node.state
        curr_r, curr_c = curr_pos

        dirty_rooms = []
        total_cleaning_effort = 0

        for r in range(len(grid)):
            for c in range(len(grid[0])):
                status = grid[r][c]
                if status == 'D':
                    dirty_rooms.append((r, c))
                    total_cleaning_effort += 1
                elif status == 'V':
                    dirty_rooms.append((r, c))
                    total_cleaning_effort += 2

        # Se pulito, vai verso il traguardo (F)
        if not dirty_rooms:
            r_goal, c_goal = self.goal_pos # Access goal_pos via self
            return abs(curr_r - r_goal) + abs(curr_c - c_goal)

        # Distanza dalla stanza sporca più vicina + lavoro rimanente
        dist_to_nearest = min(abs(curr_r - dr) + abs(curr_c - dc) for dr, dc in dirty_rooms)
        return dist_to_nearest + total_cleaning_effort
