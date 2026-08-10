import sys
import numpy as np
import random
import time
import csv
import os
import argparse
from collections import deque

# Safety net for deep recursive operations (though backprop is now iterative)
sys.setrecursionlimit(5000)

# =====================================================================
# 1. OPTIMIZED GAME ENGINES
# =====================================================================

class HexState:
    def __init__(self, board_size=11, board=None, turn=1):
        self.board_size = board_size
        self.turn = turn
        
        if board is None:
            self.board = np.zeros((board_size, board_size), dtype=int)
        else:
            self.board = np.copy(board)
            
        self._cached_winner = None
        self._is_terminal_cached = False

    def get_legal_moves(self):
        if self.is_terminal():
            return []
        rows, cols = np.where(self.board == 0)
        return list(zip(rows, cols))

    def make_move(self, move):
        row, col = move
        if self.board[row, col] != 0:
            raise ValueError(f"Move {move} is invalid.")
        next_board = np.copy(self.board)
        next_board[row, col] = self.turn
        return HexState(self.board_size, next_board, 3 - self.turn)

    def get_neighbors(self, row, col):
        neighbors = []
        directions = [
            (row - 1, col), (row - 1, col + 1),
            (row, col - 1), (row, col + 1),
            (row + 1, col - 1), (row + 1, col)
        ]
        for r, c in directions:
            if 0 <= r < self.board_size and 0 <= c < self.board_size:
                neighbors.append((r, c))
        return neighbors

    def check_winner(self):
        if self._cached_winner is not None:
            return self._cached_winner

        if self._has_connected_path(player=1):
            self._cached_winner = 1
            return 1
        if self._has_connected_path(player=2):
            self._cached_winner = 2
            return 2
        return None

    def _has_connected_path(self, player):
        queue = deque()
        visited = set()

        if player == 1:
            for c in range(self.board_size):
                if self.board[0, c] == 1:
                    queue.append((0, c))
                    visited.add((0, c))
        else:
            for r in range(self.board_size):
                if self.board[r, 0] == 2:
                    queue.append((r, 0))
                    visited.add((r, 0))

        while queue:
            curr_r, curr_c = queue.popleft() 
            if player == 1 and curr_r == self.board_size - 1:
                return True
            if player == 2 and curr_c == self.board_size - 1:
                return True

            for nr, nc in self.get_neighbors(curr_r, curr_c):
                if self.board[nr, nc] == player and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    def is_terminal(self):
        if self._is_terminal_cached:
            return True
        if self.check_winner() is not None:
            self._is_terminal_cached = True
            return True
        if not np.any(self.board == 0):
            self._is_terminal_cached = True
            return True
        return False


class GomokuState:
    def __init__(self, board_size=15, board=None, turn=1, last_move=None):
        self.board_size = board_size
        self.turn = turn
        self.last_move = last_move
        
        if board is None:
            self.board = np.zeros((board_size, board_size), dtype=int)
        else:
            self.board = np.copy(board)
            
        self._cached_winner = None
        self._is_terminal_cached = False

    def get_legal_moves(self):
        if self.is_terminal():
            return []
        rows, cols = np.where(self.board == 0)
        return list(zip(rows, cols))

    def make_move(self, move):
        row, col = move
        if self.board[row, col] != 0:
            raise ValueError(f"Move {move} is invalid.")
        next_board = np.copy(self.board)
        next_board[row, col] = self.turn
        return GomokuState(self.board_size, next_board, 3 - self.turn, last_move=move)

    def check_winner(self):
        if self._cached_winner is not None:
            return self._cached_winner
        if self.last_move is None:
            return None

        r, c = self.last_move
        player = self.board[r, c]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            step = 1
            while True:
                nr, nc = r + dr * step, c + dc * step
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr, nc] == player:
                    count += 1
                    step += 1
                else:
                    break
            step = 1
            while True:
                nr, nc = r - dr * step, c - dc * step
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr, nc] == player:
                    count += 1
                    step += 1
                else:
                    break
            if count >= 5:
                self._cached_winner = player
                return player
        return None

    def is_terminal(self):
        if self._is_terminal_cached:
            return True
        if self.check_winner() is not None:
            self._is_terminal_cached = True
            return True
        if not np.any(self.board == 0):
            self._is_terminal_cached = True
            return True
        return False

# =====================================================================
# 2. MCTS ARCHITECTURE
# =====================================================================

class MCTSNode:
    def __init__(self, state, parent=None, move_from_parent=None):
        self.state = state
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.children = {}
        self.visit_count = 0
        self.win_count = 0  
        self.untried_moves = state.get_legal_moves()
        random.shuffle(self.untried_moves)
        
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
        
    def is_terminal_node(self):
        return self.state.is_terminal()
        
    def expand(self):
        move = self.untried_moves.pop()
        next_state = self.state.make_move(move)
        child_node = MCTSNode(next_state, parent=self, move_from_parent=move)
        self.children[move] = child_node
        return child_node
        
    def backpropagate(self, result):
        # Iterative backpropagation prevents recursion limit crashes
        node = self
        while node is not None:
            node.visit_count += 1
            player_who_just_moved = 3 - node.state.turn
            if result == player_who_just_moved:
                node.win_count += 1
            elif result is None:
                node.win_count += 0.5 
            node = node.parent

def select_child_uct(node, exploration_constant=1.0):
    best_score = -float('inf')
    best_child = None
    for child in node.children.values():
        exploit = child.win_count / child.visit_count
        explore = exploration_constant * np.sqrt(np.log(node.visit_count) / child.visit_count)
        score = exploit + explore
        if score > best_score:
            best_score = score
            best_child = child
    return best_child

def select_child_thompson(node, prior_alpha=2.0, prior_beta=1.0):
    best_sample = -float('inf')
    best_child = None
    for child in node.children.values():
        alpha = prior_alpha + child.win_count
        beta = prior_beta + (child.visit_count - child.win_count)
        sampled_value = np.random.beta(alpha, beta)
        if sampled_value > best_sample:
            best_sample = sampled_value
            best_child = child
    return best_child

def select_child_hybrid(node, depth_threshold=5, exploration_constant=1.0, prior_alpha=2.0, prior_beta=1.0):
    """
    Hybrid Selection: Uses UCT for shallow, data-rich nodes and Thompson Sampling 
    for deep, data-sparse nodes to prevent deterministic over-exploration.
    """
    depth = 0
    temp = node
    while temp.parent is not None:
        depth += 1
        temp = temp.parent
        
    if depth < depth_threshold:
        return select_child_uct(node, exploration_constant=exploration_constant)
    else:
        return select_child_thompson(node, prior_alpha=prior_alpha, prior_beta=prior_beta)

def random_rollout(state, max_depth=40):
    current_state = state
    depth = 0
    while not current_state.is_terminal():
        if max_depth and depth >= max_depth:
            # Heuristic fallback if depth limit is reached
            p1 = np.sum(current_state.board == 1)
            p2 = np.sum(current_state.board == 2)
            return 1 if p1 > p2 else (2 if p2 > p1 else None)
            
        legal_moves = current_state.get_legal_moves()
        move = random.choice(legal_moves)
        current_state = current_state.make_move(move)
        depth += 1
    return current_state.check_winner()

def run_mcts(root_state, iterations, selection_strategy, **strategy_kwargs):
    root_node = MCTSNode(state=root_state)
    for _ in range(iterations):
        node = root_node
        while node.is_fully_expanded() and not node.is_terminal_node():
            node = selection_strategy(node, **strategy_kwargs)
        if not node.is_terminal_node():
            node = node.expand()
        
        result = random_rollout(node.state, max_depth=None)
        node.backpropagate(result)
        
    best_move = max(root_node.children.items(), key=lambda item: item[1].visit_count)[0]
    return best_move, root_node

def calculate_root_entropy(root_node):
    if not root_node.children:
        return 0.0
    total_visits = sum(child.visit_count for child in root_node.children.values())
    if total_visits == 0:
        return 0.0
    entropy = 0.0
    for child in root_node.children.values():
        if child.visit_count > 0:
            p_i = child.visit_count / total_visits
            entropy -= p_i * np.log(p_i)
    return entropy

def clean_move(move):
    return (int(move[0]), int(move[1]))

# =====================================================================
# 3. TOURNAMENT RUNNER (WITH CHUNKING SUPPORT)
# =====================================================================

def run_tournament_chunk(game_class, board_size, start_id, chunk_size, iterations, csv_filename, strategy_a, strategy_b):
    file_exists = os.path.isfile(csv_filename)
    
    print(f"==================================================")
    print(f"LAUNCHING HPC TOURNAMENT CHUNK")
    print(f"Game: {game_class.__name__} | Board Size: {board_size}x{board_size}")
    print(f"Matchup: {strategy_a} vs {strategy_b}")
    print(f"Target Games Range: {start_id} to {start_id + chunk_size - 1}")
    print(f"Iteration Budget: {iterations}")
    print(f"File Output Destination: {csv_filename}")
    print(f"==================================================\n")

    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Game_Type", "Board_Size", "Game_ID", "Player_1_Strategy", "Player_2_Strategy", 
                "Winner", "Total_Moves", 
                "Avg_UCT_Entropy", "Avg_TS_Entropy", "Avg_Hybrid_Entropy", 
                "Avg_UCT_Time", "Avg_TS_Time", "Avg_Hybrid_Time"
            ])

        for game_id in range(start_id, start_id + chunk_size):
            # Alternate opening turns for fairness based on dynamically passed strategies
            if game_id % 2 != 0:
                p1_strategy, p2_strategy = strategy_a, strategy_b
            else:
                p1_strategy, p2_strategy = strategy_b, strategy_a
                
            print(f"Game {game_id}: P1 ({p1_strategy}) vs P2 ({p2_strategy})...", end="", flush=True)
            state = game_class(board_size=board_size)
            
            # Dictionary maps to gracefully handle the varying strategies
            entropies = {"UCT": [], "Thompson": [], "Hybrid": []}
            times = {"UCT": [], "Thompson": [], "Hybrid": []}
            
            move_count = 0
            game_start = time.time()
            
            while not state.is_terminal():
                current_player_strategy = p1_strategy if state.turn == 1 else p2_strategy
                step_start = time.time()
                
                if current_player_strategy == "UCT":
                    move, root = run_mcts(state, iterations=iterations, selection_strategy=select_child_uct, exploration_constant=1.0)
                elif current_player_strategy == "Thompson":
                    move, root = run_mcts(state, iterations=iterations, selection_strategy=select_child_thompson, prior_alpha=2.0, prior_beta=1.0)
                elif current_player_strategy == "Hybrid":
                    move, root = run_mcts(state, iterations=iterations, selection_strategy=select_child_hybrid, depth_threshold=5, exploration_constant=1.0, prior_alpha=2.0, prior_beta=1.0)
                    
                times[current_player_strategy].append(time.time() - step_start)
                entropies[current_player_strategy].append(calculate_root_entropy(root))
                
                state = state.make_move(clean_move(move))
                move_count += 1
                
            winner_id = state.check_winner()
            game_winner = p1_strategy if winner_id == 1 else (p2_strategy if winner_id == 2 else "Draw")
            elapsed = time.time() - game_start
            
            print(f" Finished! Winner: {game_winner} ({move_count} moves, {elapsed:.1f}s)")
            
            writer.writerow([
                game_class.__name__, board_size, game_id, p1_strategy, p2_strategy,
                game_winner, move_count, 
                np.mean(entropies["UCT"]) if entropies["UCT"] else 0, 
                np.mean(entropies["Thompson"]) if entropies["Thompson"] else 0, 
                np.mean(entropies["Hybrid"]) if entropies["Hybrid"] else 0, 
                np.mean(times["UCT"]) if times["UCT"] else 0, 
                np.mean(times["Thompson"]) if times["Thompson"] else 0,
                np.mean(times["Hybrid"]) if times["Hybrid"] else 0
            ])
            f.flush() # Force immediate flush to disk to preserve data in arrays

    print(f"\nTournament chunk complete! Results saved to '{csv_filename}'.")

# =====================================================================
# 4. COMMAND LINE INTERFACE (CLI)
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCTS Tournament on CSF with Job Arrays.")
    parser.add_argument("--game", type=str, choices=["hex", "gomoku"], required=True, help="Which game to play.")
    parser.add_argument("--board_size", type=int, default=None, help="Board size. Defaults: Hex=11, Gomoku=15.")
    parser.add_argument("--iterations", type=int, default=2500, help="MCTS iterations per move.")
    parser.add_argument("--start_id", type=int, default=1, help="Starting game ID for this job's chunk.")
    parser.add_argument("--chunk_size", type=int, default=10, help="Number of games this job will run.")
    parser.add_argument("--strategy_a", type=str, choices=["UCT", "Thompson", "Hybrid"], default="UCT", help="Player 1 strategy for first game.")
    parser.add_argument("--strategy_b", type=str, choices=["UCT", "Thompson", "Hybrid"], default="Thompson", help="Player 2 strategy for first game.")
    
    args = parser.parse_args()

    if args.game == "hex":
        target_class = HexState
        b_size = args.board_size if args.board_size else 11
        out_csv = f"results/csf_hex_{b_size}x{b_size}_{args.strategy_a}_vs_{args.strategy_b}_chunk{args.start_id}.csv"
    else:
        target_class = GomokuState
        b_size = args.board_size if args.board_size else 15
        out_csv = f"results/csf_gomoku_{b_size}x{b_size}_{args.strategy_a}_vs_{args.strategy_b}_chunk{args.start_id}.csv"

    run_tournament_chunk(
        game_class=target_class,
        board_size=b_size,
        start_id=args.start_id,
        chunk_size=args.chunk_size,
        iterations=args.iterations,
        csv_filename=out_csv,
        strategy_a=args.strategy_a,
        strategy_b=args.strategy_b
    )