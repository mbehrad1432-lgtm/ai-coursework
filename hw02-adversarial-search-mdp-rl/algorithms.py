# Course scaffold and revised submitted algorithms; see README for attribution.
import numpy as np
from gymnasium import Env, spaces
import matplotlib.pyplot as plt
from collections import defaultdict
import imageio
from IPython.display import HTML
from base64 import b64encode
import gymnasium as gym
import random
import os
from PIL import Image, ImageDraw
import math
import matplotlib.animation as animation
import networkx as nx
from numpy.linalg import norm
import copy
from functools import partial

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


ALL_CARDS = {
    "Tiger": [(0, 2), (0, -1)],
    "Dragon": [(-2, 1), (2, 1), (-1, -1), (1, -1)],
    "Frog": [(-2, 0), (-1, 1), (1, -1)],
    "Rabbit": [(1, 1), (-1, -1), (2, 0)],
    "Crab": [(-2, 0), (0, 1), (2, 0)],
    "Elephant": [(-1, 1), (1, 1), (-1, 0), (1, 0)],
    "Goose": [(-1, 1), (0, 1), (0, -1), (1, -1)],
    "Rooster": [(1, 1), (0, 1), (0, -1), (-1, -1)],
    "Monkey": [(-1, 1), (1, 1), (-1, -1), (1, -1)],
    "Mantis": [(-1, 1), (1, 1), (0, -1)],
    "Horse": [(0, 1), (-1, 0), (0, -1)],
    "Ox": [(0, 1), (1, 0), (0, -1)],
    "Crane": [(0, 1), (-1, -1), (1, -1)],
    "Boar": [(-1, 0), (0, 1), (1, 0)],
    "Eel": [(-1, 1), (1, 0), (-1, -1)],
    "Cobra": [(1, 1), (-1, 0), (1, -1)],
    
    "Your Custom Card": [(10, 10)],
}

class Onitama:
    def __init__(
            self, 
            board_size=5, 
            my_symbol='o', 
            opponent_symbol='x', 
            card1="Tiger", 
            card2="Crab", 
            card3="Elephant", 
            figsize=(9, 6), # Changes the animation size
            render_interval=800 # Changes the time between frames
            ):
        self.board_size = board_size
        self.my_symbol = my_symbol
        self.opponent_symbol = opponent_symbol
        self.card1 = card1
        self.card2 = card2
        self.card3 = card3
        self.figsize = figsize
        self.interval = render_interval
        
        m = 0
        for c in [ALL_CARDS.get(card1), ALL_CARDS.get(card2), ALL_CARDS.get(card3)]:
            for a, b in c:
                if abs(a) > m:
                    m = abs(a)
                if abs(b) > m:
                    m = abs(b)
        self.max_move = m

    def _init_board(self):
        board = [['.' for _ in range(self.board_size)] for _ in range(self.board_size)]
        for i in range(1, self.board_size - 1):
            board[0][i] = self.opponent_symbol
            board[-1][i] = self.my_symbol
        return board

    def init_game(self):
        return {
            "board": self._init_board(),
            "turn": 1,  # 1 = You, -1 = Opponent
            "my_card": self.card1,
            "opponent_card": self.card2,
            "center": self.card3,
        }

    def get_pieces(self, board, player):
        symbol = self.my_symbol if player == 1 else self.opponent_symbol
        return [(r, c) for r in range(self.board_size) for c in range(self.board_size) if board[r][c] == symbol]

    def apply_move(self, state, from_pos, to_pos):
        next_state = copy.deepcopy(state)
        r1, c1 = from_pos
        r2, c2 = to_pos
        piece = next_state["board"][r1][c1]
        next_state["board"][r1][c1] = '.'
        next_state["board"][r2][c2] = piece

        if next_state["turn"] == 1:
            next_state["my_card"], next_state["center"] = next_state["center"], next_state["my_card"]
        else:
            next_state["opponent_card"], next_state["center"] = next_state["center"], next_state["opponent_card"]

        next_state["turn"] *= -1
        return next_state

    def pass_turn(self, state):
        if self.generate_moves(state):
            raise ValueError("Passing is allowed only without legal moves")
        next_state = copy.deepcopy(state)
        key = "my_card" if state["turn"] == 1 else "opponent_card"
        next_state[key], next_state["center"] = next_state["center"], next_state[key]
        next_state["turn"] *= -1
        return next_state

    def in_bounds(self, r, c):
        return 0 <= r < self.board_size and 0 <= c < self.board_size

    def generate_moves(self, state):
        moves = []
        board = state["board"]
        player = state["turn"]
        card = state["my_card"] if player == 1 else state["opponent_card"]
        pieces = self.get_pieces(board, player)

        for (r, c) in pieces:
            for (dx, dy) in ALL_CARDS[card]:
                dc, dr = (-dx, -dy) if player == 1 else (dx, dy)
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc):
                    target = board[nr][nc]
                    if player == 1 and target != self.my_symbol:
                        moves.append((r, c, nr, nc))
                    elif player == -1 and target != self.opponent_symbol:
                        moves.append((r, c, nr, nc))
        return moves

    def evaluate(self, state):
        flat = [cell for row in state["board"] for cell in row]
        return flat.count(self.my_symbol) - flat.count(self.opponent_symbol)

    def terminal(self, state):
        if all(cell != self.opponent_symbol for row in state["board"] for cell in row):
            return True, 1
        if all(cell != self.my_symbol for row in state["board"] for cell in row):
            return True, -1
        return False, 0

    def _render(self, states):
        fig, axs = plt.subplots(2, 3, figsize=self.figsize, height_ratios=[1, 4], width_ratios=[1, 4, 1])

        ax_opp, ax_center, ax_player = axs[0]
        ax_board = axs[1][1]

        axs[1][0].axis('off')
        axs[1][2].axis('off')

        def render_card_moves(ax, card_name, is_opponent, max_move):
            s = 2 * max_move + 1
            ax.clear()
            ax.set_xticks(np.arange(s))
            ax.set_yticks(np.arange(s))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(-0.5, s - 0.5)
            ax.set_ylim(-0.5, s - 0.5)
            ax.grid(True)
            ax.set_aspect('equal')
            ax.set_title(card_name, fontsize=10)

            origin = (max_move, max_move)
            ax.add_patch(plt.Circle(origin, 0.2, color='black'))

            for dx, dy in ALL_CARDS.get(card_name, []):
                if is_opponent:
                    dx, dy = -dx, -dy
                x = origin[0] + dx
                y = origin[1] + dy
                if 0 <= x < s and 0 <= y < s:
                    ax.add_patch(plt.Circle((x, y), 0.2, color='blue'))

        def set_card_text(ax, card_name, label, color, max_move):
            render_card_moves(ax, card_name, label == "Opponent", max_move)
            ax.set_title(f"{label}: {card_name}", fontsize=10, color=color)

        def draw_board(state, my_symbol, opponent_symbol, board_size):
            ax_board.clear()
            ax_board.set_xticks(np.arange(board_size))
            ax_board.set_yticks(np.arange(board_size))
            ax_board.set_xticklabels([])
            ax_board.set_yticklabels([])
            ax_board.set_xlim(-0.5, board_size - 0.5)
            ax_board.set_ylim(-0.5, board_size - 0.5)
            ax_board.grid(True)
            ax_board.set_title("Onitama", fontsize=14)

            board = state["board"]
            for r in range(board_size):
                for c in range(board_size):
                    piece = board[r][c]
                    if piece == my_symbol:
                        ax_board.text(c, board_size - 1 - r, my_symbol, ha='center', va='center', fontsize=18, color='blue', weight='bold')
                    elif piece == opponent_symbol:
                        ax_board.text(c, board_size - 1 - r, opponent_symbol, ha='center', va='center', fontsize=18, color='red', weight='bold')

        def update(frame_idx, my_symbol, opponent_symbol, board_size, max_move):
            state = states[frame_idx]
            draw_board(state, my_symbol, opponent_symbol, board_size)

            set_card_text(ax_opp, state.get("opponent_card", ""), "Opponent", "red", max_move)
            set_card_text(ax_center, state.get("center", ""), "Center", "black", max_move)
            set_card_text(ax_player, state.get("my_card", ""), "You", "blue", max_move)

            if "winner" in state:
                winner = state["winner"]
                if winner == 1:
                    ax_board.set_title("Game Over: Player Wins", fontsize=14)
                elif winner == -1:
                    ax_board.set_title("Game Over: Opponent Wins", fontsize=14)
                else:
                    ax_board.set_title("Game Over: Draw", fontsize=14)

        ani = animation.FuncAnimation(fig, partial(update, my_symbol=self.my_symbol, opponent_symbol=self.opponent_symbol, board_size=self.board_size, max_move=self.max_move), frames=len(states), interval=self.interval, repeat=False)
        plt.close(fig)
        return HTML(ani.to_jshtml())


    def run(self, agent1, agent2, render=True, max_turns=150):
        state = self.init_game()
        states = [copy.deepcopy(state)]

        for _ in range(max_turns):
            terminal, winner = self.terminal(state)
            if terminal:
                state["winner"] = winner
                states.append(copy.deepcopy(state))
                break
            
            turn = state["turn"] == 1

            if turn:
                move = agent1.make_move(state)
            else:
                move = agent2.make_move(state)

            legal = self.generate_moves(state)
            if not legal:
                state = self.pass_turn(state)
            elif move not in legal:
                raise ValueError("Agent returned an illegal move")
            else:
                r1, c1, r2, c2 = move
                state = self.apply_move(state, (r1, c1), (r2, c2))
            states.append(copy.deepcopy(state))
        else:
            terminal, winner = self.terminal(state)
            state["winner"] = winner if terminal else 0
            state["stop_reason"] = "terminal" if terminal else "turn_limit"
            states.append(copy.deepcopy(state))

        self.last_states = states
        if render:
            return self._render(states)
        elif "winner" in states[-1]:
                winner = states[-1]["winner"]
                if winner == 1:
                    print("Game Over: Player Wins")
                elif winner == -1:
                    print("Game Over: Opponent Wins")
                else:
                    print("Game Over: Draw")

class Agent:
    def make_move(self, state):
        pass

class RandomAgent(Agent):
    def __init__(self, game: Onitama):
        self.game = game
    
    def make_move(self, state):
        moves = self.game.generate_moves(state)
        if len(moves) > 0:
            return random.choice(moves)
        else:
            return None

class Minimax(Agent):
    def __init__(self, game: Onitama, maximizing, depth):
        self.game = game
        self.maximizing = maximizing
        self.depth = depth

    def make_move(self, state):
        _, move = self.minimax(state, self.depth, self.maximizing)
        return move

    def minimax(self, state, depth, maximizing):
        terminal, winner = self.game.terminal(state)
        if terminal:
            return winner * 10000, None
        if depth == 0:
            return self.game.evaluate(state), None
        if not self.game.generate_moves(state):
            score, _ = self.minimax(self.game.pass_turn(state), depth - 1, not maximizing)
            return score, None

        if maximizing:
            max_eval = float('-inf')
            best_move = None
            for move in self.game.generate_moves(state):
                next_state = self.game.apply_move(state, (move[0], move[1]), (move[2], move[3]))
                eval, _ = self.minimax(next_state, depth - 1, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in self.game.generate_moves(state):
                next_state = self.game.apply_move(state, (move[0], move[1]), (move[2], move[3]))
                eval, _ = self.minimax(next_state, depth - 1, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
            return min_eval, best_move

class AlphaBeta(Agent):
    def __init__(self, game: Onitama, maximizing: bool, depth: int):
        self.game = game
        self.maximizing = maximizing
        self.depth = depth

    def make_move(self, state):
        _, best_move = self._alphabeta(state, self.depth, float('-inf'), float('inf'), self.maximizing)
        return best_move

    def _alphabeta(self, state, depth, alpha, beta, is_maximizing):
        terminal, winner = self.game.terminal(state)
        if terminal:
            return winner * 10000, None
        if depth == 0:
            return self.game.evaluate(state), None
        if not self.game.generate_moves(state):
            score, _ = self._alphabeta(self.game.pass_turn(state), depth - 1, alpha, beta, not is_maximizing)
            return score, None

        best_move = None
        legal_moves = self.game.generate_moves(state)

        if is_maximizing:
            max_score = float('-inf')
            for move in legal_moves:
                src_pos = (move[0], move[1])
                dst_pos = (move[2], move[3])
                next_state = self.game.apply_move(state, src_pos, dst_pos)

                score, _ = self._alphabeta(next_state, depth - 1, alpha, beta, False)

                if score > max_score:
                    max_score = score
                    best_move = move

                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return max_score, best_move
        else:
            min_score = float('inf')
            for move in legal_moves:
                src_pos = (move[0], move[1])
                dst_pos = (move[2], move[3])
                next_state = self.game.apply_move(state, src_pos, dst_pos)

                score, _ = self._alphabeta(next_state, depth - 1, alpha, beta, True)

                if score < min_score:
                    min_score = score
                    best_move = move

                beta = min(beta, score)
                if beta <= alpha:
                    break
            return min_score, best_move

class ScotlandYard:
    def __init__(
            self,
            police_start="A",
            x_start="J",
            move_probs=[0.9, 0.9, 0.9],
            x_stay_prob=0.1,
            win_reward=5,
            living_reward=-1,
            moving_rewards=[0, 0, 0],
            render_interval=1000,
            figsize=(6, 6),
            color_map={"taxi": "gold", "bus": "blue", "tunnel": "black"},
            ):

        self.police_start = police_start
        self.x_start = x_start
        self.move_probs = move_probs
        self.x_stay_prob = x_stay_prob
        self.win_reward = win_reward
        self.living_reward = living_reward
        self.moving_rewards = moving_rewards
        self.interval = render_interval
        self.figsize = figsize
        self.color_map = color_map

        self.actions = ["bus", "taxi" , "tunnel"]
        self.city_graph = {
            "A": {"taxi": "B", "bus": "C", "tunnel": "D"},
            "B": {"taxi": "A", "bus": "F", "tunnel": "C"},
            "C": {"taxi": "D", "bus": "A", "tunnel": "B"},
            "D": {"taxi": "C", "bus": "E", "tunnel": "A"},
            "E": {"taxi": "H", "bus": "D", "tunnel": "I"},
            "F": {"taxi": "G", "bus": "B", "tunnel": "J"},
            "G": {"taxi": "F", "bus": "I", "tunnel": "H"},
            "H": {"taxi": "E", "bus": "J", "tunnel": "G"},
            "I": {"taxi": "J", "bus": "G", "tunnel": "E"},
            "J": {"taxi": "I", "bus": "H", "tunnel": "F"},
        }
        self.cities = list(self.city_graph.keys())
        self.history = []
        self.states = [(p, x) for p in self.cities for x in self.cities]
        self.state_index = {s: i for i, s in enumerate(self.states)}
        self.index_state = {i: s for i, s in enumerate(self.states)}
        self.action_index = {a: i for i, a in enumerate(self.actions)}
        self._init_T_and_R()
        self.reset()

    def reset(self, random_start=True):
        if random_start:
            self.police_pos, self.x_pos = random.sample(self.cities, 2)
        else:
            self.police_pos = self.police_start
            self.x_pos = self.x_start
        self.done = False
        self.history = [(self.police_pos, self.x_pos)]
        return (self.police_pos, self.x_pos)

    def get_neighbors(self, pos):
        return sorted(set(self.city_graph[pos].values()))

    def move_mr_x(self):
        if random.random() < self.x_stay_prob:
            return self.x_pos
        neighbors = self.get_neighbors(self.x_pos)
        return random.choice(neighbors)

    def step(self, action):
        if self.done:
            raise ValueError("Game has ended. Call reset() to restart.")

        success = random.random() < self.move_probs[self.actions.index(action)]
        if success:
            new_police_pos = self.city_graph[self.police_pos][action]
        else:
            fallback_options = [a for a in self.actions if a != action and a in self.city_graph[self.police_pos]]
            if fallback_options:
                new_action = random.choice(fallback_options)
                new_police_pos = self.city_graph[self.police_pos][new_action]
            else:
                new_police_pos = self.police_pos

        new_x_pos = self.move_mr_x()

        self.police_pos = new_police_pos
        self.x_pos = new_x_pos
        self.history.append((self.police_pos, self.x_pos))

        if self.police_pos == self.x_pos:
            self.done = True
            reward = self.win_reward
        else:
            reward = self.living_reward

        reward += self.moving_rewards[self.actions.index(action)]

        return (self.police_pos, self.x_pos), reward, self.done

    def render(self):
        G = nx.Graph()
        edge_colors = []

        for node, transports in self.city_graph.items():
            for mode, neighbor in transports.items():
                if G.has_edge(node, neighbor):
                    continue
                G.add_edge(node, neighbor, transport=mode)
                edge_colors.append(self.color_map[mode])

        pos = nx.spring_layout(G, seed=23)
        fig, ax = plt.subplots(figsize=self.figsize)

        def update(frame):
            ax.clear()

            transport_labels = nx.get_edge_attributes(G, "transport")
            edge_colors_frame = [self.color_map[transport_labels[e]] for e in G.edges()]
            nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue',
                    edge_color=edge_colors_frame, node_size=500, font_size=10, width=2)

            police, x = self.history[frame]

            nx.draw_networkx_nodes(G, pos, nodelist=[police], node_color='green', node_size=600, ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=[x], node_color='red', node_size=600, ax=ax)

            ax.set_title(f"Step {frame+1}: Police at {police}, Mr. X at {x}")

        ani = animation.FuncAnimation(fig, update, frames=len(self.history), interval=self.interval, repeat=False)
        plt.close(fig)
        return HTML(ani.to_jshtml())
    def _init_T_and_R(self):
        S = len(self.states)
        A = len(self.actions)

        self.T = np.zeros((S, A, S))
        self.R = np.zeros((S, A, S))

        for current_state in self.states:
            police_pos, mr_x_pos = current_state
            if police_pos == mr_x_pos:
                idx = self.state_index[current_state]
                self.T[idx, :, idx] = 1.0
                continue  

            current_state_idx = self.state_index[current_state]

            mr_x_moves = [
                mr_x_pos,  
                self.city_graph[mr_x_pos]["taxi"],
                self.city_graph[mr_x_pos]["bus"],
                self.city_graph[mr_x_pos]["tunnel"]
            ]

            for action in self.actions:
                action_idx = self.action_index[action]

                intended = self.city_graph[police_pos][action]
                alt1 = self.city_graph[police_pos][self.actions[(action_idx + 1) % 3]]
                alt2 = self.city_graph[police_pos][self.actions[(action_idx + 2) % 3]]
                police_moves = [intended, alt1, alt2]

                move_probs = [
                    self.move_probs[action_idx],
                    (1 - self.move_probs[action_idx]) / 2,
                    (1 - self.move_probs[action_idx]) / 2
                ]

                for p_move, p_prob in zip(police_moves, move_probs):
                    next_state = (p_move, mr_x_pos)
                    next_idx = self.state_index[next_state]
                    prob = p_prob * self.x_stay_prob
                    reward = self.win_reward if p_move == mr_x_pos else self.living_reward
                    reward += self.moving_rewards[action_idx]
                    self.T[current_state_idx, action_idx, next_idx] += prob
                    self.R[current_state_idx, action_idx, next_idx] = reward

                    for mr_x_next in mr_x_moves[1:]:
                        next_state = (p_move, mr_x_next)
                        next_idx = self.state_index[next_state]
                        prob = p_prob * (1 - self.x_stay_prob) / 3
                        reward = self.win_reward if p_move == mr_x_next else self.living_reward
                        reward += self.moving_rewards[action_idx]
                        self.T[current_state_idx, action_idx, next_idx] += prob
                        self.R[current_state_idx, action_idx, next_idx] = reward

    pass

def validate(env: ScotlandYard, num_episodes=10000, max_steps_per_episode=30):
    S = len(env.states)
    A = len(env.actions)
    T_hat = np.zeros((S, A, S))
    R_hat = np.zeros((S, A, S))
    
    count_transitions = np.zeros((S, A, S))
    total_rewards = np.zeros((S, A, S))
    count_actions = np.zeros((S, A))
    
    for _ in range(num_episodes):
        state = env.reset(random_start=True)
        s = env.state_index[state]
        for _ in range(max_steps_per_episode):
            action = random.choice(env.actions)
            a = env.action_index[action]
            next_state, reward, done = env.step(action)
            s_prime = env.state_index[next_state]
            
            count_transitions[s, a, s_prime] += 1
            total_rewards[s, a, s_prime] += reward
            count_actions[s, a] += 1
            
            s = s_prime
            if done:
                break
    
    for s in range(S):
        for a in range(A):
            total = count_actions[s, a]
            if total == 0:
                continue
            for s_prime in range(S):
                T_hat[s, a, s_prime] = count_transitions[s, a, s_prime] / total
                if count_transitions[s, a, s_prime] > 0:
                    R_hat[s, a, s_prime] = total_rewards[s, a, s_prime] / count_transitions[s, a, s_prime]
    
    return T_hat, R_hat

def value_iteration(env: ScotlandYard, discount=0.95, theta=1e-6, max_iterations=10000):
    if not 0 <= discount < 1 or theta <= 0:
        raise ValueError("Require 0 <= discount < 1 and theta > 0")
    V = np.zeros(len(env.states))
    for _ in range(max_iterations):
        Q = np.sum(env.T * (env.R + discount * V[None, None, :]), axis=2)
        next_V = Q.max(axis=1)
        if np.max(np.abs(next_V - V)) < theta:
            V = next_V
            Q = np.sum(env.T * (env.R + discount * V[None, None, :]), axis=2)
            return V, Q.argmax(axis=1)
        V = next_V
    raise RuntimeError("Value iteration did not converge")


class CartPoleEnv(Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self):
        super().__init__()
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02

        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max
        ], dtype=np.float32)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0 = push left, 1 = push right

        self.state = None
        self.steps_beyond_done = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return self.state, {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold or x > self.x_threshold or
            theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        )

        reward = 1.0 if not done else 0.0
        return np.array(self.state, dtype=np.float32), reward, done, False, {}

    def render(self, mode='rgb_array'):
        screen_width = 608
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        x = self.state[0]
        theta = self.state[2]

        cartx = x * scale + screen_width / 2.0
        top = carty - cartheight / 2
        bottom = carty + cartheight / 2
        left = cartx - cartwidth / 2
        right = cartx + cartwidth / 2

        image = Image.new("RGB", (screen_width, screen_height), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        draw.rectangle([left, top, right, bottom], fill=(0, 0, 0))

        pole_x1 = cartx
        pole_y1 = carty
        pole_x2 = cartx + polelen * np.sin(theta)
        pole_y2 = carty - polelen * np.cos(theta)
        draw.line([pole_x1, pole_y1, pole_x2, pole_y2], fill=(255, 0, 0), width=5)

        if mode == 'rgb_array':
            return np.array(image)
        elif mode == 'human':
            image.show()

    def close(self):
        pass

def record_video_cartpole(env: CartPoleEnv, agent, out_dir="video", fps=30, max_steps=1000):
    obs, _ = env.reset(seed=SEED + 10000)
    done = False
    total_reward = 0
    frames = []
    os.makedirs(out_dir, exist_ok=True)

    for _ in range(max_steps):
        state = agent.obs_to_state(obs)
        action = agent.act(state, greedy=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        img = env.render()
        frames.append(img)
        if done:
            break

    path = os.path.join(out_dir, "cartpole.mp4")
    imageio.mimsave(path, frames, fps=fps)
    return total_reward, path

def show_video(video_path, width=400):
    with open(video_path, "rb") as f:
        encoded = b64encode(f.read()).decode()
    return HTML(f'<video width="{width}" controls><source src="data:video/mp4;base64,{encoded}" type="video/mp4">')

def plot_rewards(rewards, window=50):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode reward')
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(rewards)), moving_avg, label=f'{window}-episode moving avg', linewidth=2)
    plt.title("Training Reward over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    # plt.ylim(0, 1000)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

class QLearningAgent:
    def __init__(self, env: CartPoleEnv, number_states_for_feature, gamma, alpha, alpha_min, alpha_max, epsilon, epsilon_min, epsilon_max):
        self.env = env
        self.gamma = gamma
        self.state_bounds = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        self.state_bounds[1] = [-0.5, 0.5]
        self.state_bounds[3] = [-math.radians(50), math.radians(50)]

        self.number_states_for_feature = number_states_for_feature
        self.rng = random.Random(SEED)
        self.env.action_space.seed(SEED)
        self.epsilon = min(epsilon_max, max(epsilon_min, epsilon))
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.Q_table = {}

    def obs_to_state(self, observation):
        states_list = []
        for i in range(len(observation)):
            if observation[i] <= self.state_bounds[i][0]:
                state_index = 0
            elif observation[i] >= self.state_bounds[i][1]:
                state_index = self.number_states_for_feature[i] - 1
            else:
                quantum_size = (self.state_bounds[i][1] - self.state_bounds[i][0]) / self.number_states_for_feature[i]
                offset = observation[i] - self.state_bounds[i][0]
                state_index = math.floor(offset / quantum_size)
            states_list.append(state_index)
        return tuple(states_list)

    def act(self, state, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return self.env.action_space.sample()  

        return max(
            range(self.env.action_space.n),
            key=lambda action: self.Q_table.get((state, action), 0)
        )

    def schedule_epsilon(self, t):
        decay_rate = 0.9982
        self.epsilon = max(self.epsilon_min, self.epsilon * decay_rate)

    def schedule_alpha(self, t):
        if t < 1000:
            decay_rate = 0.9991
        else:
            decay_rate = 0.9982
        self.alpha = max(self.alpha_min, self.alpha * decay_rate)

    def train(self, episodes=1000):
        rewards = []

        for ep in range(episodes):
            self.schedule_epsilon(ep)
            self.schedule_alpha(ep)

            obs, _ = self.env.reset(seed=SEED + ep)
            total_reward = 0
            done = False

            while not done:
                state = self.obs_to_state(obs)
                action = self.act(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.obs_to_state(next_obs)

                max_q_next = max(
                    self.Q_table.get((next_state, a), 0)
                    for a in range(self.env.action_space.n)
                )

                old_q = self.Q_table.get((state, action), 0)
                target = reward + (0 if terminated else self.gamma * max_q_next)
                self.Q_table[state, action] = (1 - self.alpha) * old_q + self.alpha * target

                obs = next_obs
                total_reward += reward

            rewards.append(total_reward)

            if (ep + 1) % 100 == 0:
                print(f"Episode {ep + 1}: Reward = {total_reward}, Epsilon = {self.epsilon:.3f}")

        return rewards

    


def evaluate(env, agent, gamma=0.99, episodes=100, seed=20000):
    returns, counts = defaultdict(float), defaultdict(int)
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        trajectory = []
        done = False
        while not done:
            state = agent.obs_to_state(obs)
            action = agent.act(state, greedy=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            trajectory.append((state, reward))
        rewards.append(sum(r for _, r in trajectory))
        G = 0.0
        for state, reward in reversed(trajectory):
            G = reward + gamma * G
            returns[state] += G
            counts[state] += 1
    estimates = {s: returns[s] / counts[s] for s in counts}
    q_values = {s: max(agent.Q_table.get((s, a), 0.0) for a in range(env.action_space.n)) for s in counts}
    rmse = float(np.sqrt(np.mean([(q_values[s] - estimates[s]) ** 2 for s in counts])))
    return {"episodes": episodes, "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)), "min_reward": float(min(rewards)),
            "max_reward": float(max(rewards)), "visited_states": len(counts),
            "mc_q_rmse_visited_states": rmse}
