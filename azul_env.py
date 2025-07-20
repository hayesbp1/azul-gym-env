import numpy as np
import gymnasium as gym
from gymnasium import spaces

class AzulEnv(gym.Env):
    """
    A Gymnasium environment for the board game Azul.
    This version includes logic for "floor moves" to prevent no-move deadlocks
    and ensures the game rules are followed more closely.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # --- Game Configuration ---
        self.rows = 5
        self.tiles = 5
        self.num_factories = 5
        self.floor_row_index = 5 # Use index 5 to represent the floor

        # --- Game State ---
        self.factories = np.zeros((self.num_factories, self.tiles), dtype=int)
        self.center_pool = np.zeros(self.tiles, dtype=int)
        self.bag = np.array([20] * self.tiles, dtype=int)
        self.discard_pile = np.zeros(self.tiles, dtype=int)
        self.penalty_token_in_center = True
        self.next_start_player = None

        # --- Player State ---
        def blank_board():
            return {
                "storage_rows": np.zeros((self.rows, self.tiles), dtype=int),
                "mosaic": np.zeros((self.rows, self.tiles), dtype=int),
                "floor_line": np.zeros(7, dtype=int),
                "floor_line_index": 0,
                "score": 0
            }

        self.players = {
            "user": blank_board(),
            "agent": blank_board()
        }
        self.current_player = "user"

        # --- Gym Spaces (MODIFIED) ---
        # Action space now includes the floor row (self.rows + 1 total rows)
        self.action_space = spaces.Discrete((self.num_factories + 1) * self.tiles * (self.rows + 1))

        obs_len = self.factories.size + self.bag.size + self.center_pool.size + 1 \
                  + 2 * (self.players['user']['storage_rows'].size +
                         self.players['user']['mosaic'].size +
                         self.players['user']['floor_line'].size)
        
        self.observation_space = spaces.Box(low=0, high=20,
                                              shape=(obs_len,),
                                              dtype=np.int32)

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state."""
        super().reset(seed=seed)
        self.bag = np.array([20, 20, 20, 20, 20], dtype=int)
        self.discard_pile.fill(0)
        self.center_pool.fill(0)
        self.factories.fill(0)
        self.penalty_token_in_center = True
        self.next_start_player = "user"
        self.current_player = "user"
        for player in self.players.values():
            player["storage_rows"].fill(0)
            player["mosaic"].fill(0)
            player["floor_line"].fill(0)
            player["floor_line_index"] = 0
            player["score"] = 0
        self.fill_factories()
        return self.get_observation(), {}

    def fill_factories(self):
        """Fills factories with random tiles from the bag."""
        for i in range(self.num_factories):
            for _ in range(4):
                if np.sum(self.bag) == 0:
                    if np.sum(self.discard_pile) == 0: return
                    self.refill_bag()
                available_colors = np.where(self.bag > 0)[0]
                if len(available_colors) == 0:
                    self.refill_bag()
                    available_colors = np.where(self.bag > 0)[0]
                    if len(available_colors) == 0: return
                color = self.np_random.choice(available_colors)
                self.factories[i, color] += 1
                self.bag[color] -= 1

    def refill_bag(self):
        """Refills the tile bag from the discard pile."""
        self.bag += self.discard_pile
        self.discard_pile.fill(0)

    def get_observation(self):
        """Assembles the observation vector from the current game state."""
        global_state = np.concatenate([
            self.factories.flatten(), self.bag, self.center_pool,
            np.array([1 if self.penalty_token_in_center else 0])
        ])
        player_keys = sorted(self.players.keys())
        player_states = []
        for key in player_keys:
            player = self.players[key]
            player_state = np.concatenate([
                player["storage_rows"].flatten(), player["mosaic"].flatten(), player["floor_line"]
            ])
            player_states.append(player_state)
        return np.concatenate([global_state, *player_states]).astype(np.int32)

    @staticmethod
    def calculate_tile_score(mosaic: np.ndarray, row: int, col: int) -> int:
        """Calculates the score for placing a tile on the mosaic."""
        if mosaic[row, col] == 0: return 0
        score, h_score, v_score = 0, 0, 0
        for i in range(5):
            if mosaic[row, i] != 0: h_score += 1
            else:
                if i < col: h_score = 0
                else: break
        for i in range(5):
            if mosaic[i, col] != 0: v_score += 1
            else:
                if i < row: v_score = 0
                else: break
        if h_score > 1 and v_score > 1: score = h_score + v_score
        elif h_score > 1: score = h_score
        elif v_score > 1: score = v_score
        else: score = 1
        return score

    def step(self, action):
        """Processes an action and updates the environment state."""
        player = self.players[self.current_player]
        score_before = player["score"]
        info = {} # Initialize info dictionary

        # --- 1. Decode Action ---
        factory_index = action // ((self.rows + 1) * self.tiles)
        color = (action // (self.rows + 1)) % self.tiles
        row = action % (self.rows + 1)

        # --- 2. Execute Action ---
        if factory_index < self.num_factories:
            source = self.factories[factory_index]
            count = source[color]
            leftovers = np.copy(source)
            leftovers[color] = 0
            self.center_pool += leftovers
            source.fill(0)
        else: # Center pool
            source = self.center_pool
            count = source[color]
            source[color] = 0
            if self.penalty_token_in_center:
                if player["floor_line_index"] < 7:
                    player["floor_line"][player["floor_line_index"]] = 1
                    player["floor_line_index"] += 1
                self.penalty_token_in_center = False
                if self.next_start_player is None:
                    self.next_start_player = self.current_player
        
        # --- 3. Place Tiles ---
        tiles_to_row = 0
        if row < self.floor_row_index:
            storage_row = player["storage_rows"][row]
            mosaic_col = self.azul_column(row, color)
            is_storage_legal = (player["mosaic"][row, mosaic_col] == 0 and
                                (storage_row.sum() == 0 or np.argmax(storage_row) == color) and
                                storage_row.sum() < row + 1)

            if is_storage_legal:
                row_capacity = row + 1
                current_fill = np.sum(storage_row)
                available_capacity = row_capacity - current_fill
                tiles_to_row = min(count, available_capacity)
                player["storage_rows"][row, color] += tiles_to_row

        remainder = count - tiles_to_row
        for _ in range(remainder):
            if player["floor_line_index"] < 7:
                player["floor_line"][player["floor_line_index"]] = color + 2
                player["floor_line_index"] += 1
            else:
                self.discard_pile[color] += 1

        # --- 4. Check for Round End ---
        round_finished = (self.factories.sum() == 0 and self.center_pool.sum() == 0)
        game_over = False
        if round_finished:
            game_over = self.end_round()
            info['round_ended'] = True 
        else:
            self.current_player = "agent" if self.current_player == "user" else "user"

        # --- 5. Calculate Reward ---
        score_after = player["score"]
        reward = score_after - score_before
        
        return self.get_observation(), reward, game_over, False, info

    @staticmethod
    def azul_column(row: int, color: int) -> int:
        """Determines the fixed column on the mosaic for a given color and row."""
        return (color + row) % 5

    def end_round(self) -> bool:
        """Resolves scoring, penalties, and prepares for the next round."""
        floor_penalty_map = [-1, -1, -2, -2, -2, -3, -3]
        for player in self.players.values():
            # Score storage rows
            for r in range(self.rows):
                capacity = r + 1
                if np.sum(player["storage_rows"][r]) == capacity:
                    color = np.argmax(player["storage_rows"][r])
                    col = self.azul_column(r, color)
                    if player["mosaic"][r, col] == 0:
                        player["mosaic"][r, col] = color + 1
                        player["score"] += self.calculate_tile_score(player["mosaic"], r, col)
                        self.discard_pile[color] += capacity - 1
                        player["storage_rows"][r].fill(0)
            # Score floor penalties
            penalty = 0
            for i in range(player["floor_line_index"]):
                tile_val = player["floor_line"][i]
                if tile_val > 1: self.discard_pile[tile_val - 2] += 1
                penalty += floor_penalty_map[i]
            player["score"] = max(0, player["score"] + penalty)
            player["floor_line"].fill(0)
            player["floor_line_index"] = 0
        # Check for game over
        game_over = any(np.any(np.all(p["mosaic"] > 0, axis=1)) for p in self.players.values())
        if game_over:
            self.apply_end_game_bonuses()
            return True
        # Prepare next round
        self.fill_factories()
        self.penalty_token_in_center = True
        self.current_player = self.next_start_player if self.next_start_player is not None else self.current_player
        self.next_start_player = None
        return False

    def apply_end_game_bonuses(self):
        """Adds end-game bonus points."""
        for player in self.players.values():
            bonus = 0
            bonus += np.sum(np.all(player["mosaic"] > 0, axis=1)) * 2
            bonus += np.sum(np.all(player["mosaic"] > 0, axis=0)) * 7
            for color_val in range(1, self.tiles + 1):
                if np.sum(player["mosaic"] == color_val) == 5:
                    bonus += 10
            player["score"] += bonus

    def render(self, mode='human'):
        """Prints a human-readable representation of the game state."""
        if mode != 'human': return
        colors = ["\033[94mB\033[0m", "\033[93mY\033[0m", "\033[91mR\033[0m", "\033[90mK\033[0m", "\033[96mC\033[0m"]
        print("\n" + "="*50)
        print(" " * 20 + "AZUL GAME STATE")
        print("="*50)
        print("\n--- Factories ---")
        for i, factory in enumerate(self.factories):
            f_str = f"F{i+1}: "
            for color_idx, count in enumerate(factory):
                if count > 0: f_str += f"{colors[color_idx]}x{count} "
            print(f_str if f_str.strip() != f"F{i+1}:" else f"F{i+1}: Empty")
        print("\n--- Center Pool ---")
        c_str = ""
        if self.penalty_token_in_center: c_str += " \033[95m(P)\033[0m "
        for color_idx, count in enumerate(self.center_pool):
            if count > 0: c_str += f"{colors[color_idx]}x{count} "
        print(c_str if c_str else "Empty")
        for player_key, player in self.players.items():
            print(f"\n--- {player_key.upper()}'S BOARD (Score: {player['score']}) ---")
            print(f"{'Storage Rows':<35}{'Mosaic'}")
            for r in range(self.rows):
                capacity = r + 1
                storage_display_list = []
                count = int(player['storage_rows'][r].sum())
                if count > 0:
                    color_idx = np.argmax(player['storage_rows'][r])
                    tile_str = f"[{colors[color_idx]}]"
                    storage_display_list.extend([tile_str] * count)
                storage_display_list.extend(["."] * (capacity - count))
                storage_content = " ".join(storage_display_list)
                visual_length = len(storage_content) - (count * 8)
                padding = 20 - visual_length
                full_storage_str = f"Row {r+1}: ({storage_content}{' ' * padding})"
                m_row_str = ""
                for c in range(self.tiles):
                    tile_val = player['mosaic'][r, c]
                    if tile_val > 0:
                        m_row_str += f"[{colors[tile_val-1]}]"
                    else:
                        required_color = (c - r + self.tiles) % self.tiles
                        m_row_str += f"({colors[required_color].lower()})"
                print(f"{full_storage_str}   {m_row_str}")
            f_line_str = "Floor Line: "
            floor_penalty_map = [-1, -1, -2, -2, -2, -3, -3]
            for i in range(player['floor_line_index']):
                f_line_str += f"[{floor_penalty_map[i]}] "
            print(f_line_str)
        print("\n" + "="*50)

    def calculate_legal_moves(self, player_key):
        """
        Returns a list of all legal actions. A move is defined by picking
        a color from a source and designating a destination row (including the floor).
        This guarantees a player always has a move if tiles are on the board.
        """
        legal_moves = []
        player = self.players[player_key]
        sources = list(enumerate(self.factories)) + [(self.num_factories, self.center_pool)]
        
        action_stride = (self.rows + 1) * self.tiles

        for factory_idx, source in sources:
            if source.sum() == 0: continue
            
            available_colors = np.where(source > 0)[0]
            for color in available_colors:
                # A player can always choose to send tiles to the floor
                legal_moves.append(factory_idx * action_stride + color * (self.rows + 1) + self.floor_row_index)
                
                # Check which storage rows are valid destinations
                for row in range(self.rows):
                    storage_row = player["storage_rows"][row]
                    mosaic_col = self.azul_column(row, color)
                    
                    is_legal_storage = (player["mosaic"][row, mosaic_col] == 0 and
                                       (storage_row.sum() == 0 or np.argmax(storage_row) == color) and
                                       storage_row.sum() < row + 1)
                    
                    if is_legal_storage:
                        action = factory_idx * action_stride + color * (self.rows + 1) + row
                        legal_moves.append(action)
        
        return list(set(legal_moves))
