import time
import random
import numpy as np
from azul_env import AzulEnv

def main():
    """Main function to run the Azul game loop with improved round handling."""
    env = AzulEnv()
    observation, info = env.reset()
    done = False
    
    color_map = {'B': 0, 'Y': 1, 'R': 2, 'K': 3, 'C': 4}
    inv_color_map = {v: k for k, v in color_map.items()}

    print("Welcome to Azul!")
    print("You are 'user'. The agent is 'agent'.")

    while not done:
        current_player = env.current_player
        
        if current_player == 'agent':
            print("\n--- Agent's Turn ---")
            legal_moves = env.calculate_legal_moves('agent')
            action = random.choice(legal_moves)
            
            factory_index = action // ((env.rows + 1) * env.tiles)
            color_index = (action // (env.rows + 1)) % env.tiles
            row_index = action % (env.rows + 1)
            
            row_display = row_index + 1 if row_index < env.floor_row_index else "Floor"
            print(f"Agent chooses: Source {factory_index+1}, Color {inv_color_map[color_index]}, Row {row_display}")
            time.sleep(1)
            
            observation, reward, done, truncated, info = env.step(action)

        else: # User's turn
            env.render()
            print(f"\n--- Your Turn ({current_player}) ---")
            
            action = -1
            while True:
                try:
                    f_input = int(input(f"Choose a source [1-{env.num_factories} for factories, {env.num_factories+1} for center]: ")) - 1
                    source_tiles = env.factories[f_input] if f_input < env.num_factories else env.center_pool
                    if source_tiles.sum() == 0:
                        print("\n*** That source is empty. Please try again. ***")
                        continue

                    available_indices = np.where(source_tiles > 0)[0]
                    available_chars = [inv_color_map[i] for i in available_indices]
                    c_input_str = input(f"Choose an available color {available_chars}: ").upper()
                    if c_input_str not in available_chars:
                        print("\n*** Invalid or unavailable color. Please try again. ***")
                        continue
                    c_input = color_map[c_input_str]

                    r_input = int(input(f"Choose a storage row [1-5], or {env.floor_row_index + 1} for Floor: ")) - 1
                    if not (0 <= r_input <= env.floor_row_index):
                        print("\n*** Invalid row. Please try again. ***")
                        continue
                    
                    action_stride = (env.rows + 1) * env.tiles
                    user_action = f_input * action_stride + c_input * (env.rows + 1) + r_input
                    
                    if user_action in env.calculate_legal_moves('user'):
                        action = user_action
                        break
                    else:
                         print("\n*** That is not a valid destination row (e.g., mosaic slot is filled, or wrong color for row). Please try again. ***")

                except (ValueError, IndexError):
                    print("\n*** Invalid input. Please enter numbers for source and row. ***")

            observation, reward, done, truncated, info = env.step(action)
            # print(f"You earned {reward} points for that move.")

        # --- MODIFIED: Announce End of Round and show scores ---
        if info.get('round_ended') and not done:
            print("\n" + "#"*20 + " END OF ROUND " + "#"*20)
            print("Scores have been updated and factories refilled.")
            user_score = env.players['user']['score']
            agent_score = env.players['agent']['score']
            print(f"Current Score: You ({user_score}) vs Agent ({agent_score})")
            time.sleep(3) # Increased pause to let user read scores

    # Game is over
    print("\n" + "="*20 + " GAME OVER " + "="*20)
    env.render()
    user_score = env.players['user']['score']
    agent_score = env.players['agent']['score']
    print(f"\nFinal Score: You ({user_score}) vs Agent ({agent_score})")
    if user_score > agent_score: print("Congratulations, you won!")
    elif agent_score > user_score: print("The agent won. Better luck next time!")
    else: print("It's a draw!")

if __name__ == "__main__":
    main()
