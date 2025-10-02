import gymnasium
import ale_py

import numpy as np
from PIL import Image
import os
import time

# --- Configuration ---
ENV_NAME = "ALE/Breakout-v5"
SAVE_DIR = "breakout_screenshots"

# Action mapping for Breakout-v0 (Discrete(4)):
# 0: NOOP, 1: FIRE (to launch the ball), 2: RIGHT, 3: LEFT
ACTION_MAP = {
    'n': 0,  # NOOP (No Operation)
    'f': 1,  # FIRE (Launches the ball or serves)
    'r': 2,  # RIGHT
    'l': 3,  # LEFT
}

ACTION_NAMES = {
    0: "NOOP",
    1: "FIRE",
    2: "RIGHT",
    3: "LEFT",
}
# --- Helper Functions ---

def ensure_directory_exists(directory):
    """Creates the target directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def save_screenshot(observation, action_name):
    """
    Saves the current observation (screenshot) to a directory named after the action.
    
    Args:
        observation (np.ndarray): The raw pixel data (RGB array) from the environment.
        action_name (str): The name of the action that led to this state.
    """
    # Create the action-specific subdirectory
    action_dir = os.path.join(SAVE_DIR, action_name)
    ensure_directory_exists(action_dir)
    
    # Generate a unique filename using a timestamp
    timestamp = int(time.time() * 1000)
    filename = f"step_{timestamp}_{action_name}.png"
    filepath = os.path.join(action_dir, filename)
    
    try:
        # Convert the NumPy array observation (H x W x C) to a PIL Image
        img = Image.fromarray(observation)
        img.save(filepath)
        # print(f"Saved screenshot to: {filepath}")
    except Exception as e:
        print(f"Error saving image: {e}")

def get_user_action():
    """Prompts the user for a valid action input."""
    valid_inputs = list(ACTION_MAP.keys())
    while True:
        prompt = (
            "\nChoose an action:\n"
            "  (n) NOOP\n"
            "  (f) FIRE/Start (use 'f' once to launch the ball)\n"
            "  (r) RIGHT\n"
            "  (l) LEFT\n"
            "  (q) QUIT\n"
            f"Enter key ({'/'.join(valid_inputs)}/q): "
        )
        user_input = input(prompt).strip().lower()
        
        if user_input == 'q':
            return None, None
        
        if user_input in ACTION_MAP:
            action_id = ACTION_MAP[user_input]
            action_name = ACTION_NAMES[action_id]
            return action_id, action_name
        else:
            print("Invalid input. Please choose one of the options.")

# --- Main Execution ---

def run_breakout_player():
    """Initializes the environment and runs the game loop."""
    print(f"Initializing Gym Environment: {ENV_NAME}")
    
    # We use both 'human' to open a game window for interaction,
    # and we rely on the 'rgb_array' observation for saving images.
    env = gymnasium.make(ENV_NAME, render_mode="human")
    
    ensure_directory_exists(SAVE_DIR)
    
    try:
        # Reset the environment and get the initial state
        initial_state = env.reset()
        # The Gym API changed; reset returns a tuple (observation, info)
        if isinstance(initial_state, tuple):
            observation, info = initial_state
        else:
            observation = initial_state # For older gym versions

        print("\n--- Breakout Player Activated ---")
        print("Use 'f' (FIRE) to start the game after the first reset.")
        print(f"Screenshots will be saved in the '{SAVE_DIR}' folder.")

        done = False
        total_reward = 0
        step_count = 0

        while not done:
            # 1. Get user input
            action_id, action_name = get_user_action()
            
            if action_id is None:
                print("Quitting game...")
                break # Exit the game loop
            
            # 2. Apply the action
            step_result = env.step(action_id)
            
            # Handle Gym API changes (often returns 4 or 5 values)
            if len(step_result) == 5:
                observation, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                # Handle older Gym API (usually 4 values)
                observation, reward, done, info = step_result 

            total_reward += reward
            step_count += 1
            
            # 3. Save the resulting state (screenshot)
            # We use the observation from the step, which is the state *after* the action.
            save_screenshot(observation, action_name)

            # 4. Display status
            print(f"Step: {step_count} | Action: {action_name} | Reward: {reward:.2f} | Total Score: {total_reward:.0f} | Lives: {info.get('lives', 'N/A')}")
            
            # Render the environment for the user to see the game state
            env.render()

        print("\n--- Game Over ---")
        print(f"Final Score: {total_reward:.0f}")

    except Exception as e:
        print(f"An error occurred during game execution: {e}")
        
    finally:
        # 5. Cleanup
        env.close()
        print("Environment closed. Thank you for playing!")

if __name__ == "__main__":
    # Check for required libraries
    try:
        # Check if Pillow is available
        import PIL
        # Check if gym is available
        import gymnasium
    except ImportError:
        print("--- DEPENDENCY ERROR ---")
        print("This script requires the 'gym' (or 'gymnasium') and 'Pillow' libraries.")
        print("Please install them using: pip install gym[atari] pillow")
        print("Note: You may need external dependencies for Atari, like 'cmake' or 'pip install cmake'")
        print("------------------------")
    else:
        run_breakout_player()
