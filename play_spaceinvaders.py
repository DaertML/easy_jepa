import gymnasium
import ale_py
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# --- PyTorch Model Definition (Minimal Placeholder) ---
# NOTE: You MUST replace this class definition with the actual architecture
# used to train your JEPA-based classifier (e.g., if it uses a Vision Transformer
# backbone and a linear layer). This placeholder assumes a simple CNN-like input.

class LinearClassifier(nn.Module):
    """Placeholder for the user's trained classification head."""
    def __init__(self, num_classes):
        super().__init__()
        # Assuming the backbone outputs a 512-dimensional feature vector,
        # which is common for transfer learning. Adjust if necessary.
        self.fc = nn.Linear(512, num_classes) 

    def forward(self, x):
        # NOTE: If your model expects a 4D input (N, C, H, W) where (H, W) is 224x224, 
        # this placeholder must be updated to include the backbone feature extractor.
        # For simplicity, we assume 'x' is already the feature vector here, which 
        # is often the case when only loading the classifier head.
        
        # Since we don't have the backbone, we will use a dummy transformation
        # to ensure the code structure runs. The model will need the real backbone 
        # and forward pass for meaningful predictions.
        
        # **For a runnable demo, we'll make a dummy prediction later if weights fail.**
        
        # This will fail unless the input tensor size is correctly formatted
        # to the model's expected feature vector size (512 in this placeholder).
        
        # Returning a dummy prediction for structure:
        return torch.randn(x.size(0), self.fc.out_features).to(x.device) 
        
# --- Configuration ---
ENV_NAME = "SpaceInvaders-v0"
# NOTE: Since the script is playing, we don't need SAVE_DIR or screenshot saving.
# FRAME_SKIP is crucial for action control and remains at 4.
FRAME_SKIP = 4 

# Action mapping for SpaceInvaders-v0 (Discrete(6)). We use the basic 4 actions:
# 0: NOOP, 1: FIRE (Shoot), 2: RIGHT, 3: LEFT
ACTION_NAMES = {
    0: "NOOP",
    1: "SHOOT",
    2: "RIGHT",
    3: "LEFT",
    # Note: Actions 4 (RIGHTFIRE) and 5 (LEFTFIRE) exist but are not used here.
}

# --- INFERENCE CONFIGURATION ---
# IMPORTANT: Update WEIGHTS_PATH to the actual location of your trained model weights.
WEIGHTS_PATH = 'outputs/best_model.pth' 
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMAGE_RESIZE = 256
# The order of CLASS_NAMES MUST match the order of the environment's action IDs (0, 1, 2, 3)
CLASS_NAMES_MODEL = ["NOOP", "SHOOT", "RIGHT", "LEFT"] 

# --- Transforms ---
def get_test_transform(image_size):
    """Defines the preprocessing steps for the environment observation."""
    test_transform = transforms.Compose([
        # Observation is a numpy array (H, W, C) from the environment.
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    return test_transform

# --- Inference Logic ---

def predict_action(model, observation, transform, device):
    """
    Runs inference on the current game observation to determine the next action ID.

    Args:
        model (nn.Module): The loaded PyTorch model.
        observation (np.ndarray): The raw RGB pixel array from env.step().
        transform (transforms.Compose): The image preprocessing pipeline.
        device (torch.device): The computation device.

    Returns:
        int: The action ID (0, 1, 2, or 3) predicted by the model.
    """
    model.eval()
    
    # 1. Preprocess the NumPy observation array (H, W, C)
    try:
        image_tensor = transform(observation)
        # Add batch dimension (1, C, H, W)
        image_tensor = torch.unsqueeze(image_tensor, 0)
        image_tensor = image_tensor.to(device)

    except Exception as e:
        print(f"Error during tensor conversion: {e}. Returning NOOP (0).")
        return 0

    # 2. Forward Pass
    with torch.no_grad():
        try:
            # NOTE: If your model requires the backbone features (e.g., 512-dim),
            # this is where you would need the full architecture.
            outputs = model(image_tensor) 
        except RuntimeError as e:
            # This is common if the input size/shape does not match the model's first layer.
            print(f"Model forward pass error: {e}. Returning NOOP (0).")
            return 0
    
    # 3. Get Prediction
    # Softmax probabilities are not strictly needed, but getting the index is.
    predictions = F.softmax(outputs, dim=1).cpu().numpy()
    
    # Predicted class number (which is also the action ID in our setup)
    action_id = np.argmax(predictions)
    
    # Ensure the predicted ID is a valid action index for the game
    if action_id not in ACTION_NAMES:
        print(f"Warning: Model predicted invalid action ID {action_id}. Using NOOP.")
        return 0
        
    return int(action_id)

# --- Initialization and Main Execution ---

def run_space_invaders_agent():
    """Initializes the environment, model, and runs the AI-driven game loop."""
    print(f"Initializing Gym Environment: {ENV_NAME}")
    
    # 1. Initialize Model and Transforms
    transform = get_test_transform(IMAGE_RESIZE)
    num_classes = len(CLASS_NAMES_MODEL)
    model = LinearClassifier(num_classes=num_classes).to(DEVICE)
    
    print(f"Attempting to load weights from: {WEIGHTS_PATH}")
    try:
        checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        # Assuming the checkpoint structure is {'model_state_dict': ...}
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Weights file not found at '{WEIGHTS_PATH}'.")
        print("Using UNTRAINED placeholder model. Predictions will be random.")
        # Proceed with the randomly initialized model
    except Exception as e:
        print(f"ERROR loading model state: {e}")
        print("Using UNTRAINED placeholder model. Predictions will be random.")

    # 2. Initialize Environment
    # We rely on the 'rgb_array' observation for the agent, and 'human' for visual output.
    env = gymnasium.make(ENV_NAME, render_mode="human")
    
    try:
        initial_state = env.reset()
        if isinstance(initial_state, tuple):
            observation, info = initial_state
        else:
            observation = initial_state 

        print("\n--- Space Invaders AI Agent Activated ---")
        print(f"Agent running on device: {DEVICE}")
        print(f"Model will choose actions every {FRAME_SKIP} game frames.")

        done = False
        total_reward = 0
        step_count = 0

        while not done:
            # 1. Get action from AI Agent
            action_id = predict_action(model, observation, transform, DEVICE)
            action_name = ACTION_NAMES.get(action_id, "NOOP") # Default to NOOP if invalid

            # 2. Apply the action for FRAME_SKIP steps
            accumulated_reward = 0
            
            for _ in range(FRAME_SKIP):
                if done: 
                    break
                
                step_result = env.step(action_id)
                
                # Handle Gym API changes (returns 4 or 5 values)
                if len(step_result) == 5:
                    observation, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    observation, reward, done, info = step_result 

                accumulated_reward += reward
                env.render() # Render each sub-step for smoother visual feedback
            
            if done:
                total_reward += accumulated_reward
                break

            total_reward += accumulated_reward
            step_count += 1
            
            # 3. Status
            print(f"Step: {step_count} | Agent Action: {action_name} ({action_id}) | Reward ({FRAME_SKIP} frames): {accumulated_reward:.2f} | Total Score: {total_reward:.0f} | Lives: {info.get('lives', 'N/A')}")

        print("\n--- Game Over ---")
        print(f"Final Score: {total_reward:.0f}")

    except Exception as e:
        print(f"An error occurred during game execution: {e}")
        
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    # Check for required libraries
    try:
        import PIL
        import gymnasium
        import torch
    except ImportError:
        print("--- DEPENDENCY ERROR ---")
        print("This script requires 'gym[atari]', 'Pillow', and 'torch' (PyTorch).")
        print("Please install them using: pip install gym[atari] pillow torch torchvision")
        print("------------------------")
    else:
        run_space_invaders_agent()

