import gymnasium as gym
import torch
import numpy as np
import os
import time
from ppo_agent import PPOAgent
from hyperparameters import config

def run_demonstration(seed=10, num_episodes=10):
    """
    Loads the trained Actor model and plays the environment visually.
    """
    # 1. Setup Environment with Human Rendering
    print(f"--- Loading Environment: {config.env_name} ---")
    env = gym.make(config.env_name, render_mode="human")
    
    # Optional: Set seed for reproducibility
    state, _ = env.reset(seed=config.eval_seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 2. Initialize Agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(state_dim, action_dim, config)
    
    # 3. Load Trained Weights
    # We only need the Actor for demonstration
    actor_path = f"saved_models/ppo_actor_seed_{seed}.pth"
    
    if os.path.exists(actor_path):
        agent.actor.load_state_dict(torch.load(actor_path, map_location=agent.device))
        print(f"Successfully loaded weights from: {actor_path}")
    else:
        print(f"ERROR: Model file not found at {actor_path}")
        print("Please run training.py first to generate the model weights.")
        return

    agent.actor.eval() # Set to evaluation mode (good practice)

    # 4. Run Demonstration Loop
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        print(f"\nRunning Episode {ep + 1}...")
        
        while not done:
            # PPO specific: Prepare state
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            
            with torch.no_grad():
                # Get Deterministic Action (Mean)
                # We do NOT sample during demonstration; we want the best action.
                mean, _ = agent.actor(state_tensor)
                action = mean.cpu().numpy()[0]
            
            # Step Environment
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step_count += 1
            
            # Optional: Slow down visualization if it's too fast
            time.sleep(0.1)

        print(f"Episode {ep + 1} Finished. Steps: {step_count} | Total Reward: {episode_reward:.2f}")

    env.close()
    print("\nDemonstration complete.")

if __name__ == "__main__":
    # You can change the seed here to view different trained agents (0, 1, or 2)
    run_demonstration(seed=0, num_episodes=3)