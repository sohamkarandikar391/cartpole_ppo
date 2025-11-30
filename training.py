import gymnasium as gym
import numpy as np
import torch
from ppo_agent import PPOAgent
from hyperparameters import config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def set_seed(seed, env):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def evaluate(agent, env, num_episodes=10, seed=None):
    """
    Evaluates the agent for a number of episodes.
    Returns the mean and std of the reward over these episodes.
    """
    if seed is not None:
        set_seed(seed, env)

    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                mean, _ = agent.actor(state_tensor)
                action = mean.cpu().numpy()[0]

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards), np.std(episode_rewards)

def train(seed=0):
    env = gym.make(config.env_name)
    set_seed(seed, env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(state_dim, action_dim, config)
    
    # Storage for plotting
    episode_return_data = [] # (timestep, reward)
    evaluation_results = []  # (timestep, eval_mean, eval_std)
    
    # PPO Buffers
    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []

    episode_reward = 0
    episode_count = 0
    eval_interval_episodes = 100 

    state, _ = env.reset()
    timestep = 0

    pbar = tqdm(total=config.total_timesteps, desc=f"Seed {seed}")

    while timestep < config.total_timesteps:
        for step in range(config.n_steps):
            action, log_prob, value = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)

            state = next_state
            episode_reward += reward
            timestep += 1
            pbar.update(1)
            
            if done:
                episode_return_data.append((timestep, episode_reward))
                episode_count += 1

                if episode_count % config.log_interval == 0:
                    # Optional: Print less frequently to keep console clean
                    pass

                # Evaluate every X episodes
                if episode_count % eval_interval_episodes == 0:
                    eval_mean, eval_std = evaluate(agent, env, config.eval_episodes, seed=config.eval_seed)
                    evaluation_results.append((timestep, eval_mean, eval_std))
                    
                    tqdm.write(f"Episode {episode_count} | Step {timestep} | Eval: {eval_mean:.2f} +/- {eval_std:.2f}")

                state, _ = env.reset()
                episode_reward = 0

            if timestep >= config.total_timesteps:
                break
        
        # PPO Update
        if len(states) >= config.n_steps or timestep >= config.total_timesteps:
            with torch.no_grad():
                if done:
                    next_value = 0
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    next_value = agent.critic(state_tensor).cpu().numpy()[0][0]

            agent.update(
                states=states,
                actions=actions,
                old_log_probs=log_probs,
                rewards=rewards,
                dones=dones,
                values=values,
                next_value=next_value
            )

            # Clear buffers
            states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

    pbar.close()
    
    # Final evaluation
    final_eval_mean, final_eval_std = evaluate(agent, env, config.eval_episodes, seed=config.eval_seed)
    evaluation_results.append((timestep, final_eval_mean, final_eval_std))
    
    env.close()

    return episode_return_data, evaluation_results, agent


def moving_average(data, window):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_results(all_episode_return_data, all_evaluation_results, seeds):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # ---------------------------------------------------------
    # 1. Training Curve Plotting (Smoothed)
    # ---------------------------------------------------------
    max_timestep = max(max(t for t, r in episode_data) for episode_data in all_episode_return_data)
    
    dense_returns = []
    for episode_data in all_episode_return_data:
        returns_at_timestep = np.zeros(max_timestep + 1)
        current_return = 0
        episode_idx = 0
        for t in range(max_timestep + 1):
            while episode_idx < len(episode_data) and episode_data[episode_idx][0] <= t:
                current_return = episode_data[episode_idx][1]
                episode_idx += 1
            returns_at_timestep[t] = current_return
        dense_returns.append(returns_at_timestep)
    
    dense_returns = np.array(dense_returns)
    mean_returns = np.mean(dense_returns, axis=0)
    std_returns = np.std(dense_returns, axis=0)
    
    timestep_window = 1000  # Window for smoothing training data
    smoothed_mean = moving_average(mean_returns, timestep_window)
    smoothed_std = moving_average(std_returns, timestep_window)
    smoothed_timesteps = np.arange(timestep_window - 1, len(smoothed_mean) + timestep_window - 1)
    
    ax.plot(smoothed_timesteps, smoothed_mean, label='Training Reward (Smoothed)', color='blue', alpha=0.8)
    ax.fill_between(smoothed_timesteps, smoothed_mean - smoothed_std, 
                      smoothed_mean + smoothed_std, alpha=0.2, color='blue')
    
    # ---------------------------------------------------------
    # 2. Evaluation Curve Plotting (Interpolated & Averaged)
    # ---------------------------------------------------------
    
    # We need to interpolate evaluation results because different seeds 
    # evaluate at slightly different timesteps (due to varying episode lengths).
    
    # Determine the range for interpolation
    eval_max_ts = 0
    for res in all_evaluation_results:
        if res:
            eval_max_ts = max(eval_max_ts, res[-1][0])
            
    # Create a common X-axis for evaluation (e.g., 100 points across the duration)
    # or match the resolution of the evaluations (approx every 100 eps)
    common_eval_x = np.linspace(0, eval_max_ts, num=100)
    
    interpolated_eval_means = []
    
    for seed_eval_data in all_evaluation_results:
        # Unzip data: (timestep, mean, std) -> separate lists
        # We only strictly need the 'mean' for the main line, 
        # but we are averaging the *means* across seeds.
        ts = [r[0] for r in seed_eval_data] if seed_eval_data else [0]
        means = [r[1] for r in seed_eval_data] if seed_eval_data else [0]
        
        # Interpolate this seed's performance onto the common X grid
        interp_y = np.interp(common_eval_x, ts, means)
        interpolated_eval_means.append(interp_y)
        
    interpolated_eval_means = np.array(interpolated_eval_means)
    
    # Calculate Mean and Std across seeds
    avg_eval_mean = np.mean(interpolated_eval_means, axis=0)
    avg_eval_std = np.std(interpolated_eval_means, axis=0)
    
    ax.plot(common_eval_x, avg_eval_mean, label='Evaluation Reward', color='red', linewidth=2)
    ax.fill_between(common_eval_x, avg_eval_mean - avg_eval_std, 
                      avg_eval_mean + avg_eval_std, alpha=0.3, color='red') 

    # ---------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Reward')
    ax.set_title(f'PPO Learning Curve (Avg over {len(seeds)} seeds)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_curve2.pdf', dpi=300)
    print("Plot saved as 'learning_curve.png'")
    plt.show()


if __name__ == "__main__":
    # Ensure config matches your hyperparameters file
    training_seeds = [0, 1, 2]
    
    all_episode_return_data = []
    all_evaluation_results = []
    all_agents = []

    for seed in training_seeds:
        print(f"--- Starting training for Seed {seed} ---")
        episode_return_data, evaluation_results, agent = train(seed) 
        all_episode_return_data.append(episode_return_data)
        all_evaluation_results.append(evaluation_results)
        all_agents.append(agent)

    plot_results(all_episode_return_data, all_evaluation_results, training_seeds)