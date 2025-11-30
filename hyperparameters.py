import torch

class config:
    env_name = 'InvertedPendulum-v5'
    max_episode_steps = 1000
    gamma = 0.99
    lambda_gae = 0.95
    clip_epsilon = 0.2
    ppo_epochs = 10
    mini_batch_size = 64

    actor_lr = 3e-4
    critic_lr = 1e-3

    hidden_dims = [64, 64]
    activation = "relu"

    total_timesteps = 200000
    n_steps = 2048
    batch_size = 2048
    num_envs = 1

    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    target_kl = 0.01

    log_interval = 10
    eval_interval = 500
    eval_episodes = 10
    save_interval = 10000

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = [0, 1, 2]
    eval_seed = 10