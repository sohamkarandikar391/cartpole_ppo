import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from actor_critic_networks import Actor, Critic

class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.device = config.device

        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=config.hidden_dims,
            activation = config.activation
        ).to(self.device)

        self.critic = Critic(
            state_dim=state_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation
        ).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = config.critic_lr)

        self.gamma = config.gamma
        self.lambda_gae = config.lambda_gae
        self.clip_epsilon = config.clip_epsilon
        self.entropy_coef = config.entropy_coef
        self.value_loss_coef = config.value_loss_coef
        self.max_grad_norm = config.max_grad_norm
        self.ppo_epochs = config.ppo_epochs
        self.mini_batch_size = config.mini_batch_size
        self.target_kl = config.target_kl

    def select_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            mean, log_std = self.actor(state)
            value = self.critic(state)

        std = log_std.exp()
        distribution = Normal(mean, std)

        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim = -1)

        return action.cpu().detach().numpy()[0], log_prob.cpu().detach().numpy()[0], value.cpu().detach().numpy()[0][0]
    
    def evaluate_actions(self, states, actions):
        mean, log_std = self.actor(states)
        values = self.critic(states)
        std = log_std.exp()
        distribution = Normal(mean, std)
        log_probs = distribution.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = distribution.entropy().sum(dim=-1, keepdim=True)
        return log_probs, values, entropy
    
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0

        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[step]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values[:-1]).to(self.device)

        return advantages, returns
    
    def update(self, states, actions, old_log_probs, rewards, dones, values, next_value):
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device).unsqueeze(1)

        policy_losses = []
        value_losses = []
        entropies = []
        kl_divs = []

        for epoch in range(self.ppo_epochs):
            batch_size = states.size(0)
            indices = np.arange(batch_size)
            np.random.shuffle(indices)

            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                new_log_probs, new_values, entropy = self.evaluate_actions(batch_states, batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages #Normal policy gradient
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages # Clipped gradient
                policy_loss = -torch.min(surr1, surr2).mean() 

                value_loss = 0.5 * (batch_returns - new_values).pow(2).mean()
                entropy_loss = entropy.mean() 
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss
                actor_loss = policy_loss - self.entropy_coef*entropy_loss
                critic_loss = value_loss * self.value_loss_coef

                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())

                with torch.no_grad():
                    kl_div = (batch_old_log_probs - new_log_probs).mean()
                    kl_divs.append(kl_div.item())

            mean_kl = np.mean(kl_divs)
            if mean_kl > self.target_kl:
                print(f"Early stopping at epoch {epoch} due to high KL divergence: {mean_kl:.4f}")
                break
                # stops updating when KL is too high to prevent instability

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'kl_div': np.mean(kl_divs)
        }