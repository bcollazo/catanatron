from dataclasses import dataclass

@dataclass
class PPOHyperparams:
    lr          = 3e-4      # learning rate
    gamma       = 0.99      # discount factor
    gae_lambda  = 0.95      # GAE smoothing (bias vs variance tradeoff)
    clip_eps    = 0.2       # PPO clip range
    value_coef  = 0.5       # how much critic loss contributes
    entropy_coef = 0.01     # encourages exploration
    epochs      = 6      # how many times to train on each rollout
    batch_size  = 64
    max_grad_norm = 0.5
