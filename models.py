import torch
import torch.nn as nn
import torch.nn.functional as F

class MHCE_Critic(nn.Module):
    """
    多视野集成评论家 (Multi-Horizon Critic Ensemble)
    输出 3 个 Q 值:
    Q1: Short-term (TD biased)
    Q2: Mid-term
    Q3: Long-term (MC biased for HER data)
    """
    def __init__(self, obs_dim, goal_dim, action_dim, hidden_dim=256):
        super().__init__()
        input_dim = obs_dim + goal_dim + action_dim

        self.q1 = self._build_net(input_dim, hidden_dim)
        self.q2 = self._build_net(input_dim, hidden_dim)
        self.q3 = self._build_net(input_dim, hidden_dim)

    def _build_net(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, goal, action):
        x = torch.cat([obs, goal, action], dim=1)
        return self.q1(x), self.q2(x), self.q3(x)
    
    def get_value_ensemble(self, obs, goal, action):
        """用于调度器: 返回 [batch, 3] 的 Tensor 用于计算均值和方差"""
        q1, q2, q3 = self(obs, goal, action)
        return torch.cat([q1, q2, q3], dim=1)

class Actor(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim, action_high, hidden_dim=256):
        super().__init__()
        self.action_high = action_high
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=1)
        x = self.net(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        # 限制 log_std 防止数值不稳定
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def get_action(self, obs, goal):
        mean, log_std = self(obs, goal)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_high
        
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound correction
        log_prob -= torch.log(self.action_high * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob