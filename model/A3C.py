import torch
import torch.nn as nn
from torch.distributions import Categorical


class A3CNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Shared layers
        self.fc_shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )

        # Actor (策略网络)
        self.fc_actor = nn.Linear(128, action_dim)

        # Critic (值函数网络)
        self.fc_critic = nn.Linear(128, 1)

    def forward(self, state):
        x = self.fc_shared(state)
        policy_logits = self.fc_actor(x)  # 动作概率
        value_estimate = self.fc_critic(x)  # 状态价值
        return policy_logits, value_estimate

    def act(self, state):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()  # 采样动作
        return action.item(), dist.log_prob(action), value


class A3CAgent:
    def __init__(self, state_dim, action_dim):
        self.network = A3CNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)

    def update(self, log_probs, values, rewards, masks, gamma=0.99):
        # 计算Advantage
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        # 计算损失
        actor_loss = []
        critic_loss = []
        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R - value.item()
            actor_loss.append(-log_prob * advantage)  # 策略梯度
            critic_loss.append(nn.functional.mse_loss(value, torch.tensor([R])))  # 值函数损失

        # 反向传播
        total_loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()