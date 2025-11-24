import numpy as np
import torch

class HerReplayBuffer:
    def __init__(self, env, max_size, device, strategy_ratios):
        self.env = env
        self.max_size = int(max_size)
        self.device = device
        
        # 归一化比例
        total = sum(strategy_ratios.values())
        self.ratios = {k: v / total for k, v in strategy_ratios.items()}
        self.strategy_keys = list(self.ratios.keys())
        self.strategy_probs = list(self.ratios.values())

        self.buffer = []

    def add(self, trajectory):
        """trajectory: list of transitions"""
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(trajectory)

    def sample(self, batch_size):
        states, goals, actions, next_states, rewards, dones = [], [], [], [], [], []
        mc_targets = []
        is_valid_mc = []

        # 1. 选轨迹
        traj_indices = np.random.randint(0, len(self.buffer), size=batch_size)
        # 2. 选策略
        strategies = np.random.choice(self.strategy_keys, size=batch_size, p=self.strategy_probs)

        for idx, mode in zip(traj_indices, strategies):
            traj = self.buffer[idx]
            t = np.random.randint(0, len(traj))
            transition = traj[t]
            
            goal = transition['g']
            steps_taken = 0
            ground_truth_flag = 0.0

            # --- 策略执行 ---
            if mode == 'original':
                goal = transition['g']
                ground_truth_flag = 0.0 # 原始目标通常无法确定真实距离
                
            elif mode == 'future':
                if t < len(traj) - 1:
                    future_t = np.random.randint(t + 1, len(traj))
                    goal = traj[future_t]['ag']
                    steps_taken = future_t - t
                    ground_truth_flag = 1.0 # 确定是真实的物理路径
                else:
                    goal = traj[-1]['ag'] # 退化为 final
                    steps_taken = len(traj) - 1 - t
                    ground_truth_flag = 1.0

            elif mode == 'final':
                goal = traj[-1]['ag']
                steps_taken = len(traj) - 1 - t
                ground_truth_flag = 1.0

            elif mode == 'random':
                # 跨轨迹采样
                rand_idx = np.random.randint(0, len(self.buffer))
                rand_traj = self.buffer[rand_idx]
                rand_step = np.random.randint(0, len(rand_traj))
                goal = rand_traj[rand_step]['ag']
                ground_truth_flag = 0.0 # 不知道怎么走过去，不能用 MC

            # 重算 Reward (GCRL logic: sparse -1)
            # 简单处理: env.compute_reward 返回的是 numpy array
            r = self.env.compute_reward(transition['ag'], goal, {})
            
            # GCRL -1 reward 下，Value = -steps
            mc_val = -1.0 * steps_taken if ground_truth_flag else 0.0

            states.append(transition['obs'])
            goals.append(goal)
            actions.append(transition['action'])
            next_states.append(transition['next_obs'])
            rewards.append(r)
            dones.append(transition['done'])
            mc_targets.append(mc_val)
            is_valid_mc.append(ground_truth_flag)

        batch = {
            "obs": torch.FloatTensor(np.array(states)).to(self.device),
            "g": torch.FloatTensor(np.array(goals)).to(self.device),
            "actions": torch.FloatTensor(np.array(actions)).to(self.device),
            "next_obs": torch.FloatTensor(np.array(next_states)).to(self.device),
            "rewards": torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device),
            "dones": torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device),
            "mc_targets": torch.FloatTensor(np.array(mc_targets)).unsqueeze(1).to(self.device),
            "is_valid_mc": torch.FloatTensor(np.array(is_valid_mc)).unsqueeze(1).to(self.device),
        }
        return batch

    def sample_states(self, batch_size):
        """用于调度器采样 Frontier"""
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        states = []
        for i in indices:
            traj = self.buffer[i]
            t = np.random.randint(0, len(traj))
            states.append(traj[t]['ag'])
        return torch.FloatTensor(np.array(states)).to(self.device)