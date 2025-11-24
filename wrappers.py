import gymnasium as gym
import numpy as np

class StagnationPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty_coef=5.0, threshold=0.005):
        super().__init__(env)
        self.penalty_coef = penalty_coef
        self.threshold = threshold 
        self.last_pos = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_pos = obs['observation'][:2] # 假设前两维是 x,y 坐标 (Maze环境)
        # 如果是 Fetch 机械臂，可能需要 obs['observation'][:3]
        
        if self.last_pos is not None:
            displacement = np.linalg.norm(current_pos - self.last_pos)
            
            # GCRL 逻辑: 如果没成功(reward=-1) 且 位移极小 -> 施加额外惩罚
            # 注意: 不同环境判定成功的 key 可能不同，这里以 gymnasium-robotics 为准
            is_success = info.get('is_success', 0.0) == 1.0
            
            if not is_success and displacement < self.threshold:
                reward -= self.penalty_coef
                info['stagnation'] = True
            else:
                info['stagnation'] = False
        
        self.last_pos = current_pos.copy()
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_pos = obs['observation'][:2].copy()
        return obs, info