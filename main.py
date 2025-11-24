import argparse
import time
import os
import shutil
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # 引入 TensorBoard

# 引入项目模块
from utils import setup_seed, get_device
from models import Actor, MHCE_Critic
from buffer import HerReplayBuffer
from wrappers import StagnationPenaltyWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="VE-DFE: Value-Ensemble Dynamic Frontier Expansion")
    
    # --- 实验管理参数 (新增) ---
    parser.add_argument("--exp_name", type=str, default="init_Ver", help="实验名称，用于生成文件夹")
    parser.add_argument("--save_interval", type=int, default=1000, help="每多少次梯度更新保存一次模型权重")
    
    # --- 1. 环境参数 ---
    parser.add_argument("--env_id", type=str, default="PointMaze_UMaze-v3")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--max_episode_steps", type=int, default=50)
    
    # --- 2. 训练参数 ---
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    
    parser.add_argument("--no-autotune", action="store_true", help="关闭自动熵调节")
    parser.add_argument("--init_alpha", type=float, default=1.0)
    
    parser.add_argument("--update_cycles", type=int, default=40)
    parser.add_argument("--start_steps", type=int, default=2000)
    
    # --- 3. HER 比例 ---
    parser.add_argument("--her_future", type=float, default=0.8)
    parser.add_argument("--her_final", type=float, default=0.0)
    parser.add_argument("--her_random", type=float, default=0.0)
    parser.add_argument("--her_original", type=float, default=0.2)
    
    # --- 4. MHCE & VE-DFE 参数 ---
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--mc_weight", type=float, default=2.0)
    parser.add_argument("--reachable_thr", type=float, default=-20.0)
    parser.add_argument("--frontier_sample_size", type=int, default=128)
    parser.add_argument("--penalty_coef", type=float, default=5.0)
    parser.add_argument("--stagnation_thr", type=float, default=0.005)
    
    return parser.parse_args()

# --- 实验环境设置函数 (新增) ---
def setup_experiment(args):
    """创建结果目录，初始化 TensorBoard，备份代码"""
    # 1. 生成带时间戳的运行名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.exp_name}_{timestamp}"
    
    # 2. 创建目录结构
    base_dir = os.path.join("results", run_name)
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    code_dir = os.path.join(base_dir, "code")
    
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(code_dir, exist_ok=True)
    
    # 3. 备份主要 Python 文件
    files_to_backup = ["main.py", "buffer.py", "models.py", "wrappers.py", "utils.py"]
    for file in files_to_backup:
        if os.path.exists(file):
            shutil.copy(file, code_dir)
            
    # 4. 初始化 TensorBoard Writer
    writer = SummaryWriter(log_dir=base_dir)
    
    print(f"[INFO] 实验结果将保存至: {base_dir}")
    return writer, ckpt_dir

# --- 调度器 (保持不变) ---
def select_task_goal(args, agent, current_obs, external_goal, buffer, device):
    obs_t = torch.FloatTensor(current_obs).unsqueeze(0).to(device)
    g_ext_t = torch.FloatTensor(external_goal).unsqueeze(0).to(device)
    
    with torch.no_grad():
        dummy_action = torch.zeros((1, agent['action_dim'])).to(device)
        q_ensemble = agent['critic'].get_value_ensemble(obs_t, g_ext_t, dummy_action)
    
    mean_q = q_ensemble.mean().item()
    
    if mean_q > args.reachable_thr:
        return external_goal, "EXECUTE"
    
    if len(buffer.buffer) < args.frontier_sample_size:
        return external_goal, "WARMUP"
        
    candidate_goals = buffer.sample_states(args.frontier_sample_size)
    obs_repeat = obs_t.repeat(args.frontier_sample_size, 1)
    
    with torch.no_grad():
        dummy_actions = torch.zeros((args.frontier_sample_size, agent['action_dim'])).to(device)
        q_c_ensemble = agent['critic'].get_value_ensemble(obs_repeat, candidate_goals, dummy_actions)
        
        variances = q_c_ensemble.var(dim=1)
        means = q_c_ensemble.mean(dim=1)

        mask = means > (args.reachable_thr * 2.5) 
        if mask.sum() == 0:
            return external_goal, "FAIL_SAFE"

        scores = variances * mask.float()
        best_idx = torch.argmax(scores)
        frontier_goal = candidate_goals[best_idx].cpu().numpy()
        
    return frontier_goal, "EXPLORE"

# --- 主函数 ---
def main():
    args = parse_args()
    args.autotune = not args.no_autotune
    setup_seed(args.seed)
    device = get_device(args)
    
    # --- 1. 设置实验记录 ---
    writer, ckpt_dir = setup_experiment(args)
    
    # 环境
    env = gym.make(args.env_id, max_episode_steps=args.max_episode_steps)
    # env = StagnationPenaltyWrapper(env, penalty_coef=args.penalty_coef, threshold=args.stagnation_thr)
    
    obs_dim = env.observation_space['observation'].shape[0]
    goal_dim = env.observation_space['desired_goal'].shape[0]
    action_dim = env.action_space.shape[0]
    action_high = float(env.action_space.high[0])

    # 模型
    actor = Actor(obs_dim, goal_dim, action_dim, action_high, args.hidden_dim).to(device)
    critic = MHCE_Critic(obs_dim, goal_dim, action_dim, args.hidden_dim).to(device)
    critic_target = MHCE_Critic(obs_dim, goal_dim, action_dim, args.hidden_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    
    agent = {'actor': actor, 'critic': critic, 'action_dim': action_dim}
    
    actor_optim = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=args.critic_lr)
    
    # Autotune
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = log_alpha.exp().item()
    else:
        alpha = args.init_alpha

    # Buffer
    her_ratios = {
        'future': args.her_future, 'final': args.her_final,
        'random': args.her_random, 'original': args.her_original
    }
    rb = HerReplayBuffer(env, args.buffer_size, device, her_ratios)
    
    # 状态追踪变量
    global_step = 0
    total_updates = 0  # 记录总的梯度更新次数
    episode_num = 0
    
    pbar = tqdm(total=args.total_timesteps)
    
    while global_step < args.total_timesteps:
        obs_dict, _ = env.reset()
        obs = obs_dict['observation']
        ag = obs_dict['achieved_goal']
        g_ext = obs_dict['desired_goal']
        
        # current_goal, mode = select_task_goal(args, agent, obs, g_ext, rb, device)
        current_goal = g_ext
        episode_traj = []
        episode_reward = 0
        
        for step in range(args.max_episode_steps):
            if global_step < args.start_steps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    o_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    g_t = torch.FloatTensor(current_goal).unsqueeze(0).to(device)
                    action, _ = actor.get_action(o_t, g_t)
                    action = action.cpu().numpy()[0]
            
            next_obs_dict, reward, terminated, truncated, info = env.step(action)
            if not info['success']:
                reward -= 1.0
            else:
                reward = 0.0
            # 累积 Reward (注意：这里包含停滞惩罚)
            episode_reward += reward
            
            episode_traj.append({
                'obs': obs, 'ag': ag, 'g': current_goal, 'action': action,
                'reward': reward, 'next_obs': next_obs_dict['observation'],
                'done': terminated or truncated
            })
            
            obs = next_obs_dict['observation']
            ag = next_obs_dict['achieved_goal']
            global_step += 1
            pbar.update(1)
            
            if terminated or truncated:
                break
        
        rb.add(episode_traj)
        episode_num += 1
        
        # --- TensorBoard: 记录 Episode 信息 ---
        # 1. 成功率 (gymnasium-robotics 的 info 中包含 'is_success')
        is_success = info.get('is_success', 0.0)
        writer.add_scalar("Train/Episode_Reward", episode_reward, global_step)
        writer.add_scalar("Train/Success_Rate", is_success, global_step)
        writer.add_scalar("Train/Episode_Length", step + 1, global_step)
        
        # 记录调度器选择的模式比例 (可选，用 text 或者 scalar)
        # 这里不做复杂处理，只打印到控制台或 debug
        
        # --- 训练更新 ---
        if global_step > args.start_steps:
            for _ in range(args.update_cycles):
                batch = rb.sample(args.batch_size)
                total_updates += 1
                
                # 1. Update Alpha
                if args.autotune:
                    with torch.no_grad():
                        _, next_log_prob = actor.get_action(batch['obs'], batch['g'])
                    alpha_loss = (-log_alpha.exp() * (next_log_prob + target_entropy)).mean()
                    alpha_optim.zero_grad()
                    alpha_loss.backward()
                    alpha_optim.step()
                    alpha = log_alpha.exp().item()
                    
                    writer.add_scalar("Loss/Alpha_Loss", alpha_loss.item(), total_updates)
                    writer.add_scalar("Param/Alpha", alpha, total_updates)
                
                # 2. Update Critic
                with torch.no_grad():
                    next_a, next_log_prob_target = actor.get_action(batch['next_obs'], batch['g'])
                    nq1, nq2, nq3 = critic_target(batch['next_obs'], batch['g'], next_a)
                    min_next_q = torch.min(torch.stack([nq1, nq2, nq3]), dim=0)[0]
                    min_next_q -= alpha * next_log_prob_target
                    td_target = batch['rewards'] + args.gamma * (1 - batch['dones']) * min_next_q

                q1, q2, q3 = critic(batch['obs'], batch['g'], batch['actions'])
                loss_q1 = F.mse_loss(q1, td_target)
                loss_q2 = F.mse_loss(q2, td_target)
                
                loss_q3_mc = F.mse_loss(q3, batch['mc_targets'], reduction='none') * batch['is_valid_mc']
                loss_q3_td = F.mse_loss(q3, td_target, reduction='none') * (1 - batch['is_valid_mc'])
                loss_q3 = (loss_q3_mc.sum() * args.mc_weight + loss_q3_td.sum()) / args.batch_size
                
                critic_loss = loss_q1 + loss_q2 + loss_q3
                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()
                
                writer.add_scalar("Loss/Critic_Loss", critic_loss.item(), total_updates)
                writer.add_scalar("Value/Mean_Q1", q1.mean().item(), total_updates)
                
                # 3. Update Actor
                new_a, new_log_prob = actor.get_action(batch['obs'], batch['g'])
                nq1, nq2, nq3 = critic(batch['obs'], batch['g'], new_a)
                min_q = torch.min(torch.stack([nq1, nq2, nq3]), dim=0)[0]
                actor_loss = (alpha * new_log_prob - min_q).mean()
                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()
                
                writer.add_scalar("Loss/Actor_Loss", actor_loss.item(), total_updates)
                
                # 4. Soft Update
                for p, tp in zip(critic.parameters(), critic_target.parameters()):
                    tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)

                # --- 5. 按更新次数保存权重 ---
                if total_updates % args.save_interval == 0:
                    save_path = os.path.join(ckpt_dir, f"model_{total_updates}.pt")
                    torch.save({
                        'actor': actor.state_dict(),
                        'critic': critic.state_dict(),
                        'log_alpha': log_alpha if args.autotune else None,
                        'total_updates': total_updates,
                        'args': args
                    }, save_path)
                    # print(f"[INFO] 模型已保存: {save_path}") # 避免刷屏，注释掉

    # 结束清理
    env.close()
    writer.close()
    print("[INFO] 训练结束。")

if __name__ == "__main__":
    main()