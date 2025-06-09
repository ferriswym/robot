import gymnasium as gym  # 使用gymnasium而不是gym
import torch
import numpy as np
from ding.model import DQN
from ding.policy import DQNPolicy
from ding.envs import DingEnvWrapper
from ditk import logging
import matplotlib.pyplot as plt
import time

def run_trained_model(config_path, ckpt_path, render=True, episodes=10):
    # 加载配置文件
    from dizoo.classic_control.cartpole.config.cartpole_dqn_config import main_config, create_config
    from ding.config import compile_config
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    
    # 创建环境 - 使用gymnasium而不是gym
    env = DingEnvWrapper(gym.make("CartPole-v1", render_mode="rgb_array" if render else None))
    
    # 构建模型
    model = DQN(**cfg.policy.model)
    
    # 加载训练好的权重
    state_dict = torch.load(ckpt_path, map_location='cpu')['model']
    model.load_state_dict(state_dict)
    
    # 创建策略
    policy = DQNPolicy(cfg.policy, model=model).eval_mode
    
    # 创建图像窗口用于渲染
    if render:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 4))
        img = ax.imshow(np.zeros((400, 600, 3), dtype=np.uint8))
        ax.axis('off')
        plt.tight_layout()
    
    total_rewards = []
    try:
        for ep in range(episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                if render:
                    # 获取当前帧并显示
                    frame = env.render()
                    img.set_data(frame)
                    plt.pause(0.01)
                
                # 使用模型预测动作
                with torch.no_grad():
                    tensor_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    output = policy.forward({'obs': tensor_obs})
                    action = output["obs"]['action'].item()
                
                # 执行动作
                next_obs, reward, done, truncated = env.step(action)
                episode_reward += reward
                obs = next_obs
                
                # 退出条件
                if done or truncated:
                    logging.info(f"Episode {ep+1}/{episodes} finished! Reward: {episode_reward}")
                    total_rewards.append(episode_reward)
                    break
                
                time.sleep(0.02)  # 控制运行速度
    finally:
        if render:
            plt.ioff()
            plt.show()
    
    logging.info(f"Average reward over {episodes} episodes: {np.mean(total_rewards):.1f}")

# def run_trained_model_headless(config_path, ckpt_path, episodes=10):
#     """无渲染模式运行，仅打印结果"""
#     env = DingEnvWrapper(gym.make("CartPole-v1"))
    
#     # ... (与前面相同的模型加载代码)
    
#     for ep in range(episodes):
#         obs = env.reset()
#         episode_reward = 0
#         done = False
        
#         while not done:
#             with torch.no_grad():
#                 tensor_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
#                 output = policy.forward({'obs': tensor_obs})
#                 action = output['action'].item()
            
#             next_obs, reward, done, truncated, _ = env.step(action)
#             episode_reward += reward
#             obs = next_obs
            
#             if done or truncated:
#                 logging.info(f"Episode {ep+1}/{episodes} finished! Reward: {episode_reward}")
#                 break

if __name__ == "__main__":
    config_path = "dizoo/classic_control/cartpole/config/cartpole_dqn_config.py"
    ckpt_path = "./cartpole_dqn_seed0/ckpt/final.pth.tar"  # 替换为你的模型路径
    run_trained_model(config_path, ckpt_path, False)
    # run_trained_model_headless(config_path, ckpt_path)