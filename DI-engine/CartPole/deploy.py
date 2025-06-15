import gymnasium as gym  # 使用gymnasium而不是gym
import torch
import numpy as np
from ding.model import DQN
from ding.policy import DQNPolicy
from ditk import logging
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
import os

def run_trained_model(config_path, ckpt_path, render=True, episodes=10, save_video=False, video_path='output.mp4'):
    logging.getLogger().setLevel(logging.INFO)
    # 加载配置文件
    from dizoo.classic_control.cartpole.config.cartpole_dqn_config import main_config, create_config
    from ding.config import compile_config
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    
    # 创建环境 - 使用gymnasium而不是gym，不使用DingEnvWrapper
    env = gym.make("CartPole-v1", render_mode="rgb_array" if render or save_video else None)
    
    # 构建模型
    model = DQN(**cfg.policy.model)
    
    # 加载训练好的权重
    state_dict = torch.load(ckpt_path, map_location='cpu')['model']
    model.load_state_dict(state_dict)
    
    # 创建策略
    policy = DQNPolicy(cfg.policy, model=model).eval_mode

    frames = []  # 用于存储视频帧
    total_rewards = []

    if render:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 4))
        img = ax.imshow(np.zeros((400, 600, 3), dtype=np.uint8))
        ax.axis('off')
        plt.tight_layout()

    try:
        for ep in range(episodes):
            obs, _ = env.reset()  # 修改reset返回值接收
            episode_reward = 0
            done = False
            
            while not done:
                frame = env.render() if render or save_video else None
                if frame is not None:
                    if render:
                        img.set_data(frame)
                        plt.pause(0.01)
                    if save_video:
                        frames.append(frame)
                
                # 使用模型预测动作
                with torch.no_grad():
                    tensor_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    output = policy.forward({'obs': tensor_obs})
                    action = output["obs"]["action"].item()  # 修正此处的键名
                
                # 执行动作
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                obs = next_obs
                
                # 退出条件
                if done:
                    logging.info(f"Episode {ep+1}/{episodes} finished! Reward: {episode_reward}")
                    total_rewards.append(episode_reward)
                    break
                
                time.sleep(0.02)  # 控制运行速度

    finally:
        if render:
            plt.ioff()
            plt.show()

        if save_video and frames:
            if not os.path.exists(os.path.dirname(video_path)):
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
            fig, ax = plt.subplots()
            ax.axis('off')
            im = ax.imshow(frames[0])

            def update(frame):
                im.set_data(frame)
                return im,

            ani = FuncAnimation(fig, update, frames=frames, interval=20, blit=True)
            ani.save(video_path, writer='ffmpeg', fps=30)
            plt.close(fig)
            logging.info(f"Video saved to {video_path}")

    logging.info(f"Average reward over {episodes} episodes: {np.mean(total_rewards):.1f}")
    env.close()

if __name__ == "__main__":
    config_path = "dizoo/classic_control/cartpole/config/cartpole_dqn_config.py"
    ckpt_path = "./cartpole_dqn_seed0/ckpt/final.pth.tar"  # 替换为你的模型路径
    # 直接可视化
    run_trained_model(config_path, ckpt_path, render=True, episodes=100)
    # 保存为视频
    # run_trained_model(config_path, ckpt_path, render=False, save_video=True, video_path='./cartpole_video.mp4')