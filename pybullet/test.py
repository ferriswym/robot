import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data as pd

class LaikagoGymEnv(gym.Env):
    def __init__(self, render=False):
        # 连接物理引擎
        self.physicsClient = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pd.getDataPath())
        self.plane = None
        self.quadruped = None

        # 强化学习参数
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,))  # 12个关节
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,))  # 示例观测维度
        
        # 初始化物理参数
        self.time_step = 1/500
        p.setTimeStep(self.time_step)
        self.max_steps = 1000  # 每个episode最大步数

    def reset(self):
        # 重置环境
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.plane = p.loadURDF("plane.urdf")
        self.quadruped = p.loadURDF("laikago/laikago_toes.urdf", 
                                   [0,0,0.5], 
                                   useFixedBase=False,
                                   flags=p.URDF_USE_SELF_COLLISION)
        
        # 返回初始观测
        return self._get_obs()

    def step(self, action):
        # 应用动作到关节
        self._apply_action(action)
        p.stepSimulation()
        
        # 获取新状态
        obs = self._get_obs()
        reward = self._calculate_reward()
        done = self._check_termination()
        info = {}
        
        return obs, reward, done, info

    def _get_obs(self):
        # 获取关节信息和机身状态
        joint_states = p.getJointStates(self.quadruped, range(12))
        joint_pos = [s[0] for s in joint_states]
        joint_vel = [s[1] for s in joint_states]
        
        base_pos, base_orn = p.getBasePositionAndOrientation(self.quadruped)
        base_euler = p.getEulerFromQuaternion(base_orn)
        
        # 组合观测（示例包含关节、姿态、速度）
        return np.concatenate([
            joint_pos,
            joint_vel,
            base_pos,
            base_euler
        ])

    def _apply_action(self, action):
        # 将动作映射到力矩控制
        for i in range(12):
            p.setJointMotorControl2(
                bodyUniqueId=self.quadruped,
                jointIndex=i,
                controlMode=p.TORQUE_CONTROL,
                force=action[i]*50  # 缩放动作范围到实际力矩
            )

    def _calculate_reward(self):
        # 奖励函数设计（需调试）
        base_pos, base_orientation = p.getBasePositionAndOrientation(self.quadruped)
        linear_vel, _ = p.getBaseVelocity(self.quadruped)
        
        # 前进速度奖励
        forward_reward = linear_vel[0]
        
        # 姿态稳定惩罚
        angles = p.getEulerFromQuaternion(base_orientation)
        orientation_penalty = abs(angles[0]) + abs(angles[1])
        
        # 存活奖励
        survival_reward = 0.1
        
        return forward_reward - 0.2*orientation_penalty + survival_reward
    
    def _check_termination(self):
        # 终止条件检测
        base_pos, _ = p.getBasePositionAndOrientation(self.quadruped)
        return base_pos[2] < 0.2  # 机身高度过低时终止
    
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# 创建并行环境
env = make_vec_env(lambda: LaikagoGymEnv(render=False), n_envs=4)

# 初始化PPO算法
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01,
)

# 训练模型
model.learn(total_timesteps=1_000_000)
model.save("laikago_ppo")