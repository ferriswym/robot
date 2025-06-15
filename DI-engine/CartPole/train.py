"""
# Example of DQN pipeline

Use the pipeline on a single process:

> python3 -u ding/example/dqn.py

Use the pipeline on multiple processes:

We surpose there are N processes (workers) = 1 learner + 1 evaluator + (N-2) collectors

## First Example —— Execute on one machine with multi processes.

Execute 4 processes with 1 learner + 1 evaluator + 2 collectors
Remember to keep them connected by mesh to ensure that they can exchange information with each other.

> ditask --package . --main ding.example.dqn.main --parallel-workers 4 --topology mesh

## Second Example —— Execute on multiple machines.

1. Execute 1 learner + 1 evaluator on one machine.

> ditask --package . --main ding.example.dqn.main --parallel-workers 2 --topology mesh --node-ids 0 --ports 50515

2. Execute 2 collectors on another machine. (Suppose the ip of the first machine is 127.0.0.1).
    Here we use `alone` topology instead of `mesh` because the collectors do not need communicate with each other.
    Remember the `node_ids` cannot be duplicated with the learner, evaluator processes.
    And remember to set the `ports` (should not conflict with others) and `attach_to` parameters.
    The value of the `attach_to` parameter should be obtained from the log of the
    process started earlier (e.g. 'NNG listen on tcp://10.0.0.4:50515').

> ditask --package . --main ding.example.dqn.main --parallel-workers 2 --topology alone --node-ids 2 \
    --ports 50517 --attach-to tcp://10.0.0.4:50515,tcp://127.0.0.1:50516

3. You can repeat step 2 to start more collectors on other machines.
"""
import gym
from ditk import logging
from ding.model import DQN
from ding.policy import DQNPolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, \
    eps_greedy_handler, CkptSaver, ContextExchanger, ModelExchanger, online_logger
from ding.utils import set_pkg_seed
from dizoo.classic_control.cartpole.config.cartpole_dqn_config import main_config, create_config


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True, save_cfg=task.router.node_id == 0)
    print(cfg)
    ding_init(cfg)
    
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        # 创建环境管理器并显式启动它们
        collector_env = BaseEnvManagerV2(
            env_fn=[lambda: DingEnvWrapper(gym.make("CartPole-v1", render_mode="rgb_array", new_step_api=False))  # 修改这里
                    for _ in range(cfg.env.collector_env_num)],
            cfg=cfg.env.manager
        )
        evaluator_env = BaseEnvManagerV2(
            env_fn=[lambda: DingEnvWrapper(gym.make("CartPole-v1", render_mode="rgb_array", new_step_api=False))  # 修改这里
                    for _ in range(cfg.env.evaluator_env_num)],
            cfg=cfg.env.manager
        )
        
        # evaluator_env.enable_save_replay(replay_path=cfg.exp_name + '/video')

        # 关键：显式启动环境管理器
        collector_env.launch()
        evaluator_env.launch()

        # 正确设置全局种子
        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
        
        # 为环境设置种子
        collector_env.seed(cfg.seed)
        evaluator_env.seed(cfg.seed)
        
        # 初始化环境
        collector_env.reset()
        evaluator_env.reset()

        model = DQN(**cfg.policy.model)
        buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
        policy = DQNPolicy(cfg.policy, model=model)

        if task.router.is_active:
            if task.router.node_id == 0:
                task.add_role(task.role.LEARNER)
            elif task.router.node_id == 1:
                task.add_role(task.role.EVALUATOR)
            else:
                task.add_role(task.role.COLLECTOR)

            task.use(ContextExchanger(skip_n_iter=1))
            task.use(ModelExchanger(model))

        # 使用内置的评估中间件
        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        
        task.use(eps_greedy_handler(cfg))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(data_pusher(cfg, buffer_))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        task.use(online_logger(train_show_freq=10))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=100))

        task.run()
        
        # 任务结束后关闭环境管理器
        collector_env.close()
        evaluator_env.close()


if __name__ == "__main__":
    main()