import gymnasium as gym
import numpy as np
import argparse
import os
from typing import Optional, Type
import warnings
import wandb
import torch as th
from stable_baselines3.common.callbacks import EvalCallback
from wandb.integration.sb3 import WandbCallback

from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ

ALGOS = {
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "tqc": TQC,
    "crossq": CrossQ,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v1",
                        help="环境名称")
    parser.add_argument("--algo", type=str, default="tqc", choices=list(ALGOS.keys()),
                        help="算法名称")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--train-steps", type=int, default=1000,
                        help="训练步数")
    # 添加 GPU 相关参数
    parser.add_argument("--gpu", type=int, default=None,
                        help="指定使用的 GPU ID (默认: None, 使用CPU)")
    parser.add_argument("--cuda", action="store_true",
                        help="是否使用 CUDA")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="评估轮数")
    parser.add_argument("--render", action="store_true",
                        help="是否渲染环境")
    # wandb相关参数
    parser.add_argument("--wandb-project", type=str, default="gen",
                        help="WandB项目名称")
    parser.add_argument("--wandb-entity", type=str, default="rlma",
                        help="WandB实体名称")
    parser.add_argument("--debug", action="store_true",
                        help="调试模式（禁用wandb）")
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="评估频率（步数）")
    parser.add_argument("--save-model", action="store_true", default=True,
                        help="是否保存模型")
    parser.add_argument("--save-path", type=str, default="./models",
                        help="模型保存路径")
    return parser.parse_args()

def set_seed(seed):
    """完整设置随机种子"""
    if seed is not None:
        np.random.seed(seed)
        th.manual_seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed(seed)
            th.cuda.manual_seed_all(seed)
        # 设置确定性
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

def initialize_wandb(args):
    """初始化WandB"""
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.algo}_{args.env}_{args.seed}",
        config=vars(args),
        sync_tensorboard=True,
        save_code=True,
        notes="RL Algorithm Testing",
        mode='online' if not args.debug else 'disabled'
    )
    return run

def evaluate_policy(
    model,
    env: gym.Env,
    n_eval_episodes: int = 10,
    render: bool = False,
) -> tuple[float, float]:
    """评估策略的性能"""
    episode_rewards = []
    episode_lengths = []

    for i in range(n_eval_episodes):
        obs = env.reset()[0]
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            if render:
                env.render()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

def main():
    args = parse_args()

    # 设置设备
    if args.cuda:
        device = f"cuda:{args.gpu}" if args.gpu is not None else "cuda"
    else:
        device = "cpu"
    print(f"使用设备: {device}")

    # 设置随机种子
    set_seed(args.seed)

    # 初始化wandb
    run = initialize_wandb(args)

    # 创建保存目录
    if args.save_model:
        save_path = os.path.join(args.save_path, f"{args.algo}_{args.env}_{args.seed}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = None

    # 创建环境 - 分离训练和评估环境
    env = gym.make(args.env, render_mode=None)  # 训练时不需要渲染
    eval_env = gym.make(args.env, render_mode="human" if args.render else None)

    try:
        # 创建算法实例
        algo_class = ALGOS[args.algo.lower()]
        model = algo_class("MlpPolicy", env, verbose=1, seed=args.seed,
                          tensorboard_log=f"runs/{run.id}",
                          device=device)

        # 设置 WandB 回调，使用统一的保存路径
        wandb_callback = WandbCallback(
            model_save_path=save_path,  # 使用之前创建的统一保存路径
            verbose=2,
        )

        # 训练
        model.learn(total_timesteps=args.train_steps, 
                   callback=wandb_callback,
                   progress_bar=True)

        # 保存最终模型
        if args.save_model:
            final_model_path = os.path.join(save_path, "final_model")
            model.save(final_model_path)
            print(f"最终模型已保存至: {final_model_path}")

        # 最终评估
        mean_reward, std_reward = evaluate_policy(
            model, 
            eval_env,
            n_eval_episodes=args.eval_episodes,
            render=args.render
        )

        print(f"\n评估结果:")
        print(f"平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")

    except Exception as e:
        print(f"训练过程出错: {e}")
        wandb.finish(exit_code=1)
        raise

    finally:
        env.close()
        if 'eval_env' in locals():
            eval_env.close()
        wandb.finish()

if __name__ == "__main__":
    main()