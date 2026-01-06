"""
PPO 模型架构检查演示脚本
展示如何查看 PPO 模型如何与 observation/action space 关联
"""

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from loguru import logger


def main() -> None:
    """主函数：展示 PPO 模型架构检查"""
    # 1. 创建环境
    logger.info("创建 LunarLander-v2 环境...")
    env = gym.make("LunarLander-v2")

    # 2. 创建 PPO 模型
    logger.info("创建 PPO 模型...")
    model = PPO("MlpPolicy", env=env, verbose=0)

    # 3. 打印空间信息
    print("=" * 60)
    print("环境空间信息")
    print("=" * 60)
    print(f"Observation Space: {env.observation_space}")
    print(f"Observation Shape: {env.observation_space.shape}")
    print(f"Observation dtype: {env.observation_space.dtype}")
    print(f"Action Space: {env.action_space}")
    print(f"Action n: {env.action_space.n}")

    # 4. 打印策略网络架构
    print("\n" + "=" * 60)
    print("策略网络架构 (Policy Network)")
    print("=" * 60)
    print(model.policy)

    # 5. 打印网络层结构
    print("\n" + "=" * 60)
    print("MlpExtractor 层结构")
    print("=" * 60)
    print(f"输入层维度: {env.observation_space.shape[0]} (对应 observation 的 8 个维度)")
    print(f"\n隐藏层提取器:\n{model.policy.mlp_extractor}")

    # 6. 打印动作输出层
    print("\n" + "=" * 60)
    print("动作输出层 (Action Network)")
    print("=" * 60)
    print(f"输出维度: {env.action_space.n} (对应 4 个离散动作)")
    print(f"\n动作网络:\n{model.policy.action_net}")

    # 7. 打印价值输出层
    print("\n" + "=" * 60)
    print("价值输出层 (Value Network)")
    print("=" * 60)
    print(f"输出维度: 1 (状态价值 V(s))")
    print(f"\n价值网络:\n{model.policy.value_net}")

    # 8. 打印网络参数详情
    print("\n" + "=" * 60)
    print("网络参数详情")
    print("=" * 60)
    total_params = 0
    for name, param in model.policy.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"{name:40s} | Shape: {str(param.shape):20s} | Params: {param_count:6d}")
    print(f"\n总参数数量: {total_params:,}")

    # 9. 前向传播测试
    print("\n" + "=" * 60)
    print("前向传播测试")
    print("=" * 60)
    obs, info = env.reset()
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    print(f"输入观察值形状: {obs_tensor.shape}")

    with torch.no_grad():
        # 获取动作
        action, _ = model.predict(obs, deterministic=True)
        print(f"预测动作: {action}")

        # 获取动作概率分布
        obs_tensor = obs_tensor.to(model.device)
        distribution = model.policy.get_distribution(obs_tensor)
        action_probs = distribution.distribution.probs
        print(f"\n动作概率分布:")
        for i, prob in enumerate(action_probs[0].cpu().numpy()):
            print(f"  Action {i}: {prob:.4f}")

        # 获取状态价值
        values = model.policy.predict_values(obs_tensor)
        print(f"\n状态价值 V(s): {values.item():.4f}")

    # 10. 解释关联机制
    print("\n" + "=" * 60)
    print("PPO 模型与环境空间的关联机制")
    print("=" * 60)
    print("""
关键步骤:
1. 创建 PPO 模型时传入 env 参数
2. PPO 自动从 env.observation_space 读取观察空间信息
3. PPO 自动从 env.action_space 读取动作空间信息
4. 根据空间信息自动构建匹配的神经网络架构:
   - 输入层: observation_space.shape[0] 个节点
   - 输出层: action_space.n 个节点 (动作概率) + 1 个节点 (价值)
5. model.learn() 自动管理所有环境交互 (reset, step, 数据收集)

无需手动指定网络架构，一切由 Stable Baselines3 自动化处理！
    """)

    env.close()
    logger.info("检查完成！")


if __name__ == "__main__":
    main()
