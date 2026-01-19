# PPO 模型架构文档

## 概述

本文档解释 PPO 模型如何与 Gymnasium 环境的 observation space 和 action space 关联，以 LunarLander-v2 为例。

---

## 1. 关联机制详解

### 1.1 手动方式 vs 自动方式

在强化学习中，有两种与环境交互的方式：

#### 手动方式 (`luna_lander_v2`)
- **特点**: 开发者显式地管理所有交互
- **步骤**:
  1. 创建环境
  2. 显式访问 `env.observation_space` 和 `env.action_space`
  3. 手动采样动作：`action = env.action_space.sample()`
  4. 手动执行动作：`observation, reward, terminated, truncated, info = env.step(action)`
  5. 手动检查终止条件并重置

#### 自动方式 (`luna_lander_v2_model_train`)
- **特点**: Stable Baselines3 自动管理所有交互
- **关键步骤**: 通过 `env` 参数传递环境到 PPO 模型
- **优势**: 简化代码，自动化训练流程

---

## 2. PPO 模型自动关联机制

当调用 `PPO(policy="MlpPolicy", env=env, ...)` 时，Stable Baselines3 内部执行以下步骤：

### 2.1 读取空间信息

```python
# 从环境对象中读取
observation_space = env.observation_space  # LunarLander-v2: Box(8,)
action_space = env.action_space           # LunarLander-v2: Discrete(4)
```

### 2.2 构建神经网络架构

根据空间信息，`MlpPolicy` 自动构建：

- **输入层**: 根据 `observation_space.shape[0]` 设置输入节点数
- **隐藏层**: 使用默认配置（通常为 64-64）
- **输出层**: 根据 `action_space.n` 设置输出节点数

### 2.3 自动化交互

`model.learn()` 内部自动处理：
- `env.reset()` - 环境重置
- `env.step(action)` - 动作执行
- 数据收集和训练循环

---

## 3. LunarLander-v2 空间详情

### 3.1 观察空间 (Observation Space)

**类型**: `Box(low=-inf, high=inf, shape=(8,), dtype=float32)`

**8个观察维度**:
1. x 坐标（着陆点水平位置）
2. y 坐标（着陆点垂直位置）
3. x 速度（水平速度）
4. y 速度（垂直速度）
5. 角度（着陆器倾斜角度）
6. 角速度（旋转速度）
7. 左腿接触地面（布尔值）
8. 右腿接触地面（布尔值）

### 3.2 动作空间 (Action Space)

**类型**: `Discrete(4)`

**4个离散动作**:
- Action 0: 不做任何操作
- Action 1: 点燃左侧引擎
- Action 2: 点燃主引擎
- Action 3: 点燃右侧引擎

---

## 4. PPO + MlpPolicy 网络架构

### 4.1 ASCII 架构图

以下 ASCII 图展示了 PPO 模型的整体架构，**可在任何环境下正常查看**：

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          PPO 模型架构总览                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌────────────────────┐     ┌────────────────────┐                              │
│  │  LunarLander-v2    │     │  PPO Model         │                              │
│  │  Environment       │     │  Initialization    │                              │
│  ├────────────────────┤     ├────────────────────┤                              │
│  │ Observation Space  │ ──► │ 读取 observation   │                               │
│  │ Box(8,)            │     │ space 信息         │                               │
│  ├────────────────────┤     ├────────────────────┤                              │
│  │ Action Space       │ ──► │ 读取 action space  │                              │
│  │ Discrete(4)        │     │ 信息               │                              │
│  └────────────────────┘     └─────────┬──────────┘                             │
│                                       │                                        │
│                                       ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                      PPO MlpPolicy 神经网络                                │  │
│  ├──────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                          │   │
│  │   输入层 (Input Layer)                                                    │   │
│  │   ┌─────────┐                                                            │   │
│  │   │  8 节点  │  ◄── observation_space.shape[0]                            │   │
│  │   └────┬────┘                                                            │   │
│  │        │                                                                 │   │
│  │        ▼                                                                 │   │
│  │   隐藏层 1 (Hidden Layer 1)                                               │   │
│  │   ┌─────────┐                                                            │   │
│  │   │ 64 节点  │  ReLU 激活函数                                              │   │
│  │   └────┬────┘                                                            │   │
│  │        │                                                                 │   │
│  │        ▼                                                                 │   │
│  │   隐藏层 2 (Hidden Layer 2)                                               │   │
│  │   ┌─────────┐                                                            │   │
│  │   │ 64 节点  │  ReLU 激活函数                                              │   │
│  │   └────┬────┘                                                            │   │
│  │        │                                                                 │   │
│  │        ├──────────────────┬─────────────────────────────┐                │   │
│  │        │                  │                             │                │   │
│  │        ▼                  ▼                             ▼                │   │
│  │   ┌─────────┐       ┌─────────┐                   ┌─────────┐            │   │
│  │   │ 4 节点   │       │  1 节点  │                  │ 状态     │            │   │
│  │   │ Softmax │       │  线性    │                   │ 价值    │            │   │
│  │   └────┬────┘       └────┬────┘                   │ V(s)    │            │   │
│  │        │                 │                        │ 用于     │            │   │
│  │        ▼                 │                        │ A(s,a)  │            │    │
│  │   动作概率分布             │                        │ 计算     │            │    │
│  │   [0.25, 0.25,           │                        └─────────┘             │   │
│  │    0.25, 0.25]           │                                                │   │
│  │        │                 │                                                │   │
│  │        ▼                 │                                                │   │
│  │   选择动作                │                                                 │   │
│  │   (argmax 或采样)         │                                                 │   │
│  │        │                 │                                                 │   │
│  │        └────────┬────────┘                                                 │   │
│  │                 │                                                          │   │
│  └─────────────────│──────────────────────────────────────────────────────────┘   │
│                    │                                                              │
│                    ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐     │
│  │                      自动训练流程                                          │     │
│  ├──────────────────────────────────────────────────────────────────────────┤     │
│  │                                                                          │     │
│  │   model.learn(total_timesteps=1000000)                                   │     │
│  │        │                                                                 │     │
│  │        ├────────────────────────────────────────────────────────────┐    │     │
│  │        │                                                            │    │     │
│  │        ▼                                                            ▼    │     │
│  │   ┌───────────┐                                               收集经验数据 │     │
│  │   │env.reset()│  ◄── 初始观察                                        │    │     │
│  │   └─────┬─────┘                                                     │    │     │
│  │         │                                                           │    │     │
│  │         ▼                                                           │    │     │
│  │   ┌───────────┐                                                     │    │     │
│  │   │env.step() │  ◄── 执行动作，获取 (obs, reward, done, ...)          │    │     │
│  │   └─────┬─────┘                                                     │    │     │
│  │         │                                                           │    │     │
│  │         │                   ┌─────────────────────────────┐         │    │     │
│  │         │                   │  策略更新 (Policy Update)     │        │    │     │
│  │         │                   │  - 计算优势函数 A(s,a)         │───────│     │     │
│  │         │                   │  - 更新 Actor 网络参数         │       │     │     │
│  │         │                   │  - 更新 Critic 网络参数       │        │     │     │
│  │         │                   └─────────────────────────────┘        │     │     │
│  │         │                                                     │    │     │     │
│  │         └──────────────────────────────────────────────────────────┘     │     │
│  │                                                               │          │     │
│  │                                                               ▼          │     │
│  │                                                         循环继续          │     │
│  │                                                               │          │     │
│  └───────────────────────────────────────────────────────────────┼──────────┘     │
│                                                           达到总步数                │
│                                                                   │               │
│                                                                   ▼               │
│                                                              训练完成              │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 简化流程图

更简洁的流程表示：

```
┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│   环境空间      │       │   PPO 模型     │      │   神经网络       │
├────────────────┤      ├────────────────┤      ├────────────────┤
│ Observation    │ ──►  │ 自动读取并构建   │ ──►  │ 输入层: 8 节点   │
│ Space: Box(8,) │      │ 匹配的网络架构   │      │ 隐藏层: 64×2    │
├────────────────┤      ├────────────────┤      │ 输出层: 4+1     │
│ Action Space   │ ──►  │ env 参数传递    │      │ (动作 + 价值)   │
│ Discrete(4)    │      │                │      └────────────────┘
└────────────────┘      └───────┬────────┘              │
                                │                       │
                                ▼                       ▼
                        ┌─────────────────────────────────────┐
                        │        model.learn() 自动训练        │
                        │  - 自动 reset/step                   │
                        │  - 自动数据收集                       │
                        │  - 自动策略更新                       │
                        └─────────────────────────────────────┘
```

### 4.3 Mermaid 架构图

> **注意**: 以下图表在支持 Mermaid 的环境中可以自动渲染。如果无法看到图表，请使用上面的 ASCII 图。

#### 整体架构图

```mermaid
graph TB
    subgraph ENV["LunarLander-v2 环境"]
        OBS[Observation Space<br/>Box(8,)] --> ACT[Action Space<br/>Discrete(4)]
    end
    
    subgraph INIT["PPO 模型初始化"]
        ACT --> CREATE[传入 env 参数]
    end
    
    subgraph NN["PPO MlpPolicy 神经网络"]
        INPUT[输入层<br/>8 节点] --> H1[隐藏层 1<br/>64 节点<br/>ReLU]
        H1 --> H2[隐藏层 2<br/>64 节点<br/>ReLU]
        H2 --> ACT_OUT[动作输出<br/>4 节点<br/>Softmax]
        H2 --> VAL_OUT[价值输出<br/>1 节点]
    end
    
    subgraph TRAIN["自动训练流程"]
        LEARN[model.learn] --> RESET[env.reset]
        RESET --> COLLECT[收集经验]
        COLLECT --> STEP[env.step]
        STEP --> UPDATE[更新网络]
        UPDATE -->|循环| COLLECT
    end
    
    CREATE --> INPUT
    ACT_OUT -->|动作概率| SELECT[选择动作]
    VAL_OUT -->|状态价值| ADV[优势计算]
    SELECT --> STEP
    ADV --> UPDATE
    
    style OBS fill:#e1f5ff
    style ACT fill:#fff4e1
    style ACT_OUT fill:#ffe1e1
    style VAL_OUT fill:#e1ffe1
    style LEARN fill:#f0e1ff
```

#### 数据流向序列图

```mermaid
sequenceDiagram
    participant E as Environment
    participant M as PPO Model
    participant A as Actor
    participant C as Critic
    
    E->>M: env.reset() 获取初始观察
    M->>A: 前向传播计算动作概率
    A-->>M: 返回动作概率分布
    M->>M: 根据概率选择动作
    M->>E: env.step(action) 执行动作
    E->>E: 环境模拟
    E-->>M: 返回 (obs, reward, done, info)
    M->>C: 前向传播计算状态价值
    C-->>M: 返回 V(s)
    M->>M: 存储经验
    M->>M: 累积多个步骤
    M->>A: 更新 Actor 参数
    M->>C: 更新 Critic 参数
    M->>E: 如果 done，调用 env.reset()
    
    Note over M,A,C: 上述循环在 model.learn() 中自动进行
```

### 4.4 网络层详解

#### 输入层
- **维度**: 8
- **功能**: 接收 8 维观察向量
- **对应**: LunarLander 的 8 个状态变量

#### 隐藏层 1
- **维度**: 64
- **激活函数**: ReLU
- **功能**: 提取初级特征

#### 隐藏层 2
- **维度**: 64
- **激活函数**: ReLU
- **功能**: 提取高级特征

#### 动作输出层
- **维度**: 4
- **激活函数**: Softmax
- **功能**: 输出 4 个动作的概率分布
- **对应**: LunarLander 的 4 个离散动作

#### 价值输出层
- **维度**: 1
- **激活函数**: 无（线性）
- **功能**: 估计当前状态的价值 V(s)
- **用途**: 用于计算优势函数 A(s,a)

---

## 5. 如何检查模型架构

### 5.1 运行演示脚本

我们提供了一个专门的演示脚本来检查 PPO 模型架构：

```bash
python scripts/inspect_ppo_model.py
```

### 5.2 手动检查方法

在 Python 代码中，你可以通过以下方式检查模型架构：

```python
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("LunarLander-v2")
model = PPO("MlpPolicy", env=env, verbose=0)

# 查看完整策略
print(model.policy)

# 查看观察和动作空间
print(f"Observation Space: {env.observation_space}")
print(f"Action Space: {env.action_space}")

# 查看隐藏层提取器
print(model.policy.mlp_extractor)

# 查看动作网络
print(model.policy.action_net)

# 查看所有参数
for name, param in model.policy.named_parameters():
    print(f"{name}: {param.shape}")
```

---

## 6. 代码对比

### 6.1 手动方式代码示例

```python
import gymnasium as gym

env = gym.make("LunarLander-v2")
observation, info = env.reset()

# 显式访问空间信息
print(f"Observation Space: {env.observation_space.shape}")  # (8,)
print(f"Action Space: {env.action_space.n}")                # 4

# 手动采样和执行
for _ in range(20):
    action = env.action_space.sample()  # 手动采样
    observation, reward, terminated, truncated, info = env.step(action)  # 手动执行
    
    if terminated or truncated:
        observation, info = env.reset()  # 手动重置

env.close()
```

### 6.2 自动方式代码示例

```python
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("LunarLander-v2")

# 只需传递 env，一切自动处理
model = PPO(
    policy="MlpPolicy",
    env=env,  # ⚠️ 关键：传入环境
    n_steps=1024,
    batch_size=64,
)

# 自动读取空间信息、构建网络、训练
model.learn(total_timesteps=1000000)

env.close()
```

---

## 7. 常见问题

### Q1: 为什么不需要显式指定输入输出维度？
**A**: 因为 PPO 从 `env` 参数自动读取 `observation_space` 和 `action_space`，并根据空间信息自动构建网络。

### Q2: 如何自定义网络架构？
**A**: 可以在 `policy_kwargs` 中指定自定义网络结构：
```python
model = PPO(
    "MlpPolicy",
    env=env,
    policy_kwargs={
        "net_arch": [128, 128, 64]  # 自定义隐藏层
    }
)
```

### Q3: MlpPolicy 和 CnnPolicy 有什么区别？
**A**: 
- `MlpPolicy`: 用于向量观察（如 Box 空间）
- `CnnPolicy`: 用于图像观察（需要 CNN 提取特征）

### Q4: 如何查看训练过程中网络的参数变化？
**A**: 可以使用 TensorBoard：
```python
model = PPO(..., tensorboard_log="./ppo_lunar_tensorboard/")
model.learn(total_timesteps=1000000)
# 然后运行: tensorboard --logdir ./ppo_lunar_tensorboard/
```

---

## 8. 相关资源

- [Stable Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [Gymnasium 文档](https://gymnasium.farama.org/)
- [LunarLander-v2 说明](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Mermaid 语法文档](https://mermaid.js.org/intro/)

---

## 9. 总结

PPO 模型与环境的关联是通过 `env` 参数自动完成的：

1. **初始化阶段**: 从 `env` 读取 `observation_space` 和 `action_space`
2. **构建阶段**: 根据空间信息自动构建匹配的神经网络
3. **训练阶段**: 自动管理所有环境交互（reset, step, 数据收集）

这种设计使得开发者无需关注底层细节，可以专注于算法和超参数的调优。

---

## 附录：图表渲染说明

### 支持 Mermaid 的平台
以下平台可以自动渲染本文档中的 Mermaid 图表：
- **GitHub**: 直接在 README 或 .md 文件中查看
- **GitLab**: 直接在 MR/issue 中查看
- **VS Code**: 安装 "Markdown Preview Mermaid Support" 扩展
- **Notion**: 原生支持 Mermaid
- **Typora**: 原生支持 Mermaid
- **Docusaurus**: 支持 Mermaid
- **MkDocs with mermaid2 plugin**: 支持 Mermaid

### 不支持 Mermaid 的平台
如果上面的 Mermaid 图表无法显示，请使用：
1. **ASCII 架构图**（本文档第 4.1 节）
2. **简化流程图**（本文档第 4.2 节）
3. 将文档复制到支持 Mermaid 的平台查看

### 本地渲染 Mermaid
如果需要在本地渲染 Mermaid 图表：
```bash
# 安装 mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# 渲染图表
mmdc -i docs/model-architecture.md -o docs/model-architecture.png
```

---

## 10. 快速参考

### 10.1 架构速览

```
LunarLander-v2 环境                    PPO 模型
┌─────────────────────┐               ┌─────────────────┐
│ Observation: Box(8) │ ───────────►  │ 输入层: 8 节点    │
│ Action: Discrete(4) │ ───────────►  │ 隐藏层: 64×2     │
└─────────────────────┘               │ 输出层: 4+1      │
                                      └─────────────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │ model.learn()   │
                                     │ 自动训练流程      │
                                     └─────────────────┘
```

### 10.2 关键代码

```python
# 创建环境
env = gym.make("LunarLander-v2")

# 创建模型（自动关联空间信息）
model = PPO("MlpPolicy", env=env)

# 训练（自动管理交互）
model.learn(total_timesteps=1000000)
```

### 10.3 网络参数统计

| 层类型 | 维度 | 参数量 |
|--------|------|--------|
| 输入层 → 隐藏层1 | 8 → 64 | 512 + 64 = 576 |
| 隐藏层1 → 隐藏层2 | 64 → 64 | 4,096 + 64 = 4,160 |
| 隐藏层2 → 动作输出 | 64 → 4 | 256 + 4 = 260 |
| 隐藏层2 → 价值输出 | 64 → 1 | 64 + 1 = 65 |
| **总计** | - | **9,797** |

### 10.4 查看命令

```bash
# 运行演示脚本
python scripts/inspect_ppo_model.py

# 查看文档
cat docs/model-architecture.md
```

---

## 变更日志

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| 1.0 | 2026-01-05 | 初始版本，包含 ASCII 和 Mermaid 双版本图表 |
| 1.1 | 2026-01-05 | 添加快速参考章节和图表渲染说明 |
