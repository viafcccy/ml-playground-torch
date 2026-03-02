import os
import random
import time
import numpy as np

"""
RL 抽象

Agent 观察状态 → 选动作 → 环境执行 → 返回新状态+奖励 → Agent 再观察 → ...

四个核心概念
1. 观察空间（Observation Space）——Agent 能看到什么
2. 动作空间（Action Space）——Agent 能做什么
3. 奖励（Reward）——做得好不好
4. 环境（Env）——游戏规则本身
"""


# ===================== 基础设施 =====================

class Discrete:
    def __init__(self, n: int):
        self.n = n

    def sample(self) -> int:
        return random.randint(0, self.n - 1)


# ===================== 环境 =====================

class Env:
    """
    5x5 网格世界，seeker 从 (0,0) 走到 goal (4,4)。
    坐标约定：(row, col)，row 向下增大，col 向右增大。

        col →  0  1  2  3  4
    row ↓
      0       S  .  .  .  .
      1       .  .  .  .  .
      2       .  .  .  .  .
      3       .  .  .  .  .
      4       .  .  .  .  G
    """

    def __init__(self):
        self.seeker, self.goal = (0, 0), (4, 4)
        self.action_space = Discrete(4)          # 4 个动作: 下/左/上/右
        self.observation_space = Discrete(5 * 5) # 25 个状态
        self.info = {"seeker": self.seeker, "goal": self.goal}

    def reset(self):
        self.seeker = (0, 0)
        return self.get_obs()

    def get_obs(self):
        """将 (row, col) 映射到 0~24 的离散状态"""
        return 5 * self.seeker[0] + self.seeker[1]

    def get_reward(self):
        return 1 if self.seeker == self.goal else 0

    def is_done(self):
        return self.seeker == self.goal

    def step(self, action: int) -> tuple:
        row, col = self.seeker

        if action == 0:    # 下：row + 1
            row = min(row + 1, 4)
        elif action == 1:  # 左：col - 1
            col = max(col - 1, 0)
        elif action == 2:  # 上：row - 1
            row = max(row - 1, 0)
        elif action == 3:  # 右：col + 1
            col = min(col + 1, 4)

        self.seeker = (row, col)
        self.info["seeker"] = self.seeker

        return self.get_obs(), self.get_reward(), self.is_done(), self.info

    def render(self):
        os.system("cls" if os.name == "nt" else "clear")

        grid = [["| " for _ in range(5)] + ["|\n"] for _ in range(5)]
        grid[self.goal[0]][self.goal[1]] = "|G"
        grid[self.seeker[0]][self.seeker[1]] = "|S"

        print("".join(["".join(row) for row in grid]))


# ===================== 策略（Q-Table） =====================

class Policy:

    def __init__(self, env):
        """A Policy suggests actions based on the current state.
        We do this by tracking the value of each state-action pair.
        """
        self.state_action_table = [                        # ❶
            [0 for _ in range(env.action_space.n)]
            for _ in range(env.observation_space.n)
        ]
        self.action_space = env.action_space

    def get_action(self, state, explore=True, epsilon=0.1):  # ❷
        """Explore randomly or exploit the best value currently available."""
        if explore and random.uniform(0, 1) < epsilon:      # ❸
            return self.action_space.sample()
        return np.argmax(self.state_action_table[state])     # ❹


# ===================== 仿真 =====================

class Simulation(object):

    def __init__(self, env):
        """Simulates rollouts of an environment, given a policy to follow."""
        self.env = env

    def rollout(self, policy, render=False, explore=True, epsilon=0.1):  # ❶
        """Returns experiences for a policy rollout."""
        experiences = []
        state = self.env.reset()   # ❷
        done = False
        while not done:
            action = policy.get_action(state, explore, epsilon)             # ❸
            next_state, reward, done, info = self.env.step(action)          # ❹
            experiences.append([state, action, reward, next_state])         # ❺
            state = next_state
            if render:             # ❻
                time.sleep(0.05)
                self.env.render()
        return experiences


# ===================== Q-Learning 更新 =====================

def update_policy(policy, experiences, weight=0.1, discount_factor=0.9):
    """Updates a given policy with a list of (state, action, reward, state)
    experiences."""
    for state, action, reward, next_state in experiences:          # ❶ 按顺序遍历所有经验
        next_max = np.max(policy.state_action_table[next_state])   # ❷ 下一个状态中所有动作的最大Q值
        value = policy.state_action_table[state][action]           # ❸ 当前状态-动作的Q值
        new_value = (1 - weight) * value + weight * \
            (reward + discount_factor * next_max)                  # ❹ 加权平均：旧值 + 新估计
        policy.state_action_table[state][action] = new_value       # ❺ 写回Q-Table


def train_policy(env, num_episodes=10000, weight=0.1, discount_factor=0.9):
    """Training a policy by updating it with rollout experiences."""
    policy = Policy(env)
    sim = Simulation(env)
    for _ in range(num_episodes):
        experiences = sim.rollout(policy)                          # ❶ 收集每次游戏的经验
        update_policy(policy, experiences, weight, discount_factor) # ❷ 用经验更新策略
    return policy


# ===================== 主程序 =====================

if __name__ == "__main__":
    environment = Env()

    # --- 1. 未训练的策略：epsilon=1.0 → 完全随机探索 ---
    print("===== 未训练策略（随机行走）=====")
    untrained_policy = Policy(environment)
    sim = Simulation(environment)
    exp = sim.rollout(untrained_policy, render=True, epsilon=1.0)
    print(f"随机策略走了 {len(exp)} 步才到达目标\n")

    # --- 2. 训练策略：10000 个 episode ---
    print("===== 开始训练（10000 episodes）=====")
    trained_policy = train_policy(environment)                     # ❸ 训练并返回策略
    print("训练完成！\n")

    # --- 3. 用训练好的策略跑一次（不探索） ---
    print("===== 训练后策略 =====")
    exp_trained = sim.rollout(trained_policy, render=True, explore=False)
    print(f"训练后策略只走了 {len(exp_trained)} 步到达目标\n")

    # --- 4. 打印训练后的 Q-Table ---
    print("===== 训练后的 Q-Table =====")
    print("状态  |  下      左      上      右")
    print("-" * 45)
    for i, row in enumerate(trained_policy.state_action_table):
        formatted = [f"{v:6.3f}" for v in row]
        print(f"  {i:2d}  | {'  '.join(formatted)}")