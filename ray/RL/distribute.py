import ray
import base_env

# from base_env import Env, Policy, update_policy


@ray.remote
class SimActor:
    def __init__():
        env = base_env.Env()
        super().__init__(env)


def train_policy_parallel(env, num_episodes=1000, num_simulations=4):
    """Parallel policy training function."""
    # 1. 初始化一个新的策略（Q-Table 全 0）
    policy = base_env.Policy(env)

    # 2. 创建多个并行的仿真 Actor（默认 4 个）
    simulations = [SimActor.remote() for _ in range(num_simulations)]

    # 3. 将策略对象放入 Ray 的共享对象存储，避免重复拷贝
    policy_ref = ray.put(policy)

    # 4. 迭代训练多轮 episode
    for _ in range(num_episodes):
        # 并行调用每个 Actor 的 rollout 方法，收集经验
        experiences = [sim.rollout.remote(policy_ref) for sim in simulations]

        # 5. 分批等待结果，避免阻塞，边收边更新
        while len(experiences) > 0:
            # 等待一批任务完成
            finished, experiences = ray.wait(experiences)
            # 取出完成的经验，更新策略
            for xp in ray.get(finished):
                base_env.update_policy(policy, xp)

    return policy


# ===================== 主程序调用示例 =====================
if __name__ == "__main__":
    # 初始化 Ray
    ray.init()
    
    # 创建环境实例
    environment = base_env.Env()
    
    # 并行训练策略
    parallel_policy = train_policy_parallel(environment)
    
    # 评估训练好的策略（假设 evaluate_policy 已定义）
    # evaluate_policy(environment, parallel_policy)
