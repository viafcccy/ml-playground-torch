"""
NCCL 示例 1: 基础 All-Reduce 操作
环境: T4 + PyTorch + CUDA 11 + Python 3.9

功能: 演示最基本的多 GPU All-Reduce 操作
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 使用 NCCL 后端（GPU 通信优化）
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def demo_allreduce(rank, world_size):
    """演示 All-Reduce 操作"""
    print(f"[Rank {rank}] 进程启动")
    
    # 初始化分布式环境
    setup(rank, world_size)
    
    # 设置当前进程使用的 GPU
    torch.cuda.set_device(rank)
    
    # 创建一个张量，每个 GPU 上的值不同
    tensor = torch.ones(3) * (rank + 1)
    tensor = tensor.cuda(rank)
    
    print(f"[Rank {rank}] 初始张量: {tensor}")
    
    # All-Reduce: 所有 GPU 的张量求和，结果广播给所有 GPU
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    print(f"[Rank {rank}] All-Reduce 后: {tensor}")
    # 期望结果: 如果有 2 个 GPU，结果应该是 [1+2, 1+2, 1+2] = [3, 3, 3]
    # 如果有 4 个 GPU，结果应该是 [1+2+3+4, ...] = [10, 10, 10]
    
    cleanup()


def main():
    """主函数"""
    world_size = torch.cuda.device_count()
    print(f"检测到 {world_size} 个 GPU")
    
    if world_size < 2:
        print("此示例需要至少 2 个 GPU")
        return
    
    # 使用 spawn 启动多个进程，每个进程对应一个 GPU
    mp.spawn(demo_allreduce,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()