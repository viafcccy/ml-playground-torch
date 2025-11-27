"""
NCCL 示例 2: 各种集合通信操作
环境: T4 + PyTorch + CUDA 11 + Python 3.9

功能: 演示 Broadcast, All-Gather, Reduce-Scatter 等操作
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def demo_broadcast(rank, world_size):
    """Broadcast: 从 rank 0 广播数据到所有 GPU"""
    torch.cuda.set_device(rank)
    
    if rank == 0:
        tensor = torch.tensor([100.0, 200.0, 300.0]).cuda()
        print(f"[Rank {rank}] 发送数据: {tensor}")
    else:
        tensor = torch.zeros(3).cuda()
        print(f"[Rank {rank}] 初始数据: {tensor}")
    
    # Broadcast from rank 0
    dist.broadcast(tensor, src=0)
    
    print(f"[Rank {rank}] Broadcast 后: {tensor}")
    # 所有 GPU 都应该得到 [100, 200, 300]


def demo_reduce(rank, world_size):
    """Reduce: 将所有 GPU 的数据聚合到 rank 0"""
    torch.cuda.set_device(rank)
    
    # 每个 GPU 创建不同的数据
    tensor = torch.ones(3).cuda() * (rank + 1)
    print(f"[Rank {rank}] 初始数据: {tensor}")
    
    # Reduce to rank 0
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        print(f"[Rank {rank}] Reduce 后 (汇总): {tensor}")
    else:
        print(f"[Rank {rank}] Reduce 后 (未改变): {tensor}")


def demo_all_gather(rank, world_size):
    """All-Gather: 收集所有 GPU 的数据"""
    torch.cuda.set_device(rank)
    
    # 每个 GPU 有不同的数据
    tensor = torch.tensor([rank * 10.0, rank * 10.0 + 1]).cuda()
    print(f"[Rank {rank}] 本地数据: {tensor}")
    
    # 准备接收列表
    gather_list = [torch.zeros(2).cuda() for _ in range(world_size)]
    
    # All-Gather
    dist.all_gather(gather_list, tensor)
    
    print(f"[Rank {rank}] All-Gather 后收集到:")
    for i, t in enumerate(gather_list):
        print(f"  从 Rank {i}: {t}")


def demo_reduce_scatter(rank, world_size):
    """Reduce-Scatter: 聚合后分散"""
    torch.cuda.set_device(rank)
    
    # 每个 GPU 创建一个列表（要分散给各个 GPU）
    input_list = [torch.tensor([float(rank + 1)]).cuda() for _ in range(world_size)]
    output = torch.zeros(1).cuda()
    
    print(f"[Rank {rank}] 输入列表: {[t.item() for t in input_list]}")
    
    # Reduce-Scatter
    dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)
    
    print(f"[Rank {rank}] Reduce-Scatter 后: {output.item()}")
    # 每个 GPU 收到其位置对应的所有 GPU 贡献的和


def run_demo(rank, world_size, demo_func, demo_name):
    """运行单个演示"""
    print(f"\n{'='*60}")
    print(f"演示: {demo_name}")
    print(f"{'='*60}")
    
    setup(rank, world_size)
    demo_func(rank, world_size)
    
    # 同步所有进程
    dist.barrier()
    
    if rank == 0:
        print(f"{'='*60}\n")
    
    cleanup()


def main():
    world_size = torch.cuda.device_count()
    print(f"检测到 {world_size} 个 GPU\n")
    
    if world_size < 2:
        print("此示例需要至少 2 个 GPU")
        return
    
    demos = [
        (demo_broadcast, "Broadcast - 广播"),
        (demo_reduce, "Reduce - 归约"),
        (demo_all_gather, "All-Gather - 全收集"),
        (demo_reduce_scatter, "Reduce-Scatter - 归约分散"),
    ]
    
    for demo_func, demo_name in demos:
        mp.spawn(run_demo,
                 args=(world_size, demo_func, demo_name),
                 nprocs=world_size,
                 join=True)


if __name__ == "__main__":
    main()