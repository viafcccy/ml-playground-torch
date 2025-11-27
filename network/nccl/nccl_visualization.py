"""
NCCL 可视化演示
环境: T4 + PyTorch + CUDA 11 + Python 3.9

功能: 用简单的文本可视化展示各种 NCCL 操作的数据流动
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12360'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def print_separator(rank, title=""):
    """打印分隔线"""
    if rank == 0:
        print("\n" + "="*70)
        if title:
            print(f" {title}")
            print("="*70)


def visualize_allreduce(rank, world_size):
    """可视化 All-Reduce"""
    torch.cuda.set_device(rank)
    
    # 初始数据
    data = torch.tensor([rank + 1.0, rank + 1.0]).cuda(rank)
    
    print_separator(rank, "All-Reduce 可视化")
    
    if rank == 0:
        print("\n步骤 1: 初始状态")
        print("┌─────────────────────────────────────┐")
    
    dist.barrier()
    time.sleep(0.1 * rank)  # 错开打印时间
    print(f"│ GPU {rank}: {data.cpu().numpy()}                     │")
    dist.barrier()
    
    if rank == 0:
        print("└─────────────────────────────────────┘")
        print("\n步骤 2: 执行 All-Reduce (SUM)...")
        print("        ↓  所有 GPU 求和  ↓")
    
    dist.barrier()
    
    # All-Reduce
    dist.all_reduce(data, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        print("\n步骤 3: 结果（所有 GPU 相同）")
        print("┌─────────────────────────────────────┐")
    
    dist.barrier()
    time.sleep(0.1 * rank)
    print(f"│ GPU {rank}: {data.cpu().numpy()}                    │")
    dist.barrier()
    
    if rank == 0:
        print("└─────────────────────────────────────┘")


def visualize_broadcast(rank, world_size):
    """可视化 Broadcast"""
    torch.cuda.set_device(rank)
    
    print_separator(rank, "Broadcast 可视化")
    
    if rank == 0:
        data = torch.tensor([100.0, 200.0]).cuda(rank)
        print("\n步骤 1: 初始状态")
        print("┌─────────────────────────────────────┐")
        print(f"│ GPU 0 (源): {data.cpu().numpy()}           │")
    else:
        data = torch.zeros(2).cuda(rank)
    
    dist.barrier()
    time.sleep(0.1 * rank)
    if rank > 0:
        print(f"│ GPU {rank}:       {data.cpu().numpy()}                │")
    dist.barrier()
    
    if rank == 0:
        print("└─────────────────────────────────────┘")
        print("\n步骤 2: 执行 Broadcast from GPU 0...")
        print("        ↓  GPU 0 → 所有 GPU  ↓")
    
    dist.barrier()
    
    # Broadcast
    dist.broadcast(data, src=0)
    
    if rank == 0:
        print("\n步骤 3: 结果")
        print("┌─────────────────────────────────────┐")
    
    dist.barrier()
    time.sleep(0.1 * rank)
    print(f"│ GPU {rank}: {data.cpu().numpy()}           │")
    dist.barrier()
    
    if rank == 0:
        print("└─────────────────────────────────────┘")


def visualize_reduce(rank, world_size):
    """可视化 Reduce"""
    torch.cuda.set_device(rank)
    
    print_separator(rank, "Reduce 可视化")
    
    data = torch.tensor([float(rank + 1), float(rank + 1)]).cuda(rank)
    
    if rank == 0:
        print("\n步骤 1: 初始状态")
        print("┌─────────────────────────────────────┐")
    
    dist.barrier()
    time.sleep(0.1 * rank)
    print(f"│ GPU {rank}: {data.cpu().numpy()}                     │")
    dist.barrier()
    
    if rank == 0:
        print("└─────────────────────────────────────┘")
        print("\n步骤 2: 执行 Reduce to GPU 0 (SUM)...")
        print("        ↓  所有 GPU → GPU 0 求和  ↓")
    
    dist.barrier()
    
    # Reduce
    dist.reduce(data, dst=0, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        print("\n步骤 3: 结果")
        print("┌─────────────────────────────────────┐")
        print(f"│ GPU 0 (目标): {data.cpu().numpy()}            │")
    
    dist.barrier()
    time.sleep(0.1 * rank)
    if rank > 0:
        print(f"│ GPU {rank}:         {data.cpu().numpy()}               │")
    dist.barrier()
    
    if rank == 0:
        print("└─────────────────────────────────────┘")


def visualize_allgather(rank, world_size):
    """可视化 All-Gather"""
    torch.cuda.set_device(rank)
    
    print_separator(rank, "All-Gather 可视化")
    
    data = torch.tensor([float(rank * 10)]).cuda(rank)
    
    if rank == 0:
        print("\n步骤 1: 初始状态（每个 GPU 一个值）")
        print("┌─────────────────────────────────────┐")
    
    dist.barrier()
    time.sleep(0.1 * rank)
    print(f"│ GPU {rank}: [{data.item():.0f}]                          │")
    dist.barrier()
    
    if rank == 0:
        print("└─────────────────────────────────────┘")
        print("\n步骤 2: 执行 All-Gather...")
        print("        ↓  收集所有 GPU 数据  ↓")
    
    dist.barrier()
    
    # All-Gather
    gather_list = [torch.zeros(1).cuda(rank) for _ in range(world_size)]
    dist.all_gather(gather_list, data)
    
    if rank == 0:
        print("\n步骤 3: 结果（所有 GPU 都收集到全部数据）")
        print("┌─────────────────────────────────────┐")
    
    dist.barrier()
    gathered_data = [t.item() for t in gather_list]
    time.sleep(0.1 * rank)
    print(f"│ GPU {rank}: {gathered_data}                │")
    dist.barrier()
    
    if rank == 0:
        print("└─────────────────────────────────────┘")


def visualize_gradient_sync(rank, world_size):
    """可视化 DDP 梯度同步过程"""
    torch.cuda.set_device(rank)
    
    print_separator(rank, "DDP 梯度同步过程可视化")
    
    if rank == 0:
        print("\n在分布式训练中，DDP 自动完成以下步骤：")
        print()
        print("1. 前向传播（各 GPU 独立）")
        print("   GPU 0: loss = 0.5")
        print("   GPU 1: loss = 0.7")
        print()
        print("2. 反向传播（计算梯度，各 GPU 独立）")
        print("   GPU 0: grad = [-0.1, -0.2]")
        print("   GPU 1: grad = [-0.15, -0.25]")
        print()
        print("3. All-Reduce 同步梯度（NCCL 自动完成）")
        print("   ↓  所有 GPU 梯度求平均  ↓")
        print("   GPU 0: grad = [-0.125, -0.225]")
        print("   GPU 1: grad = [-0.125, -0.225]")
        print()
        print("4. 参数更新（各 GPU 独立，但因梯度相同，更新后参数相同）")
        print("   所有 GPU 的模型参数保持同步！")
    
    dist.barrier()


def run_visualization(rank, world_size):
    """运行所有可视化"""
    setup(rank, world_size)
    
    # 各种操作的可视化
    visualize_allreduce(rank, world_size)
    dist.barrier()
    time.sleep(0.5)
    
    visualize_broadcast(rank, world_size)
    dist.barrier()
    time.sleep(0.5)
    
    visualize_reduce(rank, world_size)
    dist.barrier()
    time.sleep(0.5)
    
    visualize_allgather(rank, world_size)
    dist.barrier()
    time.sleep(0.5)
    
    visualize_gradient_sync(rank, world_size)
    dist.barrier()
    
    if rank == 0:
        print("\n" + "="*70)
        print(" 可视化演示完成！")
        print("="*70)
    
    cleanup()


def main():
    world_size = torch.cuda.device_count()
    print(f"检测到 {world_size} 个 GPU")
    
    if world_size < 2:
        print("此示例需要至少 2 个 GPU")
        return
    
    print("\n开始 NCCL 操作可视化演示...")
    print("(各操作之间有 0.5 秒延迟以便观看)\n")
    
    mp.spawn(run_visualization,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()