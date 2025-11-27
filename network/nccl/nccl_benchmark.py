"""
NCCL 示例 4: 性能测试和监控
环境: T4 + PyTorch + CUDA 11 + Python 3.9

功能: 测试不同通信操作的性能，特别是在 T4 的 PCIe 环境下
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def benchmark_allreduce(rank, world_size, sizes, iterations=100):
    """测试 All-Reduce 性能"""
    torch.cuda.set_device(rank)
    results = []
    
    for size in sizes:
        # 创建测试张量
        tensor = torch.randn(size).cuda(rank)
        
        # 预热
        for _ in range(10):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        # 计时
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(iterations):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        torch.cuda.synchronize()
        end = time.time()
        
        elapsed = (end - start) / iterations
        bandwidth = (size * 4 * 2) / elapsed / 1e9  # 4 bytes per float, 2x for send+receive
        
        results.append({
            'size': size,
            'time_ms': elapsed * 1000,
            'bandwidth_GB/s': bandwidth
        })
    
    return results


def benchmark_broadcast(rank, world_size, sizes, iterations=100):
    """测试 Broadcast 性能"""
    torch.cuda.set_device(rank)
    results = []
    
    for size in sizes:
        tensor = torch.randn(size).cuda(rank)
        
        # 预热
        for _ in range(10):
            dist.broadcast(tensor, src=0)
        
        # 计时
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(iterations):
            dist.broadcast(tensor, src=0)
        
        torch.cuda.synchronize()
        end = time.time()
        
        elapsed = (end - start) / iterations
        bandwidth = (size * 4) / elapsed / 1e9
        
        results.append({
            'size': size,
            'time_ms': elapsed * 1000,
            'bandwidth_GB/s': bandwidth
        })
    
    return results


def run_benchmarks(rank, world_size):
    """运行所有性能测试"""
    setup(rank, world_size)
    
    # 测试不同的数据大小 (元素数量)
    sizes = [
        1024,           # 4 KB
        1024 * 1024,    # 4 MB
        10 * 1024 * 1024,  # 40 MB
        50 * 1024 * 1024,  # 200 MB
    ]
    
    if rank == 0:
        print("\n" + "="*70)
        print("NCCL 性能测试 - T4 GPU (PCIe 连接)")
        print("="*70)
    
    dist.barrier()
    
    # 测试 All-Reduce
    if rank == 0:
        print("\n1. All-Reduce 性能测试:")
        print(f"{'数据大小':<15} {'延迟 (ms)':<15} {'带宽 (GB/s)':<15}")
        print("-" * 45)
    
    allreduce_results = benchmark_allreduce(rank, world_size, sizes)
    
    if rank == 0:
        for result in allreduce_results:
            size_str = f"{result['size'] * 4 / 1024 / 1024:.1f} MB"
            print(f"{size_str:<15} {result['time_ms']:<15.3f} {result['bandwidth_GB/s']:<15.2f}")
    
    dist.barrier()
    
    # 测试 Broadcast
    if rank == 0:
        print("\n2. Broadcast 性能测试:")
        print(f"{'数据大小':<15} {'延迟 (ms)':<15} {'带宽 (GB/s)':<15}")
        print("-" * 45)
    
    broadcast_results = benchmark_broadcast(rank, world_size, sizes)
    
    if rank == 0:
        for result in broadcast_results:
            size_str = f"{result['size'] * 4 / 1024 / 1024:.1f} MB"
            print(f"{size_str:<15} {result['time_ms']:<15.3f} {result['bandwidth_GB/s']:<15.2f}")
    
    if rank == 0:
        print("\n" + "="*70)
        print("注意: T4 通过 PCIe 3.0 x16 连接，理论带宽约 16 GB/s")
        print("实际带宽通常低于理论值，这是正常的")
        print("="*70 + "\n")
    
    cleanup()


def test_communication_pattern(rank, world_size):
    """测试通信模式和延迟"""
    iterations = 100
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print("\n" + "="*70)
        print("通信模式测试")
        print("="*70)
    
    # 测试 ping-pong 延迟
    if world_size >= 2:
        tensor = torch.ones(1).cuda(rank)
        
        if rank == 0:
            print("\nPing-Pong 延迟测试 (Rank 0 <-> Rank 1):")
            torch.cuda.synchronize()
            start = time.time()
            
            for _ in range(iterations):
                if rank == 0:
                    dist.send(tensor, dst=1)
                    dist.recv(tensor, src=1)
        
        elif rank == 1:
            for _ in range(iterations):
                dist.recv(tensor, src=0)
                dist.send(tensor, dst=0)
        
        if rank == 0:
            torch.cuda.synchronize()
            end = time.time()
            latency = (end - start) / iterations / 2 * 1000  # 单程延迟，毫秒
            print(f"  平均单程延迟: {latency:.3f} ms")
    
    dist.barrier()
    
    if rank == 0:
        print("="*70 + "\n")
    
    cleanup()


def main():
    world_size = torch.cuda.device_count()
    print(f"检测到 {world_size} 个 GPU")
    
    if world_size < 2:
        print("此示例需要至少 2 个 GPU")
        return
    
    # 运行性能测试
    mp.spawn(run_benchmarks,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
    # 运行延迟测试
    mp.spawn(test_communication_pattern,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()