"""
GPU调度层级示例 - PyTorch版本

展示在深度学习框架中如何理解和利用GPU调度的不同层级
"""

import torch
import time
import numpy as np

def demonstrate_grid_level():
    """演示Grid层级 - 整个内核的调度"""
    print("=" * 50)
    print("1. Grid层级演示")
    print("=" * 50)
    
    # 在PyTorch中，每个操作会启动一个或多个CUDA内核
    # 内核的Grid大小由数据大小和硬件自动确定
    
    sizes = [1000, 10000, 100000, 1000000]
    
    for size in sizes:
        x = torch.randn(size, device='cuda')
        y = torch.randn(size, device='cuda')
        
        # 这个操作会启动一个CUDA内核
        start = time.time()
        z = x + y
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"数据大小: {size:>8} | 耗时: {elapsed*1000:.3f}ms")
        print(f"  -> GPU会自动将工作分配到多个Block中")
    
    print()

def demonstrate_block_level():
    """演示Block层级 - 线程块的调度"""
    print("=" * 50)
    print("2. Block层级演示")
    print("=" * 50)
    
    # Block大小会影响性能
    # 典型的Block大小：128, 256, 512线程
    
    size = 1024 * 1024
    x = torch.randn(size, device='cuda')
    
    print("不同的操作需要不同的Block配置:")
    print("- 简单操作（加法）：较大的Block")
    print("- 复杂操作（矩阵乘法）：需要优化Block大小和共享内存")
    print()
    
    # 演示矩阵操作如何利用Block结构
    matrix_sizes = [128, 256, 512, 1024]
    
    for size in matrix_sizes:
        A = torch.randn(size, size, device='cuda')
        B = torch.randn(size, size, device='cuda')
        
        start = time.time()
        C = torch.mm(A, B)  # 矩阵乘法会使用优化的Block配置
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"矩阵大小: {size}x{size} | 耗时: {elapsed*1000:.2f}ms")
        print(f"  -> CUDA使用Tile/Block策略优化内存访问")
    
    print()

def demonstrate_warp_level():
    """演示Warp层级 - 32线程的执行单元"""
    print("=" * 50)
    print("3. Warp层级演示 - 内存合并访问")
    print("=" * 50)
    
    print("Warp是GPU的SIMT执行单元，包含32个线程")
    print("内存访问模式会影响Warp效率:\n")
    
    # 演示合并访问 vs 非合并访问
    size = 1024 * 1024
    
    # 合并访问 - 连续访问
    x = torch.arange(size, device='cuda', dtype=torch.float32)
    start = time.time()
    y = x * 2  # 顺序访问，Warp可以合并内存事务
    torch.cuda.synchronize()
    coalesced_time = time.time() - start
    
    # 非合并访问 - 使用stride访问
    indices = torch.arange(0, size, 32, device='cuda')  # 每32个取一个
    start = time.time()
    y = x[indices] * 2  # 跨步访问，Warp无法有效合并
    torch.cuda.synchronize()
    non_coalesced_time = time.time() - start
    
    print(f"合并访问（连续）:    {coalesced_time*1000:.3f}ms")
    print(f"非合并访问（跨步）:  {non_coalesced_time*1000:.3f}ms")
    print(f"效率差异: {non_coalesced_time/coalesced_time:.1f}x")
    print("\n-> Warp内的线程应该访问连续的内存地址以获得最佳性能")
    print()

def demonstrate_warp_divergence():
    """演示Warp分歧问题"""
    print("=" * 50)
    print("4. Warp分歧演示")
    print("=" * 50)
    
    print("Warp分歧发生在同一Warp内的线程执行不同分支\n")
    
    size = 1024 * 1024
    x = torch.randn(size, device='cuda')
    
    # 有分歧的版本 - 使用mask
    start = time.time()
    mask = x > 0
    y1 = torch.where(mask, x * 2, x * 3)
    torch.cuda.synchronize()
    divergent_time = time.time() - start
    
    # 优化版本 - 使用向量操作
    start = time.time()
    y2 = x * torch.where(x > 0, 
                          torch.tensor(2.0, device='cuda'), 
                          torch.tensor(3.0, device='cuda'))
    torch.cuda.synchronize()
    optimized_time = time.time() - start
    
    print(f"分支操作: {divergent_time*1000:.3f}ms")
    print(f"向量操作: {optimized_time*1000:.3f}ms")
    print("\n-> 避免在Warp内使用条件分支，使用向量化操作")
    print()

def demonstrate_thread_level():
    """演示Thread层级 - 单个线程的执行"""
    print("=" * 50)
    print("5. Thread层级 - 并行度与占用率")
    print("=" * 50)
    
    print("GPU需要足够的并行线程来隐藏内存延迟\n")
    
    # 演示不同并行度的影响
    batch_sizes = [1, 16, 64, 256, 1024]
    hidden_size = 1024
    
    print(f"矩阵运算性能 (输入大小 x {hidden_size}):")
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, hidden_size, device='cuda')
        weight = torch.randn(hidden_size, hidden_size, device='cuda')
        
        # 预热
        for _ in range(3):
            _ = torch.mm(x, weight)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(100):
            y = torch.mm(x, weight)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        ops = batch_size * hidden_size * hidden_size * 2 * 100  # FLOPS
        gflops = ops / elapsed / 1e9
        
        print(f"批次大小: {batch_size:>4} | 性能: {gflops:>6.1f} GFLOPS")
    
    print("\n-> 较大的批次提供更多并行性，提高GPU利用率")
    print()

def demonstrate_memory_hierarchy():
    """演示内存层次结构"""
    print("=" * 50)
    print("6. 内存层次结构")
    print("=" * 50)
    
    print("""
GPU内存层次（从快到慢）:
┌─────────────────────────────────────────┐
│ 1. 寄存器 (Registers)                   │ <- 每个Thread专属，最快
│    - 每个Thread可用                      │
│    - ~100GB/s，<1 cycle延迟              │
├─────────────────────────────────────────┤
│ 2. 共享内存 (Shared Memory)             │ <- Block内共享
│    - Block内的Thread共享                 │
│    - ~1-2TB/s，~20-30 cycles延迟         │
├─────────────────────────────────────────┤
│ 3. L1/L2 缓存                           │ <- 自动管理
│    - 硬件管理的缓存                      │
├─────────────────────────────────────────┤
│ 4. 全局内存 (Global Memory)             │ <- 所有Thread可访问
│    - 所有Thread可以访问                  │
│    - ~200-900GB/s，~200-400 cycles延迟   │
└─────────────────────────────────────────┘
    """)
    
    # 演示不同内存访问模式的性能
    size = 10 * 1024 * 1024  # 10M elements
    
    print("内存访问性能测试:")
    
    # 全局内存访问
    x = torch.randn(size, device='cuda')
    start = time.time()
    y = x * 2
    torch.cuda.synchronize()
    global_time = time.time() - start
    bandwidth = (size * 4 * 2) / global_time / 1e9  # 读写，4bytes per float
    
    print(f"全局内存访问: {global_time*1000:.2f}ms ({bandwidth:.1f} GB/s)")
    print()

def demonstrate_scheduling_summary():
    """总结GPU调度层级"""
    print("=" * 50)
    print("GPU调度层级总结")
    print("=" * 50)
    print("""
┌────────────────────────────────────────────────────┐
│ Grid (网格)                                        │
│ ├─ 由内核启动参数决定                              │
│ ├─ 包含多个Block                                   │
│ └─ Block之间独立执行，不能同步                     │
│                                                    │
│   ┌──────────────────────────────────────────┐   │
│   │ Block (线程块)                           │   │
│   │ ├─ 调度到SM (Streaming Multiprocessor)   │   │
│   │ ├─ Block内可以使用共享内存                │   │
│   │ ├─ Block内可以同步 (__syncthreads())    │   │
│   │ └─ 包含多个Warp                          │   │
│   │                                          │   │
│   │   ┌────────────────────────────────┐    │   │
│   │   │ Warp (线程束)                  │    │   │
│   │   │ ├─ 固定32个Thread               │    │   │
│   │   │ ├─ SIMT执行：同一指令          │    │   │
│   │   │ ├─ 是调度的基本单元            │    │   │
│   │   │ └─ 避免分支分歧               │    │   │
│   │   │                                │    │   │
│   │   │   ┌──────────────────────┐    │    │   │
│   │   │   │ Thread (线程)        │    │    │   │
│   │   │   │ ├─ 独立寄存器        │    │    │   │
│   │   │   │ ├─ 独立程序计数器    │    │    │   │
│   │   │   │ └─ 执行实际计算      │    │    │   │
│   │   │   └──────────────────────┘    │    │   │
│   │   └────────────────────────────────┘    │   │
│   └──────────────────────────────────────────┘   │
└────────────────────────────────────────────────────┘

关键性能考虑:
• 提供足够的并行度来饱和GPU
• 优化内存访问模式（合并访问）
• 避免Warp分歧
• 平衡寄存器、共享内存和占用率
• 使用适当的Block大小（通常128-512线程）
    """)

def main():
    if not torch.cuda.is_available():
        print("错误: 需要CUDA支持的GPU")
        return
    
    device_name = torch.cuda.get_device_name(0)
    print(f"\n使用GPU: {device_name}\n")
    
    demonstrate_grid_level()
    demonstrate_block_level()
    demonstrate_warp_level()
    demonstrate_warp_divergence()
    demonstrate_thread_level()
    demonstrate_memory_hierarchy()
    demonstrate_scheduling_summary()

if __name__ == "__main__":
    main()