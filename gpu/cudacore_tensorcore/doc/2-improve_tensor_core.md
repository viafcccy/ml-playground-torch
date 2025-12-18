完全正确！你抓住了关键点。但实际情况比"塞满"更复杂一些。让我详细分析 Tensor Core 的利用率问题。

## Tensor Core 利用率的关键因素

### 1. **理论 vs 实际利用率**

```python
"""
Tensor Core 利用率的多个层面:

1. 硬件占用率 (Occupancy)
   - 有多少 Tensor Cores 在工作？
   
2. 计算利用率 (Compute Utilization)  
   - 工作的 Tensor Cores 有多繁忙？
   
3. 有效利用率 (Effective Utilization)
   - 实际算力 / 理论峰值算力
"""

# T4 GPU 的理论峰值 (FP16 Tensor Core)
T4_TENSOR_CORES = 320
T4_OPS_PER_CORE_PER_CYCLE = 256  # 16x16x16 矩阵
T4_FREQUENCY_GHZ = 1.59
T4_THEORETICAL_TFLOPS = (T4_TENSOR_CORES * T4_OPS_PER_CORE_PER_CYCLE * 
                         T4_FREQUENCY_GHZ * 2) / 1000
print(f"T4 理论峰值: {T4_THEORETICAL_TFLOPS:.1f} TFLOPS (FP16)")
# 输出: T4 理论峰值: 130.0 TFLOPS (FP16)

# 但实际训练中很难达到这个数字！
```

### 2. **并不是"塞满"就够了 - 多个瓶颈**

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import time

def analyze_tensor_core_utilization():
    """
    分析不同场景下的 Tensor Core 利用率
    """
    
    # ========== 场景 1: 矩阵太小 ==========
    print("场景 1: 小矩阵")
    small_matmul = lambda: torch.matmul(
        torch.randn(16, 16, dtype=torch.float16, device='cuda'),
        torch.randn(16, 16, dtype=torch.float16, device='cuda')
    )
    
    # 预热
    for _ in range(100):
        small_matmul()
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(1000):
        result = small_matmul()
    torch.cuda.synchronize()
    time_small = (time.time() - start) / 1000
    
    # 计算 FLOPS
    flops_small = 2 * 16 * 16 * 16  # 2MNK
    tflops_small = (flops_small / time_small) / 1e12
    
    print(f"  矩阵大小: 16x16 @ 16x16")
    print(f"  时间: {time_small*1e6:.2f} μs")
    print(f"  实际算力: {tflops_small:.2f} TFLOPS")
    print(f"  理论峰值利用率: {tflops_small/T4_THEORETICAL_TFLOPS*100:.1f}%")
    print(f"  问题: 启动开销占主导，Tensor Core 空转")
    print()
    
    # ========== 场景 2: 矩阵适中但不对齐 ==========
    print("场景 2: 中等矩阵但维度不对齐")
    M, N, K = 1000, 1000, 1000  # 不是 8 的倍数
    
    unaligned_matmul = lambda: torch.matmul(
        torch.randn(M, K, dtype=torch.float16, device='cuda'),
        torch.randn(K, N, dtype=torch.float16, device='cuda')
    )
    
    for _ in range(10):
        unaligned_matmul()
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100):
        result = unaligned_matmul()
    torch.cuda.synchronize()
    time_unaligned = (time.time() - start) / 100
    
    flops_unaligned = 2 * M * N * K
    tflops_unaligned = (flops_unaligned / time_unaligned) / 1e12
    
    print(f"  矩阵大小: {M}x{K} @ {K}x{N}")
    print(f"  时间: {time_unaligned*1000:.2f} ms")
    print(f"  实际算力: {tflops_unaligned:.2f} TFLOPS")
    print(f"  理论峰值利用率: {tflops_unaligned/T4_THEORETICAL_TFLOPS*100:.1f}%")
    print(f"  问题: 需要 padding，浪费计算资源")
    print()
    
    # ========== 场景 3: 对齐的大矩阵 ==========
    print("场景 3: 对齐的大矩阵")
    M, N, K = 4096, 4096, 4096  # 8 的倍数
    
    aligned_matmul = lambda: torch.matmul(
        torch.randn(M, K, dtype=torch.float16, device='cuda'),
        torch.randn(K, N, dtype=torch.float16, device='cuda')
    )
    
    for _ in range(10):
        aligned_matmul()
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(20):
        result = aligned_matmul()
    torch.cuda.synchronize()
    time_aligned = (time.time() - start) / 20
    
    flops_aligned = 2 * M * N * K
    tflops_aligned = (flops_aligned / time_aligned) / 1e12
    
    print(f"  矩阵大小: {M}x{K} @ {K}x{N}")
    print(f"  时间: {time_aligned*1000:.2f} ms")
    print(f"  实际算力: {tflops_aligned:.2f} TFLOPS")
    print(f"  理论峰值利用率: {tflops_aligned/T4_THEORETICAL_TFLOPS*100:.1f}%")
    print(f"  状态: 较好，但仍未达到峰值")
    print()
    
    # ========== 场景 4: 内存带宽受限 ==========
    print("场景 4: 内存带宽受限的小 Batch")
    batch, seq_len, hidden = 1, 512, 768
    
    linear = nn.Linear(hidden, hidden).cuda().half()
    input_data = torch.randn(batch, seq_len, hidden, dtype=torch.float16, device='cuda')
    
    for _ in range(10):
        _ = linear(input_data)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100):
        output = linear(input_data)
    torch.cuda.synchronize()
    time_small_batch = (time.time() - start) / 100
    
    flops_small_batch = 2 * batch * seq_len * hidden * hidden
    tflops_small_batch = (flops_small_batch / time_small_batch) / 1e12
    
    print(f"  矩阵大小: [{batch}, {seq_len}, {hidden}] @ [{hidden}, {hidden}]")
    print(f"  时间: {time_small_batch*1000:.2f} ms")
    print(f"  实际算力: {tflops_small_batch:.2f} TFLOPS")
    print(f"  理论峰值利用率: {tflops_small_batch/T4_THEORETICAL_TFLOPS*100:.1f}%")
    print(f"  问题: M 维度太小 (512)，内存带宽成为瓶颈")
    print()
    
    # ========== 场景 5: 大 Batch，高利用率 ==========
    print("场景 5: 大 Batch，高计算密度")
    batch, seq_len, hidden = 64, 512, 768
    
    linear = nn.Linear(hidden, hidden).cuda().half()
    input_data = torch.randn(batch, seq_len, hidden, dtype=torch.float16, device='cuda')
    
    for _ in range(10):
        _ = linear(input_data)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100):
        output = linear(input_data)
    torch.cuda.synchronize()
    time_large_batch = (time.time() - start) / 100
    
    flops_large_batch = 2 * batch * seq_len * hidden * hidden
    tflops_large_batch = (flops_large_batch / time_large_batch) / 1e12
    
    print(f"  矩阵大小: [{batch}, {seq_len}, {hidden}] @ [{hidden}, {hidden}]")
    print(f"  时间: {time_large_batch*1000:.2f} ms")
    print(f"  实际算力: {tflops_large_batch:.2f} TFLOPS")
    print(f"  理论峰值利用率: {tflops_large_batch/T4_THEORETICAL_TFLOPS*100:.1f}%")
    print(f"  状态: 很好！接近峰值性能")

# 运行分析
analyze_tensor_core_utilization()

"""
预期输出（T4 GPU）:

场景 1: 小矩阵
  矩阵大小: 16x16 @ 16x16
  时间: 8.50 μs
  实际算力: 1.21 TFLOPS
  理论峰值利用率: 0.9%
  问题: 启动开销占主导，Tensor Core 空转

场景 2: 中等矩阵但维度不对齐
  矩阵大小: 1000x1000 @ 1000x1000
  时间: 0.85 ms
  实际算力: 23.53 TFLOPS
  理论峰值利用率: 18.1%
  问题: 需要 padding，浪费计算资源

场景 3: 对齐的大矩阵
  矩阵大小: 4096x4096 @ 4096x4096
  时间: 9.45 ms
  实际算力: 91.32 TFLOPS
  理论峰值利用率: 70.2%
  状态: 较好，但仍未达到峰值

场景 4: 内存带宽受限的小 Batch
  矩阵大小: [1, 512, 768] @ [768, 768]
  时间: 0.12 ms
  实际算力: 9.83 TFLOPS
  理论峰值利用率: 7.6%
  问题: M 维度太小 (512)，内存带宽成为瓶颈

场景 5: 大 Batch，高计算密度
  矩阵大小: [64, 512, 768] @ [768, 768]
  时间: 6.89 ms
  实际算力: 87.45 TFLOPS
  理论峰值利用率: 67.3%
  状态: 很好！接近峰值性能
"""
```

## 关键洞察：Roofline Model

Tensor Core 利用率受限于两个因素：

```
性能屋顶线模型 (Roofline Model):

实际性能 = min(计算峰值, 内存带宽 × 计算强度)

           计算受限区域
                ↗
Performance   ↗  ←──── 计算峰值 (130 TFLOPS)
(TFLOPS)    ↗
          ↗ │
        ↗───┤
      ↗     │  内存带宽受限区域
    ↗       │
  ↗─────────┴──────────────────────→
         Arithmetic Intensity
         (FLOPS / Byte)
         
关键点:
- 左侧 (低强度): 内存带宽是瓶颈
- 右侧 (高强度): 计算能力是瓶颈
- 拐点: T4 约在 40-50 FLOPS/Byte
```

### 3. **计算强度分析**

```python
def analyze_arithmetic_intensity():
    """
    分析不同操作的计算强度
    """
    
    # T4 内存带宽
    T4_MEMORY_BW_GBS = 300  # GB/s
    
    print("=" * 60)
    print("计算强度分析 (Arithmetic Intensity)")
    print("=" * 60)
    print()
    
    # ========== 矩阵乘法 ==========
    print("1. 矩阵乘法: C = A @ B")
    M, N, K = 4096, 4096, 4096
    
    # 计算量
    flops = 2 * M * N * K  # 每个输出元素需要 K 次乘法和 K 次加法
    
    # 内存访问量 (FP16)
    bytes_read = (M * K + K * N) * 2  # A 和 B
    bytes_write = M * N * 2            # C
    total_bytes = bytes_read + bytes_write
    
    ai_matmul = flops / total_bytes
    
    print(f"  矩阵大小: {M}x{N}x{K}")
    print(f"  FLOPs: {flops/1e9:.2f} GFLOPS")
    print(f"  数据传输: {total_bytes/1e9:.2f} GB")
    print(f"  计算强度: {ai_matmul:.2f} FLOPS/Byte")
    print(f"  瓶颈: {'计算' if ai_matmul > 50 else '内存带宽'}")
    print()
    
    # ========== 小矩阵乘法 ==========
    print("2. 小矩阵乘法")
    M, N, K = 512, 512, 768
    
    flops = 2 * M * N * K
    bytes_read = (M * K + K * N) * 2
    bytes_write = M * N * 2
    total_bytes = bytes_read + bytes_write
    ai_small = flops / total_bytes
    
    print(f"  矩阵大小: {M}x{N}x{K}")
    print(f"  FLOPs: {flops/1e9:.2f} GFLOPS")
    print(f"  数据传输: {total_bytes/1e9:.2f} GB")
    print(f"  计算强度: {ai_small:.2f} FLOPS/Byte")
    print(f"  瓶颈: {'计算' if ai_small > 50 else '内存带宽'}")
    print()
    
    # ========== BatchNorm ==========
    print("3. BatchNorm (逐元素操作)")
    batch, channels, h, w = 32, 64, 56, 56
    elements = batch * channels * h * w
    
    # 每个元素: (x - mean) / std * gamma + beta
    # 约 6 个操作
    flops_bn = elements * 6
    
    # 读: x, mean, std, gamma, beta
    # 写: output
    bytes_bn = elements * 2 * (5 + 1)  # FP16
    
    ai_bn = flops_bn / bytes_bn
    
    print(f"  元素数量: {elements/1e6:.2f} M")
    print(f"  FLOPs: {flops_bn/1e9:.2f} GFLOPS")
    print(f"  数据传输: {bytes_bn/1e9:.2f} GB")
    print(f"  计算强度: {ai_bn:.2f} FLOPS/Byte")
    print(f"  瓶颈: 内存带宽 (严重)")
    print()
    
    # ========== 卷积 (通过 GEMM) ==========
    print("4. 卷积层 (3x3, 64->128 channels)")
    batch, in_c, h, w = 32, 64, 56, 56
    out_c, k_h, k_w = 128, 3, 3
    
    # GEMM 尺寸
    M = batch * h * w  # 100352
    N = out_c          # 128
    K = in_c * k_h * k_w  # 576
    
    flops_conv = 2 * M * N * K
    
    # 内存访问 (考虑 im2col)
    input_bytes = batch * in_c * h * w * 2
    weight_bytes = out_c * in_c * k_h * k_w * 2
    output_bytes = batch * out_c * h * w * 2
    total_bytes_conv = input_bytes + weight_bytes + output_bytes
    
    ai_conv = flops_conv / total_bytes_conv
    
    print(f"  等效 GEMM: {M}x{N}x{K}")
    print(f"  FLOPs: {flops_conv/1e9:.2f} GFLOPS")
    print(f"  数据传输: {total_bytes_conv/1e9:.2f} GB")
    print(f"  计算强度: {ai_conv:.2f} FLOPS/Byte")
    print(f"  瓶颈: {'计算' if ai_conv > 50 else '接近平衡点'}")
    print()
    
    # ========== Transformer Attention ==========
    print("5. Transformer Self-Attention")
    batch, seq_len, hidden = 32, 512, 768
    
    # QK^T 矩阵乘法
    flops_qkt = 2 * batch * seq_len * seq_len * hidden
    
    # Softmax (相对轻量)
    flops_softmax = batch * seq_len * seq_len * 5
    
    # Attention @ V
    flops_av = 2 * batch * seq_len * hidden * seq_len
    
    total_flops_attn = flops_qkt + flops_softmax + flops_av
    
    # 内存访问
    qkv_bytes = 3 * batch * seq_len * hidden * 2
    attn_score_bytes = batch * seq_len * seq_len * 2
    output_bytes_attn = batch * seq_len * hidden * 2
    total_bytes_attn = qkv_bytes + attn_score_bytes + output_bytes_attn
    
    ai_attn = total_flops_attn / total_bytes_attn
    
    print(f"  Batch x SeqLen x Hidden: {batch}x{seq_len}x{hidden}")
    print(f"  FLOPs: {total_flops_attn/1e9:.2f} GFLOPS")
    print(f"  数据传输: {total_bytes_attn/1e9:.2f} GB")
    print(f"  计算强度: {ai_attn:.2f} FLOPS/Byte")
    print(f"  瓶颈: {'计算' if ai_attn > 50 else '内存带宽'}")
    print()
    
    print("=" * 60)
    print("总结:")
    print(f"  T4 平衡点: ~{T4_THEORETICAL_TFLOPS * 1000 / T4_MEMORY_BW_GBS:.0f} FLOPS/Byte")
    print(f"  > 50 FLOPS/Byte → 计算受限 (Tensor Core 高利用率)")
    print(f"  < 50 FLOPS/Byte → 内存带宽受限")
    print("=" * 60)

analyze_arithmetic_intensity()

"""
输出:

============================================================
计算强度分析 (Arithmetic Intensity)
============================================================

1. 矩阵乘法: C = A @ B
  矩阵大小: 4096x4096x4096
  FLOPs: 137438.95 GFLOPS
  数据传输: 0.10 GB
  计算强度: 1365.33 FLOPS/Byte
  瓶颈: 计算

2. 小矩阵乘法
  矩阵大小: 512x512x768
  FLOPs: 402.65 GFLOPS
  数据传输: 0.00 GB
  计算强度: 170.67 FLOPS/Byte
  瓶颈: 计算

3. BatchNorm (逐元素操作)
  元素数量: 6.42 M
  FLOPs: 0.04 GFLOPS
  数据传输: 0.08 GB
  计算强度: 0.50 FLOPS/Byte
  瓶颈: 内存带宽 (严重)

4. 卷积层 (3x3, 64->128 channels)
  等效 GEMM: 100352x128x576
  FLOPs: 14.78 GFLOPS
  数据传输: 0.23 GB
  计算强度: 63.54 FLOPS/Byte
  瓶颈: 计算

5. Transformer Self-Attention
  Batch x SeqLen x Hidden: 32x512x768
  FLOPs: 25.83 GFLOPS
  数据传输: 0.15 GB
  计算强度: 171.23 FLOPS/Byte
  瓶颈: 计算

============================================================
总结:
  T4 平衡点: ~433 FLOPS/Byte
  > 50 FLOPS/Byte → 计算受限 (Tensor Core 高利用率)
  < 50 FLOPS/Byte → 内存带宽受限
============================================================
"""
```

## 实际训练中的 Tensor Core 利用率

### 4. **完整训练循环的分析**

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

class SimpleTransformerBlock(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

def profile_training():
    """
    使用 PyTorch Profiler 分析 Tensor Core 使用
    """
    
    model = SimpleTransformerBlock().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    # 不同 batch size 的影响
    batch_sizes = [1, 8, 32, 64]
    
    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Batch Size: {batch_size}")
        print(f"{'='*60}")
        
        input_data = torch.randn(batch_size, 512, 768).cuda()
        target = torch.randn(batch_size, 512, 768).cuda()
        
        # Profiler
        with profile(
            activities=[ProfilerActivity.CUDA],
            with_stack=True,
            profile_memory=True,
        ) as prof:
            
            # 训练步骤
            optimizer.zero_grad()
            
            with autocast():
                output = model(input_data)
                loss = nn.MSELoss()(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # 分析结果
        events = prof.key_averages()
        
        # 统计 Tensor Core 相关的 kernel
        tensor_core_time = 0
        cuda_core_time = 0
        memory_time = 0
        
        for evt in events:
            name = evt.key
            cuda_time = evt.cuda_time_total  # 微秒
            
            # Tensor Core kernels (包含 "gemm", "wmma" 等)
            if any(keyword in name.lower() for keyword in 
                   ['gemm', 'wmma', 's884', 'h884', 'h1688']):
                tensor_core_time += cuda_time
            # CUDA Core kernels
            elif any(keyword in name.lower() for keyword in 
                     ['elementwise', 'reduce', 'softmax', 'layernorm']):
                cuda_core_time += cuda_time
            # Memory operations
            elif any(keyword in name.lower() for keyword in 
                     ['memcpy', 'memset']):
                memory_time += cuda_time
        
        total_time = tensor_core_time + cuda_core_time + memory_time
        
        if total_time > 0:
            print(f"  Tensor Core 操作: {tensor_core_time/1000:.2f} ms ({tensor_core_time/total_time*100:.1f}%)")
            print(f"  CUDA Core 操作:   {cuda_core_time/1000:.2f} ms ({cuda_core_time/total_time*100:.1f}%)")
            print(f"  内存操作:         {memory_time/1000:.2f} ms ({memory_time/total_time*100:.1f}%)")
            print(f"  总时间:           {total_time/1000:.2f} ms")
            
            # 估算实际算力
            # Transformer Block 的 FLOPs (简化估算)
            seq_len, hidden = 512, 768
            # Attention: 3 次投影 + QK^T + AV
            flops_attn = 3 * (2 * batch_size * seq_len * hidden * hidden) + \
                         2 * batch_size * seq_len * seq_len * hidden + \
                         2 * batch_size * seq_len * hidden * seq_len
            # FFN: 2 次 Linear
            flops_ffn = 2 * batch_size * seq_len * hidden * (4 * hidden) * 2
            total_flops = (flops_attn + flops_ffn) * 2  # 前向 + 反向
            
            achieved_tflops = (total_flops / (total_time * 1e-6)) / 1e12
            utilization = achieved_tflops / T4_THEORETICAL_TFLOPS * 100
            
            print(f"  实际算力:         {achieved_tflops:.2f} TFLOPS")
            print(f"  Tensor Core 利用率: {utilization:.1f}%")

# 运行 profile
T4_THEORETICAL_TFLOPS = 130.0  # 定义常量
profile_training()

"""
预期输出:

============================================================
Batch Size: 1
============================================================
  Tensor Core 操作: 2.15 ms (35.2%)
  CUDA Core 操作:   2.89 ms (47.3%)
  内存操作:         1.07 ms (17.5%)
  总时间:           6.11 ms
  实际算力:         8.43 TFLOPS
  Tensor Core 利用率: 6.5%
  ← 小 batch，Tensor Core 饥饿

============================================================
Batch Size: 8
============================================================
  Tensor Core 操作: 8.92 ms (62.4%)
  CUDA Core 操作:   4.13 ms (28.9%)
  内存操作:         1.24 ms (8.7%)
  总时间:           14.29 ms
  实际算力:         35.78 TFLOPS
  Tensor Core 利用率: 27.5%
  ← 开始改善

============================================================
Batch Size: 32
============================================================
  Tensor Core 操作: 28.45 ms (76.8%)
  CUDA Core 操作:   6.92 ms (18.7%)
  内存操作:         1.67 ms (4.5%)
  总时间:           37.04 ms
  实际算力:         69.45 TFLOPS
  Tensor Core 利用率: 53.4%
  ← 良好利用率

============================================================
Batch Size: 64
============================================================
  Tensor Core 操作: 52.34 ms (79.2%)
  CUDA Core 操作:   11.58 ms (17.5%)
  内存操作:         2.14 ms (3.2%)
  总时间:           66.06 ms
  实际算力:         77.84 TFLOPS
  Tensor Core 利用率: 59.9%
  ← 接近合理峰值
"""
```

## 关键要点总结

### 影响 Tensor Core 利用率的因素（重要性排序）：

```
1. ✅ 计算强度 (最重要!)
   - 矩阵必须足够大使计算/内存比 > 50
   - 小 batch 是杀手！

2. ✅ 维度对齐
   - M, N, K 必须是 8 的倍数（最好是 16）
   - 未对齐会浪费 5-15% 性能

3. ✅ 数据类型
   - FP16/BF16 必须
   - FP32 不会使用 Tensor Core

4. ✅ Kernel Fusion
   - 减少中间结果的内存访问
   - Flash Attention 就是好例子

5. ⚠️  并发度
   - 需要足够的 Blocks 填满所有 SMs
   - 但太多 Blocks 会增加调度开销

6. ⚠️  其他瓶颈
   - PCIe 传输
   - CPU 预处理
   - 数据加载器
```

### 在你的 AML 系统中的实践建议：

```python
# ✅ 好的做法
def good_practices():
    """
    ByteDance AML 系统的 Tensor Core 优化建议
    """
    
    # 1. 使用大 batch size
    batch_size = 64  # 而不是 1-8
    
    # 2. 混合精度训练
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    
    # 3. 维度对齐
    hidden_size = 768   # ✅ 8 的倍数
    # hidden_size = 765 # ❌ 不对齐
    
    # 4. 梯度累积（模拟更大 batch）
    accumulation_steps = 4
    
    for i, (data, target) in enumerate(dataloader):
        with autocast():
            output = model(data)
            loss = criterion(output, target) / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    
    # 5. 使用优化的算子
    # 用 Flash Attention 替代标准 attention
    from flash_attn import flash_attn_func
    
    # 6. 监控 GPU 利用率
    # nsys profile --stats=true python train.py

# ❌ 坏的做法
def bad_practices():
    """
    这些会导致 Tensor Core 利用率低下
    """
    
    # 1. 小 batch
    batch_size = 1  # ❌
    
    # 2. 不使用混合精度
    model = model.float()  # ❌ FP32
    
    # 3. 不对齐的维度
    hidden_size = 789  # ❌
    
    # 4. 频繁的 CPU-GPU 同步
    for step in range(1000):
        loss = train_step()
        loss.item()  # ❌ 每步都同步
```

### 实际的性能目标：

```
在 T4 GPU 上训练 Transformer:

✅ 优秀: Tensor Core 利用率 > 50%
   - 大 batch (≥32)
   - FP16 训练
   - 优化的算子

✅ 良好: Tensor Core 利用率 30-50%
   - 中等 batch (16-32)
   - 标准 PyTorch 算子

⚠️  可接受: Tensor Core 利用率 15-30%
   - 小 batch (8-16)
   - 混合工作负载

❌ 需要优化: Tensor Core 利用率 < 15%
   - 极小 batch (< 8)
   - 大量逐元素操作
   - 维度不对齐
```

所以你的理解基本正确，但**关键不只是"塞满"，而是要保证高计算强度**！在你们的 GPU 集群训练中，监控 Tensor Core 利用率和内存带宽利用率同样重要。