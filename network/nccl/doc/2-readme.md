# NCCL 学习示例集 - T4 + PyTorch + CUDA 11 + Python 3.9

这是一套完整的 NCCL (NVIDIA Collective Communications Library) 学习示例，特别针对 T4 GPU 环境优化。

## 环境要求

```bash
- NVIDIA T4 GPU (2+ 块)
- CUDA 11.x
- Python 3.9
- PyTorch 1.10+ (with CUDA 11 support)
```

## 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 示例列表

### 1. 基础示例 - All-Reduce (`01_nccl_basic.py`)

**学习内容:**
- NCCL 的基本初始化流程
- 分布式进程的启动方式
- All-Reduce 操作的原理

**运行:**
```bash
python 01_nccl_basic.py
```

**预期输出:**
```
检测到 2 个 GPU
[Rank 0] 初始张量: tensor([1., 1., 1.])
[Rank 1] 初始张量: tensor([2., 2., 2.])
[Rank 0] All-Reduce 后: tensor([3., 3., 3.])
[Rank 1] All-Reduce 后: tensor([3., 3., 3.])
```

---

### 2. 集合通信操作 (`02_nccl_operations.py`)

**学习内容:**
- Broadcast: 广播操作
- Reduce: 归约操作
- All-Gather: 全收集操作
- Reduce-Scatter: 归约分散操作

**运行:**
```bash
python 02_nccl_operations.py
```

**核心概念:**

| 操作 | 输入 | 输出 | 应用场景 |
|------|------|------|----------|
| Broadcast | 一个源 | 所有 GPU 相同 | 分发模型参数 |
| Reduce | 所有 GPU | 一个目标 | 收集损失值 |
| All-Reduce | 所有 GPU | 所有 GPU 相同 | 梯度同步 |
| All-Gather | 所有 GPU | 所有 GPU 收集所有 | 收集预测结果 |

---

### 3. 分布式训练 (`03_nccl_ddp_training.py`)

**学习内容:**
- DistributedDataParallel (DDP) 的使用
- 分布式数据采样器
- 梯度自动同步
- 跨 GPU 指标计算

**运行:**
```bash
python 03_nccl_ddp_training.py
```

**关键点:**
- DDP 会在 `backward()` 时自动使用 NCCL 同步梯度
- 使用 DistributedSampler 确保每个 GPU 看到不同数据
- 训练指标需要手动跨 GPU 聚合

---

### 4. 性能测试 (`04_nccl_benchmark.py`)

**学习内容:**
- 测量 NCCL 通信延迟和带宽
- 了解 T4 在 PCIe 下的性能特征
- 性能瓶颈分析

**运行:**
```bash
python 04_nccl_benchmark.py
```

**T4 性能参考值 (PCIe 3.0 x16):**
- 理论带宽: ~16 GB/s
- 实际 All-Reduce 带宽: 5-10 GB/s (取决于数据大小)
- Ping-Pong 延迟: 50-100 μs

**性能优化建议:**
1. 增大通信数据量可提高带宽利用率
2. 减少小数据通信次数
3. 使用梯度累积减少同步频率

---

### 5. 高级技巧 (`05_nccl_advanced.py`)

**学习内容:**
- 梯度累积: 增加有效 batch size
- 混合精度训练: 减少通信量和显存
- DDP 参数优化: bucket_cap_mb, gradient_as_bucket_view
- 进程组: 自定义通信拓扑

**运行:**
```bash
python 05_nccl_advanced.py
```

**最佳实践:**

1. **梯度累积**
   ```python
   accumulation_steps = 4
   loss = loss / accumulation_steps
   loss.backward()
   if (step + 1) % accumulation_steps == 0:
       optimizer.step()
       optimizer.zero_grad()
   ```

2. **混合精度**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   
   with autocast():
       output = model(input)
       loss = criterion(output, target)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

3. **DDP 优化**
   ```python
   model = DDP(
       model,
       device_ids=[rank],
       bucket_cap_mb=25,  # 调整梯度桶大小
       gradient_as_bucket_view=True,  # 减少显存
   )
   ```

---

## NCCL 核心概念总结

### 通信模式

```
All-Reduce (最常用):
GPU0: [1,2,3]  ─┐
GPU1: [4,5,6]  ─┤ Sum ──> GPU0: [5,7,9]
GPU2: [0,0,0]  ─┘          GPU1: [5,7,9]
                           GPU2: [5,7,9]

Broadcast:
GPU0: [1,2,3]  ────────> GPU0: [1,2,3]
GPU1: [?,?,?]  ────────> GPU1: [1,2,3]
GPU2: [?,?,?]  ────────> GPU2: [1,2,3]
```

### T4 特点与优化

**T4 的限制:**
- ❌ 无 NVLink，GPU 间通过 PCIe 3.0 通信
- ⚠️ 带宽限制: ~16 GB/s (vs NVLink 300+ GB/s)
- ⚠️ 延迟较高: ~100 μs (vs NVLink ~10 μs)

**针对 T4 的优化策略:**

1. **减少通信频率**
   - 使用梯度累积
   - 增大 batch size
   - 减少同步点

2. **减少通信数据量**
   - 混合精度训练 (FP16)
   - 梯度压缩 (可选)
   - 只同步必要的参数

3. **重叠计算与通信**
   - DDP 自动实现
   - bucket_cap_mb 调优

4. **合理规划拓扑**
   - 2-4 卡训练效果较好
   - 8 卡以上考虑通信瓶颈

---

## 常见问题

### Q1: 如何检查 NCCL 是否正常工作？

```bash
# 设置环境变量启用 NCCL 调试
export NCCL_DEBUG=INFO
python 01_nccl_basic.py
```

### Q2: 多机多卡怎么配置？

```python
# 在每台机器上设置
os.environ['MASTER_ADDR'] = 'master节点IP'
os.environ['MASTER_PORT'] = '12355'
os.environ['WORLD_SIZE'] = '8'  # 总GPU数
os.environ['RANK'] = '0'  # 当前节点的全局rank

dist.init_process_group("nccl")
```

### Q3: 训练速度没有提升？

可能原因:
1. **通信开销过大**: 模型太小，通信时间占比高
2. **数据加载瓶颈**: num_workers 设置不当
3. **batch size 不当**: 每个 GPU 的 batch 太小

解决方案:
- 增大模型或 batch size
- 使用梯度累积
- 优化数据加载 pipeline

### Q4: Out of Memory 错误？

```python
# 1. 减小 batch size
batch_size = 16  # 从 32 降到 16

# 2. 使用混合精度
with autocast():
    output = model(input)

# 3. 启用梯度检查点
from torch.utils.checkpoint import checkpoint
output = checkpoint(model.layer, input)
```

---

## 调试技巧

### 启用详细日志

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

### 检查通信性能

```bash
# NCCL 自带的测试工具
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make
./build/all_reduce_perf -b 8 -e 256M -f 2 -g 2
```

---

## 学习路径建议

1. **Day 1**: 运行 01-02，理解基本通信操作
2. **Day 2**: 运行 03，理解 DDP 训练流程
3. **Day 3**: 运行 04，了解性能特征
4. **Day 4**: 运行 05，学习优化技巧
5. **Day 5**: 在自己的项目中应用

---

## 参考资源

- [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [PyTorch DDP 教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL Tests](https://github.com/NVIDIA/nccl-tests)

---

## 贡献与反馈

如有问题或建议，欢迎提 Issue！