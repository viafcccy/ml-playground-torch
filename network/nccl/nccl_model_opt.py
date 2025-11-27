"""
NCCL 示例 5: 高级技巧和最佳实践
环境: T4 + PyTorch + CUDA 11 + Python 3.9

功能: 演示梯度累积、混合精度训练、通信优化等技巧
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import os


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class SimpleDataset(Dataset):
    def __init__(self, size=10000):
        self.data = torch.randn(size, 100)
        self.labels = torch.randint(0, 10, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_with_gradient_accumulation(rank, world_size, epochs=3):
    """
    技巧 1: 梯度累积
    - 用于增加有效 batch size 而不增加显存占用
    - 减少通信频率，提高 T4 在 PCIe 下的效率
    """
    print(f"[Rank {rank}] 启动梯度累积训练")
    
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    # 模型设置
    model = SimpleNet().cuda(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    
    # 数据
    dataset = SimpleDataset(size=1000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)
    
    # 梯度累积步数
    accumulation_steps = 4  # 有效 batch size = 8 * 4 = 32
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(rank), target.cuda(rank)
            
            # 前向传播
            output = ddp_model(data)
            loss = criterion(output, target)
            
            # 归一化损失（因为要累积）
            loss = loss / accumulation_steps
            loss.backward()
            
            # 每 accumulation_steps 才更新参数和同步梯度
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                if rank == 0 and batch_idx % 20 == 0:
                    print(f"Epoch {epoch}, Step {batch_idx}, Loss: {loss.item() * accumulation_steps:.4f}")
    
    cleanup()


def train_with_mixed_precision(rank, world_size, epochs=3):
    """
    技巧 2: 混合精度训练 (AMP)
    - 减少显存占用
    - 加速计算
    - 减少通信数据量
    """
    print(f"[Rank {rank}] 启动混合精度训练")
    
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    model = SimpleNet().cuda(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    
    # GradScaler 用于混合精度训练
    scaler = GradScaler()
    
    dataset = SimpleDataset(size=1000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(rank), target.cuda(rank)
            
            optimizer.zero_grad()
            
            # 使用自动混合精度
            with autocast():
                output = ddp_model(data)
                loss = criterion(output, target)
            
            # 缩放损失并反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Step {batch_idx}, Loss: {loss.item():.4f}")
    
    cleanup()


def train_with_bucketing(rank, world_size):
    """
    技巧 3: DDP 通信优化
    - bucket_cap_mb: 控制梯度桶大小
    - find_unused_parameters: 处理部分参数不参与前向传播的情况
    """
    print(f"[Rank {rank}] 启动带通信优化的训练")
    
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    model = SimpleNet().cuda(rank)
    
    # DDP 配置优化
    ddp_model = DDP(
        model,
        device_ids=[rank],
        bucket_cap_mb=25,  # 默认 25MB，可以调整
        find_unused_parameters=False,  # 如果确定所有参数都用到，设为 False 提高性能
        gradient_as_bucket_view=True,  # 减少显存使用
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    
    dataset = SimpleDataset(size=500)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    for epoch in range(2):
        sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(rank), target.cuda(rank)
            
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if rank == 0 and batch_idx == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    cleanup()


def demonstrate_process_group(rank, world_size):
    """
    技巧 4: 自定义进程组
    - 可以创建子组进行局部通信
    - 适合更复杂的通信拓扑
    """
    print(f"[Rank {rank}] 演示进程组")
    
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    # 创建一个子组（例如：只包含前一半的 GPU）
    if world_size >= 4:
        if rank < world_size // 2:
            # 前一半 GPU 形成一个组
            sub_group = dist.new_group(ranks=list(range(world_size // 2)))
            
            tensor = torch.ones(3).cuda(rank) * rank
            print(f"[Rank {rank}] 子组通信前: {tensor}")
            
            # 只在子组内进行 all-reduce
            dist.all_reduce(tensor, group=sub_group, op=dist.ReduceOp.SUM)
            print(f"[Rank {rank}] 子组通信后: {tensor}")
    else:
        print(f"[Rank {rank}] 需要至少 4 个 GPU 才能演示进程组")
    
    dist.barrier()
    cleanup()


def main():
    world_size = torch.cuda.device_count()
    print(f"检测到 {world_size} 个 GPU\n")
    
    if world_size < 2:
        print("此示例需要至少 2 个 GPU")
        return
    
    print("=" * 70)
    print("演示 1: 梯度累积")
    print("=" * 70)
    mp.spawn(train_with_gradient_accumulation, args=(world_size,), nprocs=world_size, join=True)
    
    print("\n" + "=" * 70)
    print("演示 2: 混合精度训练")
    print("=" * 70)
    mp.spawn(train_with_mixed_precision, args=(world_size,), nprocs=world_size, join=True)
    
    print("\n" + "=" * 70)
    print("演示 3: DDP 通信优化")
    print("=" * 70)
    mp.spawn(train_with_bucketing, args=(world_size,), nprocs=world_size, join=True)
    
    print("\n" + "=" * 70)
    print("演示 4: 进程组")
    print("=" * 70)
    mp.spawn(demonstrate_process_group, args=(world_size,), nprocs=world_size, join=True)
    
    print("\n训练完成！")


if __name__ == "__main__":
    main()