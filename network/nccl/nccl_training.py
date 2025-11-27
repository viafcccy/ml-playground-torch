"""
NCCL 示例 3: 分布式数据并行训练 (DDP)
环境: T4 + PyTorch + CUDA 11 + Python 3.9

功能: 演示如何使用 NCCL 进行实际的多 GPU 训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import os


# 简单的模型
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# 简单的数据集
class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.data = torch.randn(size, 10)
        self.labels = torch.randn(size, 1)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, epochs=5):
    """分布式训练函数"""
    print(f"[Rank {rank}] 开始训练进程")
    
    # 初始化
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    # 创建模型并移到对应 GPU
    model = SimpleNet().cuda(rank)
    
    # 用 DDP 包装模型（内部使用 NCCL 同步梯度）
    ddp_model = DDP(model, device_ids=[rank])
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    
    # 创建数据集和分布式采样器
    dataset = SimpleDataset(size=1000)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler
    )
    
    # 训练循环
    for epoch in range(epochs):
        # 设置 epoch（用于打乱数据）
        sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(rank), target.cuda(rank)
            
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()  # 反向传播时，DDP 会自动使用 NCCL all-reduce 同步梯度
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx == 0:  # 只打印第一个 batch
                print(f"[Rank {rank}] Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # 计算平均损失（需要跨 GPU 同步）
        avg_loss = epoch_loss / len(dataloader)
        
        # 使用 all-reduce 计算全局平均损失
        loss_tensor = torch.tensor([avg_loss]).cuda(rank)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        global_avg_loss = loss_tensor.item() / world_size
        
        if rank == 0:
            print(f"Epoch {epoch}, 全局平均损失: {global_avg_loss:.4f}")
    
    # 清理
    cleanup()
    print(f"[Rank {rank}] 训练完成")


def main():
    world_size = torch.cuda.device_count()
    print(f"检测到 {world_size} 个 GPU")
    print("开始分布式训练...\n")
    
    if world_size < 2:
        print("此示例需要至少 2 个 GPU")
        return
    
    # 启动多进程训练
    mp.spawn(train,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
    print("\n训练完成！")


if __name__ == "__main__":
    main()