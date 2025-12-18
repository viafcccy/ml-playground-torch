"""
完整的 ResNet-50 4-GPU 分布式训练示例
每个关键步骤都有详细注释，说明每张卡在做什么
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def setup(rank, world_size):
    """
    初始化分布式环境
    
    参数:
        rank: 当前进程的全局 rank (0, 1, 2, 3)
        world_size: 总进程数 (4)
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 配置 NCCL (只在必要时)
    os.environ['NCCL_IB_DISABLE'] = '1'      # 单机无 InfiniBand
    os.environ['NCCL_DEBUG'] = 'WARN'         # 减少日志
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    print(f"[Rank {rank}] 初始化完成")


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def prepare_dataloader(rank, world_size, batch_size=128):
    """
    准备数据加载器
    
    关键: 使用 DistributedSampler 确保每个 GPU 拿到不同数据
    
    返回:
        dataloader: 数据加载器
        sampler: 分布式采样器 (需要每个 epoch 设置 seed)
    """
    # 数据预处理
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载 ImageNet 数据集 (这里用 CIFAR10 演示)
    dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # 分布式采样器
    # 作用: 将数据集分成 world_size 份，每个 rank 拿一份
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,  # 总 GPU 数
        rank=rank,                # 当前 GPU 编号
        shuffle=True,             # 打乱数据
        seed=42                   # 随机种子
    )
    
    # 数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size // world_size,  # 每个 GPU 的 batch size
        sampler=sampler,          # 使用分布式采样器
        num_workers=4,            # 数据加载线程数
        pin_memory=True,          # 加速 CPU→GPU 传输
    )
    
    if rank == 0:
        print(f"数据加载器准备完成:")
        print(f"  - 总 batch size: {batch_size}")
        print(f"  - 每 GPU batch size: {batch_size // world_size}")
        print(f"  - 数据集大小: {len(dataset)}")
        print(f"  - 每 GPU 样本数: {len(dataset) // world_size}")
    
    return dataloader, sampler


def create_model(rank):
    """
    创建模型
    
    步骤:
    1. 在 CPU 上创建模型
    2. 移动到对应的 GPU
    3. 用 DDP 包装
    
    返回:
        model: DDP 包装后的模型
    """
    # 1. 创建 ResNet-50
    model = models.resnet50(pretrained=False, num_classes=10)  # CIFAR10 用 10 类
    
    if rank == 0:
        print(f"模型创建完成:")
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  - 参数量: {param_count:,} ({param_count * 4 / 1024**2:.1f} MB in FP32)")
    
    # 2. 移动到对应 GPU
    model = model.cuda(rank)
    
    # 3. DDP 包装
    # 这是关键！DDP 会:
    #   - 在所有 GPU 上广播初始参数 (确保起点相同)
    #   - 在 backward() 时自动调用 NCCL All-Reduce 同步梯度
    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        
        # 梯度桶大小 (默认 25MB)
        # 梯度累积到 25MB 就触发 All-Reduce
        bucket_cap_mb=25,
        
        # 梯度作为桶视图 (节省显存)
        gradient_as_bucket_view=True,
        
        # 如果模型结构固定，设为 True 加速
        static_graph=False,
    )
    
    if rank == 0:
        print(f"DDP 包装完成")
    
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, epoch, rank, world_size):
    """
    训练一个 epoch
    
    这个函数展示了每个 step 中每张卡的详细工作
    """
    model.train()
    
    # 用于统计
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for step, (images, labels) in enumerate(dataloader):
        # ========== Phase 1: 数据传输 ==========
        # 每个 GPU 将自己的 mini-batch 从 CPU 传到 GPU
        images = images.cuda(rank, non_blocking=True)
        labels = labels.cuda(rank, non_blocking=True)
        
        # 此时:
        # GPU 0: images[0:32], labels[0:32]
        # GPU 1: images[32:64], labels[32:64]
        # GPU 2: images[64:96], labels[64:96]
        # GPU 3: images[96:128], labels[96:128]
        
        # ========== Phase 2: 前向传播 ==========
        # 各 GPU 独立计算，无通信
        outputs = model(images)
        
        # 此时:
        # GPU 0: outputs_0 (通过 GPU 0 的 32 张图计算)
        # GPU 1: outputs_1 (通过 GPU 1 的 32 张图计算)
        # GPU 2: outputs_2 (通过 GPU 2 的 32 张图计算)
        # GPU 3: outputs_3 (通过 GPU 3 的 32 张图计算)
        
        # ========== Phase 3: 计算损失 ==========
        # 各 GPU 独立计算自己的损失
        loss = criterion(outputs, labels)
        
        # 此时:
        # GPU 0: loss_0 (只基于 GPU 0 的数据)
        # GPU 1: loss_1 (只基于 GPU 1 的数据)
        # GPU 2: loss_2 (只基于 GPU 2 的数据)
        # GPU 3: loss_3 (只基于 GPU 3 的数据)
        # 四个损失值很可能不同！
        
        # ========== Phase 4: 反向传播 + 梯度同步 ==========
        optimizer.zero_grad()
        
        # 关键！DDP 在这里自动处理梯度同步
        loss.backward()
        
        # backward() 做了什么？
        # 1. 计算本地梯度 (各 GPU 独立)
        #    GPU 0: grad_0 (基于 loss_0 反向传播)
        #    GPU 1: grad_1 (基于 loss_1 反向传播)
        #    GPU 2: grad_2 (基于 loss_2 反向传播)
        #    GPU 3: grad_3 (基于 loss_3 反向传播)
        #
        # 2. 每层梯度就绪后立即 All-Reduce (后台异步)
        #    grad_avg = (grad_0 + grad_1 + grad_2 + grad_3) / 4
        #
        # 3. 所有梯度同步完成后，backward() 返回
        #    此时所有 GPU 的梯度完全相同: grad_avg
        
        # ========== Phase 5: 参数更新 ==========
        # 各 GPU 独立更新参数
        optimizer.step()
        
        # 因为梯度相同，所以更新后参数也相同！
        # param_new = param_old - lr * grad_avg
        # 所有 GPU 的 param_new 完全一致
        
        # ========== 统计 (需要跨 GPU 汇总) ==========
        # 计算准确率
        _, predicted = outputs.max(1)
        correct = (predicted == labels).sum().item()
        
        # 这些是本地统计，需要跨 GPU 求和
        total_loss += loss.item()
        total_correct += correct
        total_samples += labels.size(0)
        
        # 打印进度 (只在 rank 0 打印)
        if rank == 0 and step % 10 == 0:
            print(f"Epoch {epoch}, Step {step}/{len(dataloader)}, "
                  f"Loss: {loss.item():.4f}")
    
    # ========== 跨 GPU 汇总指标 ==========
    # 将所有 GPU 的统计数据汇总
    
    # 方法 1: 手动 All-Reduce
    loss_tensor = torch.tensor([total_loss]).cuda(rank)
    correct_tensor = torch.tensor([total_correct]).cuda(rank)
    samples_tensor = torch.tensor([total_samples]).cuda(rank)
    
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
    
    global_loss = loss_tensor.item() / len(dataloader) / world_size
    global_accuracy = correct_tensor.item() / samples_tensor.item()
    
    if rank == 0:
        print(f"\nEpoch {epoch} 完成:")
        print(f"  - 平均损失: {global_loss:.4f}")
        print(f"  - 准确率: {global_accuracy * 100:.2f}%")
    
    return global_loss, global_accuracy


def main():
    """主函数"""
    # ========== 配置 ==========
    world_size = 4  # 4 个 GPU
    epochs = 10
    batch_size = 128  # 全局 batch size
    learning_rate = 0.1
    
    # ========== 初始化 ==========
    # 每个 GPU 启动一个进程
    import torch.multiprocessing as mp
    
    def worker(rank, world_size):
        """每个 GPU 运行的工作函数"""
        print(f"\n{'='*60}")
        print(f"GPU {rank} 启动")
        print(f"{'='*60}\n")
        
        # 1. 初始化分布式环境
        setup(rank, world_size)
        
        # 2. 设置当前 GPU
        torch.cuda.set_device(rank)
        
        # 3. 准备数据
        dataloader, sampler = prepare_dataloader(rank, world_size, batch_size)
        
        # 4. 创建模型
        model = create_model(rank)
        
        # 5. 损失函数和优化器
        criterion = nn.CrossEntropyLoss().cuda(rank)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        # 6. 训练循环
        for epoch in range(epochs):
            # 设置 epoch (用于 shuffle)
            sampler.set_epoch(epoch)
            
            # 训练一个 epoch
            train_one_epoch(
                model, dataloader, criterion, optimizer,
                epoch, rank, world_size
            )
        
        # 7. 清理
        cleanup()
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"训练完成！")
            print(f"{'='*60}\n")
    
    # 启动多进程
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()