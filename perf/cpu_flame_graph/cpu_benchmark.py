"""
PyTorch T4 GPU 场景的 CPU Benchmark 程序
包含数据预处理、模型推理等典型 CPU 密集操作
"""

import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
import json


class SyntheticDataset(Dataset):
    """模拟真实场景的数据集，包含 CPU 密集的预处理"""
    
    def __init__(self, size=10000):
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 模拟 CPU 密集的数据预处理操作
        # 1. 生成随机数据
        data = np.random.randn(3, 224, 224).astype(np.float32)
        
        # 2. 数据增强操作（CPU 密集）
        data = self._augment(data)
        
        # 3. 归一化
        data = (data - data.mean()) / (data.std() + 1e-8)
        
        label = np.random.randint(0, 1000)
        
        return torch.from_numpy(data), label
    
    def _augment(self, data):
        """模拟数据增强操作"""
        # 随机翻转
        if np.random.rand() > 0.5:
            data = np.flip(data, axis=2).copy()
        
        # 随机裁剪和缩放
        h, w = data.shape[1:]
        crop_size = int(h * (0.8 + 0.2 * np.random.rand()))
        start_h = np.random.randint(0, h - crop_size + 1)
        start_w = np.random.randint(0, w - crop_size + 1)
        data = data[:, start_h:start_h+crop_size, start_w:start_w+crop_size]
        
        # 简单的插值放大（CPU 密集）
        data = self._resize(data, (h, w))
        
        return data
    
    def _resize(self, data, target_size):
        """简单的双线性插值（CPU 密集操作）"""
        from scipy.ndimage import zoom
        c, h, w = data.shape
        scale_h = target_size[0] / h
        scale_w = target_size[1] / w
        return zoom(data, (1, scale_h, scale_w), order=1)


class SimpleResNet(nn.Module):
    """简单的 ResNet-like 模型"""
    
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 简化的残差块
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4)
        self.layer3 = self._make_layer(128, 256, 6)
        self.layer4 = self._make_layer(256, 512, 3)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            if i == 0 and in_channels != out_channels:
                layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def cpu_intensive_postprocess(outputs):
    """模拟 CPU 密集的后处理操作"""
    # 1. 转换到 CPU 并转为 numpy（常见操作）
    outputs_np = outputs.cpu().numpy()
    
    # 2. 复杂的后处理计算
    probs = np.exp(outputs_np) / np.sum(np.exp(outputs_np), axis=1, keepdims=True)
    
    # 3. Top-K 选择
    top5_indices = np.argsort(probs, axis=1)[:, -5:]
    
    # 4. 生成结果字典（序列化操作）
    results = []
    for i in range(len(probs)):
        result = {
            'top5_classes': top5_indices[i].tolist(),
            'top5_probs': probs[i, top5_indices[i]].tolist(),
            'entropy': -np.sum(probs[i] * np.log(probs[i] + 1e-8))
        }
        results.append(result)
    
    return results


def run_benchmark(num_iterations=100, batch_size=32, num_workers=4):
    """
    运行 CPU benchmark
    
    Args:
        num_iterations: 运行迭代次数
        batch_size: 批次大小
        num_workers: DataLoader 工作进程数
    """
    print("=" * 60)
    print("PyTorch T4 GPU 场景 CPU Benchmark")
    print("=" * 60)
    print(f"配置:")
    print(f"  - 迭代次数: {num_iterations}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - DataLoader Workers: {num_workers}")
    print(f"  - CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    # 创建数据集和 DataLoader
    dataset = SyntheticDataset(size=num_iterations * batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleResNet().to(device)
    model.eval()
    
    # 统计信息
    stats = {
        'data_loading_times': [],
        'gpu_transfer_times': [],
        'inference_times': [],
        'postprocess_times': [],
        'total_times': []
    }
    
    print("\n开始 Benchmark...\n")
    
    with torch.no_grad():
        for iteration, (data, labels) in enumerate(dataloader):
            if iteration >= num_iterations:
                break
            
            iter_start = time.time()
            
            # 1. 数据加载时间（已经在 DataLoader 中完成）
            data_load_time = time.time() - iter_start
            
            # 2. GPU 传输时间
            transfer_start = time.time()
            data = data.to(device)
            labels = labels.to(device)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            transfer_time = time.time() - transfer_start
            
            # 3. 推理时间
            inference_start = time.time()
            outputs = model(data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_time = time.time() - inference_start
            
            # 4. CPU 密集后处理时间
            postprocess_start = time.time()
            results = cpu_intensive_postprocess(outputs)
            postprocess_time = time.time() - postprocess_start
            
            total_time = time.time() - iter_start
            
            # 记录统计
            stats['data_loading_times'].append(data_load_time)
            stats['gpu_transfer_times'].append(transfer_time)
            stats['inference_times'].append(inference_time)
            stats['postprocess_times'].append(postprocess_time)
            stats['total_times'].append(total_time)
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations} | "
                      f"Total: {total_time*1000:.2f}ms | "
                      f"Data: {data_load_time*1000:.2f}ms | "
                      f"Transfer: {transfer_time*1000:.2f}ms | "
                      f"Inference: {inference_time*1000:.2f}ms | "
                      f"Postprocess: {postprocess_time*1000:.2f}ms")
    
    # 打印统计结果
    print("\n" + "=" * 60)
    print("Benchmark 结果统计")
    print("=" * 60)
    
    for key, values in stats.items():
        values = np.array(values) * 1000  # 转换为毫秒
        print(f"\n{key}:")
        print(f"  平均: {np.mean(values):.2f} ms")
        print(f"  中位数: {np.median(values):.2f} ms")
        print(f"  标准差: {np.std(values):.2f} ms")
        print(f"  最小: {np.min(values):.2f} ms")
        print(f"  最大: {np.max(values):.2f} ms")
        print(f"  P95: {np.percentile(values, 95):.2f} ms")
        print(f"  P99: {np.percentile(values, 99):.2f} ms")
    
    # 计算各阶段占比
    print("\n各阶段时间占比:")
    total_avg = np.mean(stats['total_times']) * 1000
    for key in ['data_loading_times', 'gpu_transfer_times', 'inference_times', 'postprocess_times']:
        avg = np.mean(stats[key]) * 1000
        percentage = (avg / total_avg) * 100
        print(f"  {key}: {percentage:.1f}%")
    
    print("=" * 60)
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CPU Benchmark for PyTorch T4 GPU scenarios')
    parser.add_argument('--iterations', type=int, default=10000, help='Number of iterations')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of DataLoader workers')
    
    args = parser.parse_args()
    
    run_benchmark(
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        num_workers=args.workers
    )