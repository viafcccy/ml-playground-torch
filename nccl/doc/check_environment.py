"""
环境检查脚本
检查 T4 + PyTorch + CUDA 11 + Python 3.9 环境是否正确配置
"""

import sys
import torch
import subprocess


def print_header(title):
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def check_python_version():
    """检查 Python 版本"""
    print_header("Python 版本检查")
    version = sys.version_info
    print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 9:
        print("✅ Python 版本正确")
        return True
    else:
        print("⚠️  建议使用 Python 3.9")
        return False


def check_pytorch():
    """检查 PyTorch"""
    print_header("PyTorch 检查")
    print(f"PyTorch 版本: {torch.__version__}")
    
    if torch.cuda.is_available():
        print("✅ CUDA 可用")
        print(f"CUDA 版本: {torch.version.cuda}")
        return True
    else:
        print("❌ CUDA 不可用")
        return False


def check_gpu():
    """检查 GPU"""
    print_header("GPU 检查")
    
    if not torch.cuda.is_available():
        print("❌ 未检测到 CUDA GPU")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个 GPU")
    
    if gpu_count < 2:
        print("⚠️  NCCL 分布式训练需要至少 2 个 GPU")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}:")
        print(f"  名称: {props.name}")
        print(f"  计算能力: {props.major}.{props.minor}")
        print(f"  总显存: {props.total_memory / 1024**3:.2f} GB")
        print(f"  多处理器数: {props.multi_processor_count}")
        
        # 检查是否是 T4
        if "T4" in props.name:
            print(f"  ✅ 检测到 T4 GPU")
        
        # 检查计算能力（T4 是 7.5）
        if props.major >= 7:
            print(f"  ✅ 支持 Tensor Core 和混合精度训练")
    
    return gpu_count >= 2


def check_nccl():
    """检查 NCCL"""
    print_header("NCCL 检查")
    
    try:
        # 检查 PyTorch 是否支持 NCCL
        if torch.distributed.is_nccl_available():
            print("✅ NCCL 后端可用")
            return True
        else:
            print("❌ NCCL 后端不可用")
            return False
    except Exception as e:
        print(f"❌ NCCL 检查失败: {e}")
        return False


def check_cuda_ipc():
    """检查 CUDA IPC（进程间通信）"""
    print_header("CUDA IPC 检查")
    
    try:
        # 尝试创建一个可以在进程间共享的 tensor
        tensor = torch.ones(10).cuda()
        tensor.share_memory_()
        print("✅ CUDA IPC 工作正常")
        return True
    except Exception as e:
        print(f"❌ CUDA IPC 失败: {e}")
        return False


def test_basic_nccl():
    """测试基本的 NCCL 操作"""
    print_header("NCCL 功能测试")
    
    if torch.cuda.device_count() < 2:
        print("⚠️  需要至少 2 个 GPU 才能测试 NCCL")
        return False
    
    try:
        import torch.distributed as dist
        import os
        
        # 设置环境
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12361'
        
        # 初始化进程组
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=1,
            rank=0
        )
        
        print("✅ NCCL 进程组初始化成功")
        
        # 清理
        dist.destroy_process_group()
        return True
        
    except Exception as e:
        print(f"❌ NCCL 测试失败: {e}")
        return False


def check_bandwidth():
    """简单的带宽测试"""
    print_header("GPU 互连带宽估算")
    
    if torch.cuda.device_count() < 2:
        print("⚠️  需要至少 2 个 GPU 才能测试带宽")
        return
    
    try:
        import time
        
        # 在两个 GPU 之间传输数据
        size = 100 * 1024 * 1024  # 100M 个 float32
        tensor0 = torch.randn(size).cuda(0)
        
        # 预热
        for _ in range(5):
            tensor1 = tensor0.cuda(1)
        
        # 计时
        torch.cuda.synchronize()
        start = time.time()
        
        iterations = 10
        for _ in range(iterations):
            tensor1 = tensor0.cuda(1)
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        
        # 计算带宽
        data_size = size * 4  # bytes
        bandwidth = (data_size * iterations) / elapsed / 1e9  # GB/s
        
        print(f"GPU 0 -> GPU 1 传输带宽: {bandwidth:.2f} GB/s")
        
        if bandwidth < 5:
            print("⚠️  带宽较低，可能存在配置问题")
        elif bandwidth < 12:
            print("✅ 带宽正常 (PCIe 3.0 x16 预期范围)")
        else:
            print("✅ 带宽良好")
        
    except Exception as e:
        print(f"❌ 带宽测试失败: {e}")


def print_recommendations():
    """打印优化建议"""
    print_header("针对 T4 的优化建议")
    
    print("""
1. 通信优化:
   - T4 没有 NVLink，GPU 间通过 PCIe 通信
   - 使用梯度累积减少同步频率
   - 建议 2-4 卡训练，8 卡以上可能遇到通信瓶颈

2. 显存优化:
   - T4 有 16GB 显存，适中规模
   - 使用混合精度训练 (FP16/BF16)
   - 启用梯度检查点节省显存

3. 性能优化:
   - 充分利用 Tensor Core
   - batch size 设置为 8 的倍数
   - 使用 DataLoader 的 num_workers 和 pin_memory

4. 环境变量:
   export NCCL_DEBUG=WARN          # 调试时用 INFO
   export NCCL_SOCKET_IFNAME=eth0  # 多机时指定网卡
   export NCCL_IB_DISABLE=1        # 如果没有 InfiniBand
    """)


def main():
    print("="*70)
    print(" NCCL 环境检查工具")
    print(" 适用于: T4 + PyTorch + CUDA 11 + Python 3.9")
    print("="*70)
    
    checks = [
        ("Python 版本", check_python_version),
        ("PyTorch", check_pytorch),
        ("GPU", check_gpu),
        ("NCCL", check_nccl),
        ("CUDA IPC", check_cuda_ipc),
        ("NCCL 功能", test_basic_nccl),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ 检查 {name} 时出错: {e}")
            results[name] = False
    
    # 带宽测试
    try:
        check_bandwidth()
    except Exception as e:
        print(f"带宽测试出错: {e}")
    
    # 打印建议
    print_recommendations()
    
    # 总结
    print_header("检查总结")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n通过: {passed}/{total} 项检查")
    
    if all(results.values()):
        print("\n✅ 环境配置完整，可以开始学习 NCCL！")
        print("\n建议运行顺序:")
        print("  1. python 01_nccl_basic.py")
        print("  2. python 02_nccl_operations.py")
        print("  3. python 06_nccl_visualization.py")
        print("  4. python 03_nccl_ddp_training.py")
        print("  5. python 04_nccl_benchmark.py")
        print("  6. python 05_nccl_advanced.py")
    else:
        print("\n⚠️  部分检查未通过，请根据上述信息修复问题")
    
    print("="*70)


if __name__ == "__main__":
    main()