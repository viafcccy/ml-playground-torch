import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"可用GPU数量: {torch.cuda.device_count()}")

# 列出所有GPU
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
# 测试GPU运算
if torch.cuda.is_available():
    x = torch.rand(5, 3).cuda()
    print(f"\n测试张量在GPU上: {x.device}")
    print("GPU加速功能正常！")