# 需要面临的版本
nvcc -V
nvidia-smi  # 查看驱动支持的 CUDA 版本
python -c "import torch; print(torch.version.cuda)"  # PyTorch 的 CUDA 版本
理解版本关系

nvidia-smi 显示的版本：驱动支持的最高 CUDA 版本
nvcc 版本：编译 CUDA 代码用的工具链版本
PyTorch 的 CUDA 版本：PyTorch 编译时使用的 CUDA 版本