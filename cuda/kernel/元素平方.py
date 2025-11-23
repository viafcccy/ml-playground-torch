import os
import sys
import torch
from torch.utils.cpp_extension import load_inline

# 获取当前 Conda 环境的路径
conda_prefix = os.environ.get('CONDA_PREFIX')

if conda_prefix:
    # 1. 告诉 PyTorch，CUDA_HOME 就在 Conda 环境里
    os.environ['CUDA_HOME'] = conda_prefix
    
    # 2. 确保编译器能找到头文件 (cuda_runtime.h)
    # 有时候 nvcc 需要显式指定 include 路径
    cflags = [f"-I{conda_prefix}/include"]
else:
    cflags = []

# 1. 定义 CUDA C++ 源代码
# 这里是真正的 CUDA 核心逻辑
cuda_source = """
__global__ void square_matrix_kernel(const float* input, float* output, int size) {
    // --- 核心逻辑开始 ---
    
    // 1. 计算当前线程的全局唯一 ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. 边界检查 (Guard)
    // 因为线程总数往往是 block_size 的倍数，可能稍微多于数据总数 size
    if (idx < size) {
        // 3. 执行计算
        output[idx] = input[idx] * input[idx];
    }
    
    // --- 核心逻辑结束 ---
}

// 这是 C++ 到 Python 的接口绑定代码 (PyTorch boilerplate)
torch::Tensor square_matrix(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);  // 改这里：torch:: 而不是 torch.

    // 设定配置
    const int threads_per_block = 256;
    // 计算需要多少个 block (向上取整)
    const int blocks = (size + threads_per_block - 1) / threads_per_block;

    // 启动核函数 (Launch Kernel)
    square_matrix_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size
    );

    return output;
}
"""

# C++ 头文件定义
cpp_source = "torch::Tensor square_matrix(torch::Tensor input);"

# 2. 使用 PyTorch 实时编译 CUDA 代码
# 第一次运行会慢几秒钟，因为在后台编译
square_cuda = load_inline(
    name='square_cuda',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['square_matrix'],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    # 确保构建目录存在，避免权限问题
    verbose=True
)

def main():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    device = torch.device("cuda")

    # 3. 准备数据
    N = 10000
    # 创建一个随机 Tensor 并移动到 T4 GPU 上
    x = torch.randn(N, device=device)
    
    # 4. 运行我们的自定义 CUDA 核函数
    print("Running custom CUDA kernel...")
    y_custom = square_cuda.square_matrix(x)

    # 5. 运行 PyTorch 原生函数作为对照
    print("Running PyTorch native...")
    y_ref = x ** 2

    # 6. 验证结果
    # 检查两者是否非常接近 (由于浮点数精度，不能用 ==)
    if torch.allclose(y_custom, y_ref):
        print("✅ Success! Your CUDA kernel matches PyTorch result.")
    else:
        print("❌ Error! Results do not match.")
        
    print(f"Input (first 5): {x[:5]}")
    print(f"Output (first 5): {y_custom[:5]}")

if __name__ == "__main__":
    main()