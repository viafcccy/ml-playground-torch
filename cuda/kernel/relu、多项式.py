import os
import torch
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------
# 1. 自动配置环境路径 (避免之前的报错)
# ----------------------------------------------------------------
conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix and 'CUDA_HOME' not in os.environ:
    os.environ['CUDA_HOME'] = conda_prefix
    print(f"Set CUDA_HOME to {conda_prefix}")

# ----------------------------------------------------------------
# 2. CUDA 源代码: 包含两个新的 Kernel
# ----------------------------------------------------------------
cuda_source = """
// --- Kernel 1: 多项式计算 f(x) = x^2 + x + 5 ---
__global__ void poly_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // [优化] 读取一次到寄存器
        float x = input[idx];
        // 计算多项式
        output[idx] = (x * x) + x + 5.0f;
    }
}

// --- Kernel 2: ReLU 激活函数 f(x) = max(0, x) ---
__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        // C++ 的三元运算符: 如果 x > 0 返回 x，否则返回 0
        output[idx] = (x > 0.0f) ? x : 0.0f;
        
        // 也可以使用 CUDA 内置数学函数: output[idx] = fmaxf(0.0f, x);
    }
}

// --- C++ 包装函数 (Host 端代码) ---
torch::Tensor poly_compute(torch::Tensor input) {
    auto output = torch::empty_like(input);  // ✅ 修改: torch. -> torch::
    int size = input.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    poly_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size
    );
    return output;
}

torch::Tensor relu_compute(torch::Tensor input) {
    auto output = torch::empty_like(input);  // ✅ 修改: torch. -> torch::
    int size = input.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size
    );
    return output;
}
"""

cpp_source = """
torch::Tensor poly_compute(torch::Tensor input);
torch::Tensor relu_compute(torch::Tensor input);
"""

# ----------------------------------------------------------------
# 3. 编译与加载
# ----------------------------------------------------------------
print("Compiling CUDA kernels... (This might take a moment)")
my_cuda_module = load_inline(
    name='advanced_kernels',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['poly_compute', 'relu_compute'], # 注册两个函数
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    # 这一行是为了确保能找到头文件，以防万一
    extra_cflags=[f"-I{conda_prefix}/include"] if conda_prefix else []
)
print("Compilation complete!")

# ----------------------------------------------------------------
# 4. 测试逻辑
# ----------------------------------------------------------------
def main():
    device = torch.device("cuda")
    N = 1000
    # 生成包含正负数的随机数据，用于测试 ReLU
    x = torch.randn(N, device=device) 
    
    print(f"\nData sample: {x[:5]}")

    # --- 测试 1: 多项式 x^2 + x + 5 ---
    print("\n--- Testing Polynomial (x^2 + x + 5) ---")
    y_cuda_poly = my_cuda_module.poly_compute(x)
    y_torch_poly = (x ** 2) + x + 5
    
    if torch.allclose(y_cuda_poly, y_torch_poly):
        print(f"✅ Poly Match! Sample: {y_cuda_poly[:3]}")
    else:
        print(f"❌ Poly Mismatch! Max diff: {(y_cuda_poly - y_torch_poly).abs().max()}")

    # --- 测试 2: ReLU ---
    print("\n--- Testing ReLU (max(0, x)) ---")
    y_cuda_relu = my_cuda_module.relu_compute(x)
    y_torch_relu = torch.relu(x)
    
    if torch.allclose(y_cuda_relu, y_torch_relu):
        print(f"✅ ReLU Match! Sample: {y_cuda_relu[:3]}")
    else:
        print(f"❌ ReLU Mismatch!")

if __name__ == "__main__":
    main()