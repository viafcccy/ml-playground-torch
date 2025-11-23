太棒了，这种\*\*“修改-运行-验证”\*\*的循环是掌握 CUDA 的唯一捷径。

我们直接在之前的代码基础上，编写两个新的核函数：

1.  **Polynomial Kernel**: 计算 $y = x^2 + x + 5$
2.  **ReLU Kernel**: 计算 $y = \max(0, x)$

为了让你看清核心逻辑的变化，我把这两个函数写在一个完整的脚本里。

### 核心改动点（先看这里）

在写完整代码前，请注意我在 CUDA 内核中做的一个**关键优化**：

> **不要重复读取 `input[idx]`**。
>
>   * **Bad:** `output[idx] = input[idx] * input[idx] + input[idx] + 5;`
>       * 这可能会导致 GPU 去全局显存（Global Memory，很慢）里读两次数据。
>   * **Good:** `float val = input[idx]; output[idx] = val * val + val + 5;`
>       * 先把数据读到一个临时变量 `val` 里。这个变量存在\*\*寄存器（Register）\*\*中，它的速度比显存快几百倍。

-----

### 完整运行代码 (Copy & Run)

创建一个新文件 `cuda_advanced.py`，直接运行即可：

```python
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
    auto output = torch.empty_like(input);
    int size = input.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    poly_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size
    );
    return output;
}

torch::Tensor relu_compute(torch::Tensor input) {
    auto output = torch.empty_like(input);
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
```

-----

### 这里的关键知识点

#### 1\. 条件分支 (Branching)

在 ReLU 中我们用到了 `if` 或者 `? :`。

```cpp
output[idx] = (x > 0.0f) ? x : 0.0f;
```

  * **CUDA 这种写法效率高吗？**
      * **通常很高**。GPU 编译器非常聪明，对于简单的 `if-else`，它不会真的让线程“分叉”（Branch Divergence），而是使用一种叫 **Predication** 的技术（把它变成单纯的数学逻辑指令），所以这种简单的 ReLU 几乎没有性能损耗。

#### 2\. 浮点数字面量

注意我写的是 `5.0f` 和 `0.0f`，而不是 `5` 和 `0`。

  * 在 C++ CUDA 中，`5` 是整型（int），`5.0` 是双精度（double），`5.0f` 是单精度（float）。
  * 因为我们的 Input Tensor 是 `float32`，使用 `5.0f` 可以避免不必要的类型转换，保证最高性能。

#### 3\. 为什么要有 Host 包装函数？

你会发现 `poly_compute` 和 `relu_compute` 这两个 C++ 函数长得几乎一样，只是调用的 kernel 不同。

  * 这就是 CUDA 编程的常态：你需要写 **Kernel (GPU 逻辑)** 和 **Host Launcher (CPU 调度逻辑)**。
  * Host Launcher 负责计算 Block/Grid 的数量，这部分逻辑通常是通用的。

### 下一步挑战

如果你运行成功了，试着思考下面这个更难一点的问题：

> 现在的 ReLU 是把小于 0 的变成 0。
> **如果我想实现 Leaky ReLU 呢？**
> (即：如果 $x > 0$ 输出 $x$；如果 $x \le 0$ 输出 $0.01 \times x$)

你想试试修改代码实现它吗？这只需要改动一行代码。