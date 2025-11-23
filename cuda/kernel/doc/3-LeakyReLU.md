没问题！既然你已经掌握了基础，那我们就直接拿下 **Leaky ReLU**。

这次我们做一个**进阶升级**：
与其把负斜率（slope）死写成 `0.01`，不如把它做成一个**参数**。这样你在 Python 里想改成 `0.1` 或 `0.2` 都不用重新编译代码。

这是 CUDA 编程中非常重要的技能：**如何从 Python 传标量（Scalar）参数给 GPU。**

### 1\. 核心逻辑修改

我们需要修改三个地方：

1.  **Kernel**: 接收 `float alpha` 参数。
2.  **C++ Wrapper**: 接收 `float alpha` 并传给 Kernel。
3.  **Python**: 调用时传入具体的数值。

### 2\. 完整代码 (Copy & Run)

新建一个 `cuda_leaky_relu.py`，直接运行：

```python
import os
import torch
from torch.utils.cpp_extension import load_inline

# --- 环境配置 (保持之前的稳健设置) ---
conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix and 'CUDA_HOME' not in os.environ:
    os.environ['CUDA_HOME'] = conda_prefix

# --- CUDA 源码 ---
cuda_source = """
// 1. 注意这里多了一个参数: float alpha
__global__ void leaky_relu_kernel(const float* input, float* output, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        // 核心逻辑: 如果 x > 0 保持原样，否则乘以 alpha
        output[idx] = (x > 0.0f) ? x : (x * alpha);
    }
}

// 2. C++ 包装函数也需要接收 float alpha
torch::Tensor leaky_relu(torch::Tensor input, float alpha) {
    auto output = torch.empty_like(input);
    int size = input.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // 3. 启动核函数时，把 alpha 传进去
    // 注意参数顺序必须和 __global__ 函数完全一致！
    leaky_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        alpha,   // <--- 传值
        size
    );
    
    return output;
}
"""

cpp_source = "torch::Tensor leaky_relu(torch::Tensor input, float alpha);"

# --- 编译 ---
leaky_relu_module = load_inline(
    name='leaky_relu_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['leaky_relu'],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    extra_cflags=[f"-I{conda_prefix}/include"] if conda_prefix else []
)

# --- 测试 ---
def main():
    device = torch.device("cuda")
    N = 1000
    x = torch.randn(N, device=device)
    
    # 定义负半轴的斜率
    negative_slope = 0.1
    
    print(f"Running Leaky ReLU with slope = {negative_slope}...")
    
    # 1. 运行我们的 CUDA 版本
    y_custom = leaky_relu_module.leaky_relu(x, negative_slope)
    
    # 2. 运行 PyTorch 原生版本 (F.leaky_relu)
    y_ref = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
    
    # 3. 验证
    if torch.allclose(y_custom, y_ref):
        print("✅ Success! Implementation matches PyTorch.")
    else:
        print("❌ Error! Mismatch found.")
        
    # 打印几个负数看看效果
    mask = x < 0
    if mask.any():
        print("\n--- Negative value check ---")
        print(f"Input:  {x[mask][:3]}")
        print(f"Output: {y_custom[mask][:3]}")
        print(f"Expected (Input * {negative_slope}): {x[mask][:3] * negative_slope}")

if __name__ == "__main__":
    main()
```

-----

### 💡 深度解析：按值传递 vs 按指针传递

注意看这一行：

```cpp
leaky_relu_kernel<<<...>>>(..., alpha, size);
```

在 CUDA Kernel 中：

  * **指针 (`float* input`)**：传递的是显存地址。核函数通过地址去显存（Global Memory）里找数据。
  * **标量 (`float alpha`, `int size`)**：是**按值传递 (Pass-by-value)**。
      * 这些小数值会被直接复制到 GPU 的常量缓存（Constant Memory）或者通过内核参数传递。
      * **访问速度极快**，比去显存读数组快得多。

### 🚀 下一阶段的挑战

你已经完全掌握了 **Grid-Stride Loops (一维网格跨步循环)** 的基础：

1.  计算索引 `idx`。
2.  边界检查 `if (idx < size)`。
3.  处理数据。

到目前为止，我们处理的都是**一维数组**（即使 PyTorch Tensor 是二维的，我们在内存里也把它当成一维连续数组处理，这叫 `numel()`）。

**如果我们要处理真正的矩阵乘法（Matrix Multiplication），或者图像处理（2D坐标），该怎么办？**

需要我带你进入 **2D Grid / 2D Block** 的世界吗？这会涉及到 `blockIdx.y` 和 `threadIdx.y`。