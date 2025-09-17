"""
Skip-Connection RMSNorm (残差连接层归一化) 的Triton实现

RMSNorm (Root Mean Square Layer Normalization) 是一种高效的层归一化方法，
在现代Transformer模型中被广泛使用。Skip-Connection RMSNorm 结合了残差连接，
在一个操作中同时完成残差相加和层归一化。

数学原理：
1. 残差连接: x = x + residual
2. RMSNorm: y = (x / sqrt(mean(x²) + eps)) * weight

其中：
- x: 输入张量
- residual: 残差张量 (可选)
- weight: 可学习的缩放参数
- eps: 防止除零的小常数

与LayerNorm的区别：
- LayerNorm: y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
- RMSNorm: y = x / sqrt(mean(x²) + eps) * weight

RMSNorm的优势：
1. 计算更简单：不需要计算均值和方差，只需要计算均方根
2. 内存占用更少：不需要存储均值和方差
3. 数值更稳定：避免了减法运算可能导致的数值问题
4. 性能更好：计算量更少，特别适合大模型

应用场景：
- LLaMA系列模型
- T5模型的变种
- 其他现代Transformer架构
"""

"""
modified from https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/fused/skip_rms_norm.py
"""

import torch
import triton
import triton.language as tl

def calculate_settings(n):
    """
    根据问题规模计算最优的Triton内核配置
    
    参数：
        n (int): 特征维度大小
        
    返回：
        tuple: (BLOCK_SIZE, num_warps)
    """
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps

@triton.jit
def skip_rms_norm_kernel_no_view(
    Y_ptr,              # 输出张量指针
    X_ptr,              # 输入张量指针
    R_ptr,              # 残差张量指针 (可能为None)
    W_ptr,              # 权重张量指针
    B,                  # Batch size
    S,                  # Sequence length  
    N,                  # Hidden size (特征维度)
    x_stride_b,         # X张量的batch维度步幅
    x_stride_s,         # X张量的sequence维度步幅
    x_stride_n,         # X张量的hidden维度步幅
    r_stride_b,         # 残差张量的batch维度步幅
    r_stride_s,         # 残差张量的sequence维度步幅
    r_stride_n,         # 残差张量的hidden维度步幅
    y_stride_b,         # 输出张量的batch维度步幅
    y_stride_s,         # 输出张量的sequence维度步幅
    y_stride_n,         # 输出张量的hidden维度步幅
    w_stride,           # 权重张量步幅
    eps,                # 防止除零的小常数
    has_residual: tl.constexpr,  # 是否有残差连接 (编译时常量)
    BLOCK_SIZE: tl.constexpr,    # 线程块大小 (编译时常量)
):
    """
    Skip-Connection RMSNorm内核的优化版本
    
    这个内核避免了张量重塑操作，直接在原始3D张量上工作，
    提供更好的内存访问模式和性能。
    
    处理流程：
    1. 计算当前线程处理的(batch, sequence)位置
    2. 加载输入数据和可选的残差数据
    3. 执行残差相加 (如果有残差)
    4. 计算RMSNorm归一化
    5. 应用权重缩放并存储结果
    
    优化特点：
    - 避免了view操作的开销
    - 直接支持3D张量的步幅计算
    - 条件编译支持有/无残差的情况
    """
    # 计算当前程序处理的位置：每个程序处理一个(batch, sequence)位置
    pid = tl.program_id(0)
    batch_idx = pid // S    # 计算batch索引
    seq_idx = pid % S       # 计算sequence索引

    # 计算各张量在当前(batch, sequence)位置的起始地址
    X_ptr = X_ptr + batch_idx * x_stride_b + seq_idx * x_stride_s
    Y_ptr = Y_ptr + batch_idx * y_stride_b + seq_idx * y_stride_s

    # 计算当前线程块处理的特征维度索引
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N  # 边界掩码，处理特征维度不是2的幂次的情况

    # 加载输入数据并转换为float32以保证计算精度
    x = tl.load(X_ptr + cols * x_stride_n, mask=mask, other=0.0).to(tl.float32)

    # 如果有残差连接，执行残差相加并更新残差张量
    if has_residual:
        R_ptr = R_ptr + batch_idx * r_stride_b + seq_idx * r_stride_s
        r = tl.load(R_ptr + cols * r_stride_n, mask=mask, other=0.0).to(tl.float32)
        x = x + r  # 残差相加
        # 将相加后的结果写回残差张量，用于后续层的残差连接
        tl.store(R_ptr + cols * r_stride_n, x, mask=mask)

    # 计算RMSNorm
    # 1. 计算均方值: mean(x²)
    var = tl.sum(x * x, axis=0) / N
    # 2. 计算归一化系数: 1/sqrt(mean(x²) + eps)
    rrms = 1.0 / tl.sqrt(var + eps)

    # 加载权重参数
    w = tl.load(W_ptr + cols * w_stride, mask=mask, other=0.0)
    
    # 应用RMSNorm: (x * rrms) * weight
    # 先转换为float16以匹配输出类型，再应用权重缩放
    y = (x * rrms).to(tl.float16) * w

    # 存储最终结果
    tl.store(Y_ptr + cols * y_stride_n, y, mask=mask)


@torch.no_grad()
def skip_rmsnorm_no_view(X, residual, weight, eps=1e-5):
    """
    Skip-Connection RMSNorm的主函数 (无张量重塑版本)
    
    这个版本避免了张量重塑操作，直接在原始形状上工作，
    提供更好的性能，特别是对于大型张量。
    
    参数：
        X (torch.Tensor): 输入张量，形状为[B, S, N]
            - B: batch size
            - S: sequence length  
            - N: hidden size
        residual (torch.Tensor, optional): 残差张量，形状与X相同
            如果为None，则不执行残差连接
        weight (torch.Tensor): RMSNorm的权重参数，形状为[N]
        eps (float): 防止除零的小常数，默认1e-5
        
    返回：
        tuple: (Y, updated_residual)
            - Y: RMSNorm的输出，形状与X相同
            - updated_residual: 更新后的残差 (X + residual)，如果原本有残差的话
            
    性能优势：
        - 避免view操作的内存重排开销
        - 更好的内存访问局部性
        - 支持任意形状的3D张量
    """
    # 获取输入张量的形状信息
    B, S, N = X.shape
    Y = torch.empty_like(X)  # 创建输出张量

    # 获取各张量的内存步幅信息，用于计算内存地址
    x_stride_b, x_stride_s, x_stride_n = X.stride()
    y_stride_b, y_stride_s, y_stride_n = Y.stride()
    w_stride = weight.stride(0)

    # 处理残差张量的情况
    if residual is not None:
        residual = residual.contiguous()  # 确保残差张量内存连续
        r_stride_b, r_stride_s, r_stride_n = residual.stride()
        has_residual = True
    else:
        # 如果没有残差，设置默认步幅值（不会被使用）
        r_stride_b, r_stride_s, r_stride_n = 0, 0, 0
        has_residual = False

    # 计算线程块大小，设为不小于N的最小2的幂次
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    # 网格配置：每个(batch, sequence)位置启动一个程序
    grid = (B * S,)

    # 启动Triton内核
    skip_rms_norm_kernel_no_view[grid](
        Y,                    # 输出张量
        X,                    # 输入张量
        residual if residual is not None else X,  # 残差张量或占位张量
        weight,               # 权重参数
        B, S, N,             # 张量维度
        x_stride_b, x_stride_s, x_stride_n,  # 输入张量步幅
        r_stride_b, r_stride_s, r_stride_n,  # 残差张量步幅
        y_stride_b, y_stride_s, y_stride_n,  # 输出张量步幅
        w_stride,            # 权重张量步幅
        eps,                 # epsilon参数
        has_residual=has_residual,  # 是否有残差连接
        BLOCK_SIZE=BLOCK_SIZE,      # 线程块大小
    )

    # 返回结果：如果有残差则返回更新后的残差，否则返回原输入
    return (Y, residual) if residual is not None else (Y, X)


@triton.jit()
def rms_norm_kernel(
    Y,              # 输出张量指针
    X,              # 输入张量指针
    W,              # 权重张量指针
    y_stride_r,     # 输出张量行步幅
    y_stride_c,     # 输出张量列步幅
    x_stride_r,     # 输入张量行步幅
    x_stride_c,     # 输入张量列步幅
    N,              # 每行的元素数量 (特征维度)
    eps,            # 防止除零的小常数
    BLOCK_SIZE: tl.constexpr,  # 线程块大小
):
    """
    标准RMSNorm内核 (无残差连接版本)
    
    实现标准的RMSNorm归一化：
    y = (x / sqrt(mean(x²) + eps)) * weight
    
    适用于不需要残差连接的场景，如模型的第一层归一化。
    """
    pid = tl.program_id(0)  # 当前程序处理的行索引
    
    # 计算当前行在各张量中的起始地址
    Y += pid * y_stride_r
    X += pid * x_stride_r

    # 设置边界掩码和列索引
    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    
    # 加载输入数据
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)

    # 计算RMSNorm
    var = tl.sum(x * x / N, axis=0)  # 计算均方值
    rrms = 1 / tl.sqrt(var + eps)    # 计算归一化系数

    # 加载权重并应用归一化
    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    
    # 存储结果
    tl.store(Y + cols * y_stride_c, y, mask=mask)


@triton.jit()
def skip_rms_norm_kernel(
    Y,              # 输出张量指针
    X,              # 输入张量指针
    R,              # 残差张量指针
    W,              # 权重张量指针
    y_stride_r,     # 输出张量行步幅
    y_stride_c,     # 输出张量列步幅
    x_stride_r,     # 输入张量行步幅
    x_stride_c,     # 输入张量列步幅
    r_stride_r,     # 残差张量行步幅
    r_stride_c,     # 残差张量列步幅
    N,              # 每行的元素数量
    eps,            # 防止除零的小常数
    BLOCK_SIZE: tl.constexpr,  # 线程块大小
):
    """
    Skip-Connection RMSNorm内核 (传统版本)
    
    执行残差连接和RMSNorm的融合操作：
    1. x = x + residual  (残差相加)
    2. y = (x / sqrt(mean(x²) + eps)) * weight  (RMSNorm)
    3. 更新残差张量为相加后的值
    
    这个版本假设输入已经被重塑为2D张量。
    """
    pid = tl.program_id(0)  # 当前程序处理的行索引
    
    # 计算各张量当前行的起始地址
    Y += pid * y_stride_r
    X += pid * x_stride_r
    R += pid * r_stride_r

    # 设置掩码和列索引
    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    
    # 加载输入数据和残差数据
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)

    # 执行残差相加
    x += r
    # 将相加结果写回残差张量，供后续层使用
    tl.store(R + cols * r_stride_c, x, mask=mask)

    # 计算RMSNorm
    var = tl.sum(x * x / N, axis=0)  # 均方值
    rrms = 1 / tl.sqrt(var + eps)    # 归一化系数

    # 应用权重缩放
    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    
    # 存储最终结果
    tl.store(Y + cols * y_stride_c, y, mask=mask)


@torch.no_grad()
def skip_rmsnorm(X, residual, weight, eps=1e-5):
    """
    Skip-Connection RMSNorm的传统实现版本
    
    这个版本将输入张量重塑为2D进行处理，然后恢复原始形状。
    虽然在某些情况下可能有额外的内存重排开销，但兼容性更好。
    
    参数：
        X (torch.Tensor): 输入张量，任意形状，最后一个维度为特征维度
        residual (torch.Tensor, optional): 残差张量，形状与X相同
        weight (torch.Tensor): RMSNorm权重，形状为[特征维度]
        eps (float): 数值稳定性常数
        
    返回：
        tuple: (Y, updated_residual)
            - Y: RMSNorm输出，形状与X相同
            - updated_residual: 更新后的残差张量
            
    实现策略：
        1. 保存原始形状
        2. 将张量重塑为2D: (M, N)，其中N是特征维度
        3. 根据是否有残差选择不同的内核
        4. 恢复原始形状并返回
    """
    orig_shape = X.shape  # 保存原始形状
    X = X.contiguous().view(-1, orig_shape[-1])  # 重塑为2D

    M, N = X.shape  # M: 总行数, N: 特征维度
    BLOCK_SIZE, num_warps = calculate_settings(N)  # 计算最优内核配置
    Y = torch.empty_like(X)  # 创建输出张量

    if residual is not None:
        # 有残差连接的情况
        residual = residual.contiguous().view(-1, N)  # 残差也重塑为2D
        skip_rms_norm_kernel[M,](  # 启动skip-connection内核
            Y,          # 输出张量
            X,          # 输入张量
            residual,   # 残差张量
            weight,     # 权重参数
            N, 1,       # Y的步幅参数
            N, 1,       # X的步幅参数  
            N, 1,       # 残差的步幅参数
            N,          # 特征维度
            eps,        # epsilon
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return Y.view(orig_shape), residual.view(orig_shape)
    else:
        # 无残差连接的情况
        rms_norm_kernel[M,](  # 启动标准RMSNorm内核
            Y,          # 输出张量
            X,          # 输入张量
            weight,     # 权重参数
            N, 1,       # Y的步幅参数
            N, 1,       # X的步幅参数
            N,          # 特征维度
            eps,        # epsilon
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return Y.view(orig_shape), X.view(orig_shape)


import pytest
import time


def python_rmsnorm(x, w, eps=1e-5):
    # x: (B, N)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x / torch.sqrt(var + eps)
    return x_normed * w


def python_skip_rmsnorm(x, r, w, eps=1e-5):
    # x, r: (B, N)
    x = x + r
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x / torch.sqrt(var + eps)
    return (x_normed * w).half(), x.half()


@pytest.mark.parametrize(
    "batch_size, N, hidden_size", [(4, 128, 4096), (2, 256, 4096), (8, 1024, 4096)]
)
def test_rmsnorm(batch_size, N, hidden_size):
    x = torch.randn(batch_size, N, hidden_size, device="cuda", dtype=torch.float16)
    w = torch.randn(hidden_size, device="cuda", dtype=torch.float16)

    y_ref = python_rmsnorm(x.float(), w.float()).half()
    y_triton, triton_residual = skip_rmsnorm(
        x, None, w
    )  # 不传residual，就走rms_norm_kernel分支

    assert torch.allclose(y_ref, y_triton, atol=1e-3, rtol=1e-3), (
        "RMSNorm results do not match"
    )


@pytest.mark.parametrize(
    "batch_size, N, hidden_size", [(4, 128, 4096), (2, 256, 4096), (8, 1024, 4096)]
)
def test_skip_rmsnorm(batch_size, N, hidden_size):
    x = torch.randn(batch_size, N, hidden_size, device="cuda", dtype=torch.float16)
    r = torch.randn(batch_size, N, hidden_size, device="cuda", dtype=torch.float16)
    w = torch.randn(hidden_size, device="cuda", dtype=torch.float16)

    y_ref, py_residual = python_skip_rmsnorm(x.float(), r.float(), w.float())
    y_triton, triton_residual = skip_rmsnorm(x, r, w)

    assert torch.allclose(y_ref, y_triton, atol=1e-3, rtol=1e-3), (
        "Skip RMSNorm results do not match"
    )
    assert torch.allclose(py_residual, triton_residual, atol=1e-3, rtol=1e-3), (
        "Skip RMSNorm residual results do not match"
    )


def benchmark_skip_rmsnorm(batch_size, N, iters=1000):
    x = torch.randn(batch_size, N, device="cuda", dtype=torch.float16)
    r = torch.randn(batch_size, N, device="cuda", dtype=torch.float16)
    w = torch.randn(N, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        skip_rmsnorm(x, r, w)
    torch.cuda.synchronize()
    end = time.time()
    avg_time = (end - start) / iters
    print(f"skip_rmsnorm: B={batch_size}, N={N}, avg_time={avg_time * 1e3:.3f} ms/iter")


# 假设原始函数名为 rmsnorm_original
def benchmark(func, shapes, warmup=10, iters=50):
    times = []
    for shape in shapes:
        X = torch.randn(shape, dtype=torch.float16, device="cuda")
        R = torch.randn(shape, device="cuda", dtype=torch.float16)
        W = torch.randn(shape[-1], dtype=torch.float16, device="cuda")
        # warmup
        for _ in range(warmup):
            _ = func(X, R, W)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            _ = func(X, R, W)
        torch.cuda.synchronize()
        end = time.time()
        avg_time = (end - start) / iters
        times.append(avg_time)
    return times


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    # 示例运行
    shapes = [(16, 2048, 4096), (32, 2048, 4096), (64, 2048, 4096), (256, 2048, 4096)]
    original_times = benchmark(skip_rmsnorm, shapes)
    optimized_times = benchmark(skip_rmsnorm_no_view, shapes)

    plt.figure(figsize=(8, 5))
    x_axis = [s[0] * s[1] for s in shapes]
    plt.plot(x_axis, original_times, color="red", label="Original")
    plt.plot(x_axis, optimized_times, color="blue", label="Optimized")
    plt.xlabel("Batch * Seq (M dimension)")
    plt.ylabel("Time (s)")
    plt.title("RMSNorm Kernel Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("./skip_rmsnorm_benchmark.png")