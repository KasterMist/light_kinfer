"""
SwiGLU激活函数的Triton实现

SwiGLU (Swish-Gated Linear Unit) 是一种门控激活函数，广泛应用于现代Transformer模型中，
特别是在大型语言模型的FFN (Feed-Forward Network) 层。

数学定义：
SwiGLU(x, y) = Swish(x) ⊙ y = (x * sigmoid(x)) ⊙ y

其中：
- x 和 y 是两个输入张量
- Swish(x) = x * sigmoid(x) 是Swish激活函数  
- ⊙ 表示逐元素相乘

特点：
1. 门控机制：使用一个输入控制另一个输入的信息流
2. 平滑激活：相比ReLU，Swish提供更平滑的梯度
3. 性能优秀：在大模型中表现出色，优于传统的ReLU和GELU

应用场景：
- Transformer模型的FFN层
- 大型语言模型（如LLaMA、PaLM等）
- 需要门控机制的神经网络层
"""

# reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py

import torch
import triton
import triton.language as tl
import functools


def is_hip() -> bool:
    """
    检查当前是否运行在AMD HIP环境上
    
    HIP (Heterogeneous-Compute Interface for Portability) 是AMD的GPU计算平台。
    不同的GPU平台在Triton内核配置上可能有差异，需要特殊处理。
    """
    return torch.version.hip is not None


def ensure_contiguous(fn):
    """
    装饰器：确保张量在内存中连续存储
    
    Triton内核要求输入张量在内存中连续，这个装饰器会自动
    将不连续的张量转换为连续存储，避免内存访问错误。
    """
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


def calculate_settings(n):
    """
    根据问题规模计算最优的Triton内核配置
    
    参数：
        n (int): 需要处理的元素数量（通常是特征维度大小）
        
    返回：
        tuple: (BLOCK_SIZE, num_warps)
            - BLOCK_SIZE: 每个线程块处理的元素数量
            - num_warps: 每个线程块使用的warp数量
            
    优化策略：
        1. BLOCK_SIZE设为2的幂次，提高内存访问效率
        2. 根据BLOCK_SIZE调整warp数量，平衡并行度和资源利用率
        3. 限制最大BLOCK_SIZE，避免寄存器压力过大
    """
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536  # Triton推荐的最大块大小
    BLOCK_SIZE = triton.next_power_of_2(n)  # 向上取整到最近的2的幂次
    
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    # 根据BLOCK_SIZE大小调整warp数量，实现最佳性能
    num_warps = 4  # 默认值
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16  # AMD GPU需要较少的warp
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
        
    return BLOCK_SIZE, num_warps


@triton.jit
def silu(x):
    """
    Swish/SiLU激活函数的Triton实现
    
    SiLU (Sigmoid Linear Unit) 又称为Swish激活函数：
    SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    
    特点：
    - 平滑且可微：相比ReLU，在x=0处平滑过渡
    - 自门控：函数本身具有门控特性
    - 有界下界：当x→-∞时，SiLU(x)→0
    - 无界上界：当x→+∞时，SiLU(x)→x
    
    在SwiGLU中的作用：
    作为门控函数，控制信息流的通过程度
    """
    return x * tl.sigmoid(x)


@triton.jit
def _swiglu_forward_kernel(
    a_ptr,              # 第一个输入张量的指针
    b_ptr,              # 第二个输入张量的指针
    c_ptr,              # 输出张量的指针
    row_stride,         # 行步幅，用于定位不同行的起始位置
    n_cols: tl.constexpr,     # 每行的元素数量（编译时常量）
    BLOCK_SIZE: tl.constexpr, # 线程块大小（编译时常量）
):
    """
    SwiGLU前向传播的Triton内核实现
    
    计算公式：c = SiLU(a) * b = (a * sigmoid(a)) * b
    
    内核设计：
    1. 每个程序（线程块）处理一行数据
    2. 在行内使用向量化操作，提高计算效率
    3. 使用掩码处理边界情况，确保正确性
    
    性能优化：
    - 向量化加载和存储操作
    - 合并内存访问模式
    - 最小化数据类型转换开销
    """
    # 获取当前程序的ID，对应处理的行索引
    program_id = tl.program_id(0).to(tl.int64)

    # 计算当前行在各个张量中的起始位置
    a_ptr += program_id * row_stride  # a张量当前行的起始地址
    b_ptr += program_id * row_stride  # b张量当前行的起始地址
    c_ptr += program_id * row_stride  # 输出张量当前行的起始地址

    # 计算当前线程块处理的列索引范围
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols  # 边界掩码，处理最后一个块可能不满的情况

    # 向量化加载数据
    # a_row转换为float32以确保sigmoid计算的数值精度
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    
    # 计算SwiGLU：SiLU(a) * b
    c_row = silu(a_row) * b_row
    
    # 向量化存储结果
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


def swiglu_forward(a, b):
    """
    SwiGLU激活函数的前向传播主函数
    
    实现SwiGLU(a, b) = SiLU(a) * b = (a * sigmoid(a)) * b
    
    参数：
        a (torch.Tensor): 第一个输入张量，用于门控计算
            形状: (..., hidden_size)
        b (torch.Tensor): 第二个输入张量，被门控的数据
            形状: (..., hidden_size)
            
    返回：
        torch.Tensor: SwiGLU的输出结果，形状与输入相同
        
    使用场景：
        在Transformer的FFN层中，通常的使用方式为：
        1. 输入x通过两个不同的线性层得到a和b
        2. 计算SwiGLU(a, b)作为激活后的结果
        
    实现策略：
        1. 将多维张量重塑为2D，便于并行处理
        2. 每行启动一个Triton程序进行计算
        3. 使用向量化操作提高计算效率
        4. 恢复原始形状并返回结果
        
    性能优势：
        - GPU内核融合：避免多次内存访问
        - 向量化计算：充分利用GPU的并行计算能力
        - 内存访问优化：连续内存访问模式
    """
    ori_shape = a.shape  # 保存原始形状用于最后恢复

    # 获取最后一个维度的大小（通常是hidden_size）
    n_cols = ori_shape[-1]
    
    # 将张量重塑为2D：(total_elements // n_cols, n_cols)
    # 这样可以将问题简化为对多行数据的并行处理
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    
    # 创建输出张量，形状与重塑后的输入相同
    c = torch.empty_like(a)
    n_rows = a.shape[0]  # 总行数

    # 根据列数计算最优的内核配置参数
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    # 启动Triton内核进行计算
    # 网格配置：(n_rows,) - 每行启动一个程序
    _swiglu_forward_kernel[(n_rows,)](
        a,                # 第一个输入张量
        b,                # 第二个输入张量  
        c,                # 输出张量
        c.stride(-2),     # 行步幅，等于n_cols
        n_cols=n_cols,    # 每行的元素数量
        BLOCK_SIZE=BLOCK_SIZE,  # 线程块大小
        num_warps=num_warps,    # warp数量
    )
    
    # 将输出张量恢复为原始形状并返回
    return c.view(*ori_shape)
