"""
Softmax分块计算模块

这个模块实现了一个高效的分块Softmax算法，用于处理大规模的输入张量。
当输入张量的最后一个维度很大时，传统的Softmax计算可能会遇到内存限制问题。
通过分块计算，可以在保持数值稳定性的同时减少内存占用。

核心思想：
1. 将输入按最后一个维度分块处理
2. 每个块分别计算log-sum-exp
3. 合并所有块的log-sum-exp结果
4. 使用合并后的结果计算最终的Softmax

数学原理：
对于输入x，Softmax(x) = exp(x - log-sum-exp(x))
当x很大时，可以将其分块：x = [x1, x2, ..., xk]
则 log-sum-exp(x) = log-sum-exp([log-sum-exp(x1), log-sum-exp(x2), ..., log-sum-exp(xk)])

优势：
- 减少内存占用
- 提高数值稳定性
- 支持超大维度的Softmax计算
"""

# modified from https://github.com/iclementine/optimize_softmax/blob/master/softmax_split.py

import triton
from triton import language as tl
import torch


@triton.jit
def logsumexp_kernel(
    out_ptr,        # 输出指针，存储每个分块的log-sum-exp结果
    in_ptr,         # 输入指针，指向需要计算的张量
    M,              # 第一个维度大小（batch维度）
    N,              # 第二个维度大小（特征维度）
    TILE_N: tl.constexpr,  # 分块大小，编译时常量
):
    """
    计算输入张量每个分块的log-sum-exp值
    
    这是Softmax分块计算的第一步，对输入的每个分块计算其log-sum-exp值。
    log-sum-exp(x) = max(x) + log(sum(exp(x - max(x))))
    这种计算方式保证了数值稳定性，避免exp函数溢出。
    
    实现细节：
    1. 每个程序处理一个分块
    2. 先找到分块内的最大值作为基准点
    3. 计算exp(x - max)的和
    4. 最终结果是max + log(sum)
    """
    # 获取当前程序在N维度和M维度上的ID
    pid_n = tl.program_id(0)  # N维度的程序ID（处理哪个分块）
    num_programs_n = tl.num_programs(0)  # N维度的总程序数
    pid_m = tl.program_id(1)  # M维度的程序ID（处理哪一行）

    # 计算当前分块处理的元素偏移量
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N  # 边界检查，确保不超出N的范围
    
    # 计算在输入张量中的全局偏移量
    offset = pid_m * N + n_offsets
    
    # 加载当前分块的数据，超出边界的位置用负无穷填充
    inp = tl.load(in_ptr + offset, mask=mask, other=-float("inf")).to(tl.float32)
    
    # 计算log-sum-exp的数值稳定版本
    m = tl.max(inp, 0)              # 找到分块内的最大值作为基准点
    e = tl.exp(inp - m)             # 计算exp(x - max)，避免溢出
    z = tl.sum(e, 0)                # 求和
    logz = m + tl.log(z)            # 最终的log-sum-exp结果

    # 将结果存储到输出张量的对应位置
    output_ptrs = out_ptr + pid_m * num_programs_n + pid_n
    tl.store(output_ptrs, logz)


@triton.jit
def combine_logsumexp_kernel(out_ptr, inp_ptr, M, N, TILE_N: tl.constexpr):
    """
    合并多个分块的log-sum-exp结果
    
    这是Softmax分块计算的第二步，将第一步得到的多个分块log-sum-exp值
    合并成一个最终的log-sum-exp值。
    
    合并原理：
    对于多个log-sum-exp值 [logz1, logz2, ..., logzk]，
    最终的log-sum-exp = log-sum-exp([logz1, logz2, ..., logzk])
    
    这保证了分块计算的结果与整体计算的结果在数学上等价。
    """
    pid_m = tl.program_id(0)  # 当前处理的行索引
    
    # 计算需要合并的log-sum-exp值的偏移量
    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N  # 确保不超出有效的分块数量
    
    # 加载所有分块的log-sum-exp值
    logzs = tl.load(inp_ptr + pid_m * N + n_offsets, other=-float("inf"), mask=mask).to(
        out_ptr.dtype.element_ty
    )
    
    # 对所有分块的log-sum-exp值再次应用log-sum-exp操作
    m = tl.max(logzs, 0)        # 找到最大值作为基准点
    e = tl.exp(logzs - m)       # 计算exp(logz - max)
    z = tl.sum(e, 0)            # 求和
    logz = m + tl.log(z)        # 最终合并的log-sum-exp值
    
    # 存储最终结果
    tl.store(out_ptr + pid_m, logz)


@triton.jit
def softmax_kernel(out_ptr, in_ptr, logz_ptr, M, N, TILE_N: tl.constexpr):
    """
    计算最终的Softmax结果
    
    这是Softmax分块计算的第三步，使用合并后的log-sum-exp值计算最终的Softmax输出。
    
    计算公式：
    Softmax(x) = exp(x - log-sum-exp(x))
    
    其中log-sum-exp(x)已经在前两步中计算完成。
    """
    pid_n = tl.program_id(0)  # 当前处理的分块索引
    pid_m = tl.program_id(1)  # 当前处理的行索引
    
    # 计算当前分块的元素偏移量
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    offset = pid_m * N + n_offsets
    mask = n_offsets < N  # 边界检查
    
    # 加载原始输入数据
    inp = tl.load(in_ptr + offset, mask=mask, other=-float("inf")).to(
        out_ptr.dtype.element_ty
    )
    
    # 加载对应行的合并log-sum-exp值
    logz = tl.load(logz_ptr + pid_m).to(tl.float32)
    
    # 计算Softmax：exp(x - log_sum_exp)
    out = tl.exp(inp - logz)
    
    # 将结果存储到输出张量
    tl.store(out_ptr + offset, out, mask=mask)


def softmax_split(x):
    """
    分块Softmax计算的主函数
    
    对于形状为(M, N)的输入张量，当N很大时，使用分块策略计算Softmax。
    这个函数协调三个GPU内核的执行，实现高效的大规模Softmax计算。
    
    参数：
        x (torch.Tensor): 输入张量，形状为(M, N)
        
    返回：
        torch.Tensor: Softmax结果，形状与输入相同
        
    计算流程：
        1. 将输入按N维度分块，每块大小为TILE_N
        2. 对每个分块计算log-sum-exp
        3. 合并所有分块的log-sum-exp结果
        4. 使用合并结果计算最终的Softmax值
        
    内存优化：
        - 分块大小自适应调整，最大为4096
        - 中间结果使用临时张量，避免内存浪费
        - 支持任意大小的输入张量
    """
    M, N = x.shape

    # 第一步：计算每个分块的log-sum-exp
    # =====================================
    
    # 计算合适的分块大小，既要保证效率，又要避免内存溢出
    TILE_N = min(4096, triton.next_power_of_2(N))
    num_tiles_n = triton.cdiv(N, TILE_N)  # 总分块数量
    
    # 创建临时张量存储每个分块的log-sum-exp结果
    logz = torch.empty((M, num_tiles_n), dtype=x.dtype, device=x.device)
    
    # 启动第一个内核：计算分块log-sum-exp
    grid = (num_tiles_n, M, 1)  # 网格大小：(分块数, 行数, 1)
    logsumexp_kernel[grid](logz, x, M, N, TILE_N)

    # 第二步：合并所有分块的log-sum-exp结果
    # ====================================
    
    # 创建张量存储合并后的log-sum-exp结果
    combined_logz = torch.empty((M,), dtype=x.dtype, device=x.device)
    
    # 为合并操作计算分块大小
    TILE_N = triton.next_power_of_2(num_tiles_n)
    grid = (M, 1, 1)  # 网格大小：每行一个程序
    
    # 启动第二个内核：合并log-sum-exp值
    combine_logsumexp_kernel[grid](combined_logz, logz, M, num_tiles_n, TILE_N)

    # 第三步：计算最终的Softmax结果
    # =============================
    
    # 创建输出张量
    out = torch.empty_like(x)
    
    # 重新计算分块参数（与第一步相同）
    TILE_N = min(4096, triton.next_power_of_2(N))
    num_tiles_n = triton.cdiv(N, TILE_N)
    grid = (num_tiles_n, M, 1)
    
    # 启动第三个内核：计算Softmax
    softmax_kernel[grid](out, x, combined_logz, M, N, TILE_N)
    
    return out
