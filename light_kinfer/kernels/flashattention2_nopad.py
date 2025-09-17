"""
Flash Attention v2 NoP (No Padding) 实现模块

本模块实现了基于 Triton 的 Flash Attention v2 算法，专门针对无填充(nopad)场景优化。
Flash Attention 是一种内存高效的注意力机制实现，通过分块计算和在线softmax算法，
显著降低内存使用量，同时保持与标准注意力机制完全相同的数学计算结果。

主要特性：
1. 内存高效：使用分块计算避免存储完整的注意力矩阵
2. 无填充优化：支持变长序列，避免无效的填充计算
3. 因果掩码：支持自回归生成任务的因果注意力
4. GPU加速：基于Triton实现的CUDA内核，充分利用GPU并行计算能力

核心算法原理：
- 使用在线softmax算法避免存储完整的QK^T矩阵
- 通过安全的数值计算防止指数运算溢出
- 利用GPU内存层次结构优化数据访问模式

参考文献：
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

参考实现：
- https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py
- https://github.com/ELS-RD/kernl/blob/main/src/kernl/implementations/attention.py#L438
- https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
"""

import torch, math
import triton
import triton.language as tl
from torch.amp import custom_fwd


# Triton内核自动调优配置列表
# 针对不同的块大小、线程束数量和流水线阶段进行优化
configs_tma = [
    triton.Config(
        {"BLOCK_M_SIZE": BM, "BLOCK_N_SIZE": BN}, num_stages=stages, num_warps=warps
    )
    for BM in [64, 128]          # M维度块大小候选值
    for BN in [32, 64, 128]      # N维度块大小候选值  
    for warps in [4, 8, 16]      # 线程束数量候选值
    for stages in [2, 3, 4, 6]   # 流水线阶段数候选值
]


def keep_tma(conf):
    """
    配置过滤函数，用于筛选适合当前GPU架构的内核配置
    
    对于Ada Lovelace架构(计算能力9.0)的GPU，过滤掉可能性能较差的配置：
    - 当块尺寸较小(< 128*128)且线程束数为8时，性能通常不佳
    
    Args:
        conf: Triton配置对象，包含块大小和并行参数
        
    Returns:
        bool: True表示保留该配置，False表示过滤掉该配置
    """
    BLOCK_M_SIZE = conf.kwargs["BLOCK_M_SIZE"]
    BLOCK_N_SIZE = conf.kwargs["BLOCK_N_SIZE"]
    if (
        torch.cuda.get_device_capability()[0] == 9
        and BLOCK_M_SIZE * BLOCK_N_SIZE < 128 * 128
        and conf.num_warps == 8
    ):
        return False
    return True


# key 参数列表(['B_Seqlen', 'HEAD_DIM'])的值会直接影响最佳配置的选择，因为不同的输入尺寸或问题规模可能需要不同的内核调度策略。
# 可选的自动调优装饰器，根据序列长度和头维度自动选择最优配置
# @triton.autotune(
#     configs=list(filter(keep_tma, configs_tma)),
#     key=['B_Seqlen', 'HEAD_DIM']
# )
@triton.jit
def flash_attention2_nopad_kernel(
    Q,              # Query张量指针
    K,              # Key张量指针  
    V,              # Value张量指针
    O,              # Output输出张量指针
    B_Start_Loc,    # 每个批次在序列中的起始位置
    B_Seqlen,       # 每个批次的序列长度
    sm_scale,       # 缩放因子，通常为1/sqrt(head_dim)
    heads,          # 注意力头的总数量
    num_kv_groups,  # KV缓存的分组数量，用于Grouped Query Attention
    # Q张量的步长参数
    stride_q_bs,    # Q在批次维度的步长
    stride_q_heads, # Q在头维度的步长  
    stride_q_dim,   # Q在特征维度的步长
    # K张量的步长参数
    stride_k_bs,    # K在批次维度的步长
    stride_k_heads, # K在头维度的步长
    stride_k_dim,   # K在特征维度的步长
    # V张量的步长参数
    stride_v_bs,    # V在批次维度的步长
    stride_v_heads, # V在头维度的步长
    stride_v_dim,   # V在特征维度的步长
    # O输出张量的步长参数
    stride_o_bs,    # O在批次维度的步长
    stride_o_heads, # O在头维度的步长
    stride_o_dim,   # O在特征维度的步长
    # 编译时常量参数
    HEAD_DIM: tl.constexpr,        # 注意力头的特征维度
    BLOCK_M_SIZE: tl.constexpr,    # Q矩阵行方向的块大小
    BLOCK_N_SIZE: tl.constexpr,    # K/V矩阵列方向的块大小
):
    """
    Flash Attention v2 NoP 内核实现
    
    这是Flash Attention算法的核心Triton内核，实现了内存高效的注意力计算。
    与标准注意力机制相比，该实现避免了存储完整的注意力矩阵，显著降低内存使用。
    
    算法原理：
    1. 分块处理：将Q,K,V矩阵分成小块进行计算，避免一次性加载所有数据
    2. 在线softmax：使用数值稳定的在线算法计算softmax，避免存储中间结果
    3. 累积更新：逐块累积注意力输出，保持数值稳定性
    4. 因果掩码：支持自回归任务的下三角掩码
    
    内存访问模式：
    - 每个线程块处理Q的BLOCK_M_SIZE行
    - 循环遍历K,V的BLOCK_N_SIZE列
    - 利用GPU共享内存缓存当前块的数据
    
    数值稳定性：
    - 使用log-sum-exp技巧防止softmax计算溢出
    - 在线更新最大值和归一化因子
    - 支持混合精度计算(fp16/fp32)
    """
    # 获取当前线程块的索引和批次/头信息
    block_m_idx = tl.program_id(0)  # 当前处理的Q矩阵行块索引
    cur_bh = tl.program_id(1)       # 当前批次×头的组合索引
    cur_batch_idx = cur_bh // heads      # 当前批次索引
    cur_head_idx = cur_bh % heads        # 当前注意力头索引
    cur_kv_head_idx = cur_head_idx // num_kv_groups  # 对应的KV头索引(用于GQA)

    # 计算当前批次的序列长度和在连续存储中的起始位置
    cur_seq_len = tl.load(B_Seqlen + cur_batch_idx)        # 当前批次的实际序列长度
    cur_seq_start_loc = tl.load(B_Start_Loc + cur_batch_idx)  # 当前批次在展平张量中的起始位置

    # 计算当前线程块要处理的Q矩阵行范围
    block_start_loc = block_m_idx * BLOCK_M_SIZE  # 当前块在序列中的起始位置

    # 创建张量索引的偏移向量
    offs_n = tl.arange(0, BLOCK_N_SIZE)  # K,V矩阵列方向的偏移向量[0,1,2,...,BLOCK_N_SIZE-1]
    offs_d = tl.arange(0, HEAD_DIM)      # 特征维度的偏移向量[0,1,2,...,HEAD_DIM-1]  
    offs_m = block_start_loc + tl.arange(0, BLOCK_M_SIZE)  # Q矩阵行方向的全局偏移

    # 计算Q张量的内存访问偏移并加载当前块的Q数据
    # Q的形状: [total_tokens, num_heads, head_dim]
    q_offs = (
        (cur_seq_start_loc + offs_m[:, None]) * stride_q_bs    # 批次维度偏移
        + cur_head_idx * stride_q_heads                        # 头维度偏移
        + offs_d[None, :] * stride_q_dim                       # 特征维度偏移
    )
    # 加载Q数据，使用掩码确保不超出当前序列长度
    q = tl.load(Q + q_offs, mask=offs_m[:, None] < cur_seq_len, other=0.0)

    # 计算K和V张量的基础内存偏移（不包含序列位置）
    # K,V共享相同的内存布局，使用GQA时多个Q头可能对应同一个KV头
    k_offs = (
        offs_n[None, :] * stride_k_bs         # K的序列维度基础偏移
        + cur_kv_head_idx * stride_k_heads    # KV头维度偏移
        + offs_d[:, None] * stride_k_dim      # 特征维度偏移
    )
    v_offs = (
        offs_n[:, None] * stride_v_bs         # V的序列维度基础偏移  
        + cur_kv_head_idx * stride_v_heads    # KV头维度偏移
        + offs_d[None, :] * stride_v_dim      # 特征维度偏移
    )

    # 保存K和V的内存指针，后续循环中会基于这些指针计算实际地址
    k_ptrs = K + k_offs
    v_ptrs = V + v_offs

    # 初始化在线softmax算法的状态变量
    # 在线softmax算法参考：Online normalizer calculation for softmax
    # m_i: 当前块的最大值，用于数值稳定的softmax计算
    # d_i: 当前块的归一化因子(分母)，即exp(x-m)的累积和
    m_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32) - float("inf")  # 初始化为负无穷
    d_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32)                 # 初始化为0
    acc = tl.zeros((BLOCK_M_SIZE, HEAD_DIM), dtype=tl.float32)        # 输出累积器

    # 计算当前块的有效范围，确保不超出序列边界
    block_mask = tl.where(block_start_loc < cur_seq_len, 1, 0)  # 块是否有效的掩码
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M_SIZE, cur_seq_len)  # 块的结束位置

    # 主循环：遍历K,V矩阵的所有列块进行注意力计算
    # 每次循环处理BLOCK_N_SIZE个key/value向量
    for start_n in range(0, block_mask * block_end_loc, BLOCK_N_SIZE):
        start_n = tl.multiple_of(start_n, BLOCK_N_SIZE)  # 确保对齐到块边界
        
        # 加载当前K块的数据
        # K的形状: [total_tokens, num_heads, head_dim] -> 转置后用于矩阵乘法
        k = tl.load(
            k_ptrs + (cur_seq_start_loc + start_n) * stride_k_bs,  # 加上序列位置偏移
            mask=(start_n + offs_n[None, :]) < block_end_loc,      # 确保不超出序列边界
            other=0.0,  # 超出边界的位置填充0
        )

        # 计算注意力分数：Q @ K^T 
        # q: [BLOCK_M_SIZE, HEAD_DIM], k: [HEAD_DIM, BLOCK_N_SIZE]
        # qk: [BLOCK_M_SIZE, BLOCK_N_SIZE]
        qk = tl.dot(q, k)

        # 应用因果掩码和缩放因子
        # 因果掩码确保当前位置只能看到之前的位置，实现自回归的注意力模式
        casual_mask = offs_m[:, None] >= (start_n + offs_n[None, :])  # 下三角掩码
        qk = tl.where(casual_mask, qk * sm_scale, -1.0e8)  # 掩码位置设为很大的负数

        # 在线softmax算法第一步：更新最大值
        # m_ij = max(m_i, max(qk_ij))，防止指数计算溢出
        m_ij = tl.maximum(m_i, tl.max(qk, 1))  # 计算当前块的行最大值
        qk -= m_ij[:, None]                     # 减去最大值保证数值稳定性
        p = tl.math.exp2(qk)                    # 计算exp(qk - m_ij)，即softmax的分子
        d_ij = tl.sum(p, 1)                     # 计算当前块的分母（按行求和）

        # 在线softmax算法第二步：更新归一化因子
        # 当最大值更新时，需要重新缩放之前的累积值
        alpha = tl.math.exp2(m_i - m_ij)  # 之前累积值的缩放因子
        d_i = d_i * alpha + d_ij          # 更新总的归一化因子

        # 在线softmax算法第三步：更新输出累积器
        # 重新缩放之前的累积输出，然后加上当前块的贡献
        acc = acc * alpha[:, None]  # 对之前的输出进行缩放

        # 加载当前V块并计算加权输出：P @ V
        v = tl.load(
            v_ptrs + (cur_seq_start_loc + start_n) * stride_v_bs,  # V的当前块地址
            mask=(start_n + offs_n[:, None]) < block_end_loc,      # 边界掩码
            other=0.0,  # 超出边界填充0
        )
        p = p.to(v.dtype)           # 确保数据类型匹配
        acc = tl.dot(p, v, acc)     # 计算P@V并累加到输出
        
        # 更新下次迭代的最大值
        m_i = m_ij

    # 最终归一化：将累积的输出除以最终的归一化因子
    # 此时acc存储的是所有块的加权和，d_i是对应的权重总和
    acc = acc / d_i[:, None]
    
    # 计算输出张量的内存偏移并存储结果
    off_o = (
        (cur_seq_start_loc + offs_m[:, None]) * stride_o_bs  # 批次维度偏移
        + cur_head_idx * stride_o_heads                      # 头维度偏移  
        + offs_d[None, :] * stride_o_dim                     # 特征维度偏移
    )
    out_ptrs = O + off_o
    # 存储最终的注意力输出，使用掩码确保不超出序列边界
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_seq_len)


# --------------------------------------
# Flash Attention v2 NoP 主函数实现
# --------------------------------------
@torch.no_grad()
@custom_fwd(cast_inputs=torch.float16, device_type="cuda")
def flash_attention2_no_pad(
    q: torch.Tensor,      # Query张量
    k: torch.Tensor,      # Key张量  
    v: torch.Tensor,      # Value张量
    sm_scale,             # 缩放因子
    b_start_loc,          # 每个批次的起始位置
    b_seq_len,            # 每个批次的序列长度
    max_seq_len,          # 最大序列长度
):
    """
    Flash Attention v2 无填充实现的主函数
    
    这是Flash Attention算法的高级接口，处理变长序列的高效注意力计算。
    通过避免填充操作和使用分块计算，显著提高内存效率和计算速度。
    
    算法优势：
    1. 内存高效：O(N)内存复杂度而不是O(N²)
    2. 数值稳定：使用在线softmax防止溢出
    3. 变长序列：支持无填充的批处理
    4. GPU优化：充分利用GPU内存层次结构
    
    应用场景：
    - 大语言模型的自注意力计算
    - 序列到序列任务
    - 多模态模型的交叉注意力
    
    注意事项：
    - 仅支持fp16输入以保证性能
    - 自动使用因果掩码适用于自回归任务
    - 输入张量必须在CUDA设备上
    
    Args:
        q: Query张量，形状[total_tokens, n_heads, head_dim]
           在decode阶段，q的序列长度通常为1
        k: Key张量，形状[total_tokens, n_heads, head_dim]
        v: Value张量，形状与k相同
        sm_scale: 注意力缩放因子，通常为1/sqrt(head_dim)
        b_start_loc: 每个批次在展平张量中的起始位置
        b_seq_len: 每个批次的实际序列长度
        max_seq_len: 所有序列中的最大长度
        
    Returns:
        torch.Tensor: 注意力输出，形状与q相同
        
    示例:
        >>> # 假设有2个批次，序列长度分别为10和8
        >>> total_tokens = 18
        >>> q = torch.randn(total_tokens, 8, 64, device='cuda', dtype=torch.float16)
        >>> k = torch.randn(total_tokens, 8, 64, device='cuda', dtype=torch.float16) 
        >>> v = torch.randn(total_tokens, 8, 64, device='cuda', dtype=torch.float16)
        >>> b_start_loc = torch.tensor([0, 10], device='cuda', dtype=torch.int32)
        >>> b_seq_len = torch.tensor([10, 8], device='cuda', dtype=torch.int32)
        >>> output = flash_attention2_no_pad(q, k, v, 0.125, b_start_loc, b_seq_len, 10)
    """
    # 创建与输入形状相同的输出张量
    output = torch.empty_like(q)
    
    # 解析输入张量的维度信息
    batchs = b_seq_len.shape[0]           # 批次数量
    n_heads, HEAD_DIM = q.shape[1], q.shape[2]  # 注意力头数和头维度

    # 设置Triton内核的执行参数
    # 这些参数影响GPU并行执行的效率和内存使用
    BLOCK_SIZE = 64  # 基础块大小，对于Ampere架构(如3090Ti)可设为128
    num_warps = 4 if HEAD_DIM <= 64 else 8  # 线程束数量，根据头维度自动调整
    num_stages = 1    # 流水线阶段数，影响内存带宽利用

    # 计算Grouped Query Attention的分组信息
    # 在GQA中，多个Query头可能共享同一组Key-Value头
    num_kv_groups = q.shape[1] // k.shape[1]  # Q头数除以KV头数得到分组数
    
    # 设置GPU线程网格：每个线程块处理一个序列块和一个头
    grid = (triton.cdiv(max_seq_len, BLOCK_SIZE), batchs * n_heads, 1)

    # 启动Triton内核进行Flash Attention计算
    flash_attention2_nopad_kernel[grid](
        # 输入输出张量
        q,                    # Query张量
        k,                    # Key张量  
        v,                    # Value张量
        output,               # 输出张量
        # 批次信息
        b_start_loc,          # 每个批次的起始位置
        b_seq_len,            # 每个批次的序列长度
        # 计算参数
        sm_scale,             # 注意力缩放因子
        n_heads,              # 注意力头总数
        num_kv_groups,        # KV分组数量
        # Q张量的内存步长（stride）信息
        q.stride(0),          # 序列维度步长
        q.stride(1),          # 头维度步长
        q.stride(2),          # 特征维度步长
        # K张量的内存步长信息
        k.stride(0),          # 序列维度步长
        k.stride(1),          # 头维度步长
        k.stride(2),          # 特征维度步长
        # V张量的内存步长信息
        v.stride(0),          # 序列维度步长
        v.stride(1),          # 头维度步长
        v.stride(2),          # 特征维度步长
        # 输出张量的内存步长信息
        output.stride(0),     # 序列维度步长
        output.stride(1),     # 头维度步长
        output.stride(2),     # 特征维度步长
        # 编译时常量参数
        HEAD_DIM=HEAD_DIM,           # 注意力头的特征维度
        BLOCK_M_SIZE=BLOCK_SIZE,     # M维度的块大小，可启用autotune自动优化
        BLOCK_N_SIZE=BLOCK_SIZE,     # N维度的块大小  
        # Triton内核执行参数
        num_warps=num_warps,         # 线程束数量
        num_stages=num_stages,       # 流水线阶段数
    )
    return output


# --------------------------------------
# 标准注意力机制实现（纯PyTorch版本，用于对比验证）
# --------------------------------------
def _naive_attention(q, k, v):
    """
    标准的朴素注意力实现
    
    这是注意力机制的标准实现，用于验证Flash Attention的正确性。
    直接计算完整的注意力矩阵，内存消耗为O(N²)。
    
    计算步骤：
    1. 计算QK^T得到注意力分数矩阵
    2. 应用因果掩码（下三角矩阵）
    3. 计算softmax归一化
    4. 与V矩阵相乘得到最终输出
    
    Args:
        q: Query张量，形状[bs, seqlen, num_head, head_dim]
        k: Key张量，形状[bs, seqlen, num_head, head_dim]  
        v: Value张量，形状[bs, seqlen, num_head, head_dim]
        
    Returns:
        torch.Tensor: 注意力输出，形状与输入相同
    """
    import math

    bs, seqlen, num_head, head_dim = q.shape
    device = q.device
    
    # 创建因果掩码：下三角矩阵，确保位置i只能看到位置j<=i
    mask = 1.0 - torch.tril(
        torch.ones((seqlen, seqlen), device=device), diagonal=0
    ).unsqueeze(0).unsqueeze(0)
    mask.masked_fill_(mask.to(torch.bool), -100000000.0)  # 掩码位置设为大负数
    
    # 调整张量维度：[bs, seqlen, num_head, head_dim] -> [bs, num_head, seqlen, head_dim]
    q = q.transpose(1, 2)  
    k = k.transpose(1, 2)  
    v = v.transpose(1, 2)  
    
    # 计算注意力分数并应用缩放
    scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
    
    # 应用softmax和因果掩码
    scores = torch.nn.functional.softmax(scores.float() + mask, dim=-1).to(q.dtype)
    
    # 计算最终输出并恢复原始维度
    output = (
        torch.matmul(scores, v)
        .transpose(1, 2)
        .contiguous()
        .reshape(bs, seqlen, num_head, head_dim)
    )
    return output


def _sdpa(q, k, v):
    """
    使用PyTorch内置的缩放点积注意力（Scaled Dot-Product Attention）
    
    这是PyTorch 2.0+提供的高效注意力实现，通常基于Flash Attention算法。
    相比朴素实现，具有更好的内存效率和计算性能。
    
    特性：
    - 自动选择最优的注意力算法（Flash Attention、memory-efficient attention等）
    - 支持因果掩码的高效实现
    - 针对不同硬件平台的优化
    
    Args:
        q: Query张量，形状[bs, seqlen, num_head, head_dim]
        k: Key张量，形状[bs, seqlen, num_head, head_dim]
        v: Value张量，形状[bs, seqlen, num_head, head_dim]
        
    Returns:
        torch.Tensor: 注意力输出，形状与输入相同
    """
    bs, seqlen, num_head, head_dim = q.shape
    
    # 调整维度以符合SDPA接口要求：[bs, num_head, seqlen, head_dim]
    q = q.transpose(1, 2)  
    k = k.transpose(1, 2)  
    v = v.transpose(1, 2)  
    
    # 使用PyTorch内置的缩放点积注意力，自动应用因果掩码
    output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    # 恢复原始维度：[bs, seqlen, num_head, head_dim]
    output = output.transpose(1, 2).contiguous().reshape(bs, seqlen, num_head, head_dim)
    return output


def standard_attention_prefill(q, k, v, b_start_loc, b_seq_len, sdpa=True):
    """
    标准注意力机制的预填充（Prefill）阶段实现
    
    用于处理变长序列批次的注意力计算，支持两种实现方式：
    1. PyTorch内置的SDPA（推荐）
    2. 朴素的手工实现（用于调试和验证）
    
    预填充阶段特点：
    - 处理完整的输入序列（而非逐个token）
    - 一次性计算所有位置的注意力
    - 主要用于训练和推理的第一步
    
    变长序列处理：
    - 根据b_start_loc和b_seq_len切分批次
    - 避免无效的填充计算
    - 每个序列独立处理注意力
    
    Args:
        q: Query张量，形状[total_tokens, num_heads, head_dim]
        k: Key张量，形状[total_tokens, num_heads, head_dim]
        v: Value张量，形状[total_tokens, num_heads, head_dim]
        b_start_loc: 每个批次的起始位置，形状[batch_size]
        b_seq_len: 每个批次的序列长度，形状[batch_size]
        sdpa: 是否使用PyTorch内置SDPA，默认True
        
    Returns:
        torch.Tensor: 注意力输出，形状与q相同
        
    示例:
        >>> # 2个序列，长度分别为5和3，总共8个token
        >>> q = torch.randn(8, 4, 64)  # [total_tokens, heads, head_dim]
        >>> k = torch.randn(8, 4, 64)
        >>> v = torch.randn(8, 4, 64)
        >>> b_start_loc = torch.tensor([0, 5])  # 第一个序列从0开始，第二个从5开始
        >>> b_seq_len = torch.tensor([5, 3])    # 序列长度分别为5和3
        >>> output = standard_attention_prefill(q, k, v, b_start_loc, b_seq_len)
    """
    out = torch.empty_like(q)  # 创建输出张量
    Z = b_start_loc.shape[0]   # 批次数量
    
    # 遍历每个批次，分别计算注意力
    for i in range(Z):
        start = b_start_loc[i]              # 当前批次起始位置
        end = start + b_seq_len[i]          # 当前批次结束位置
        
        # 提取当前批次的数据并添加batch维度
        qi = q[start:end].unsqueeze(0)      # [1, seq_len, heads, head_dim]
        ki = k[start:end].unsqueeze(0)      # [1, seq_len, heads, head_dim]
        vi = v[start:end].unsqueeze(0)      # [1, seq_len, heads, head_dim]
        
        # 根据选择使用不同的注意力实现
        if sdpa:
            oi = _sdpa(qi, ki, vi)          # 使用PyTorch内置SDPA
        else:
            oi = _naive_attention(qi, ki, vi)  # 使用朴素实现
            
        # 将结果存储到对应位置
        out[start:end] = oi.squeeze(0)
        
    return out


# =============================================================================
# Flash Attention内核性能测试与精度验证模块
# =============================================================================
def run_flash_attention2_no_pad_benchmark(
    batch=4,                                    # 批次大小
    n_heads=32,                                 # 注意力头数量
    head_dim=128,                               # 每个头的特征维度
    max_seq_len_list=[1024, 2048, 4096]        # 测试的序列长度列表
):
    """
    Flash Attention v2 NoP 的性能基准测试与精度验证
    
    这个函数执行两个主要任务：
    1. 精度验证：对比Flash Attention与标准实现的输出差异
    2. 性能基准测试：测量不同序列长度下的执行时间
    
    测试场景：
    - 模拟真实的批处理场景，包含变长序列
    - 测试多种序列长度以评估算法的伸缩性
    - 对比GPU优化实现与标准PyTorch实现的性能差异
    
    精度验证原理：
    - Flash Attention使用在线softmax等数值技巧，理论上与标准实现完全等价
    - 由于浮点运算的数值误差，允许微小的差异（通常<1e-3）
    - 验证确保算法实现的正确性
    
    性能测试方法：
    - 使用CUDA事件进行精确的GPU时间测量
    - 多次迭代取平均值以减少测量噪声
    - 预热GPU以确保稳定的性能表现
    - 生成性能对比图表以可视化结果
    
    Args:
        batch: 批次大小，影响GPU利用率
        n_heads: 注意力头数量，影响并行度
        head_dim: 头维度，影响计算复杂度
        max_seq_len_list: 测试序列长度列表，评估伸缩性
        
    Returns:
        dict: 包含测试结果的字典
            - max_seq_len_list: 测试的序列长度
            - flash_times: Flash Attention的执行时间
            - standard_times: 标准实现的执行时间
            
    示例:
        >>> # 运行基准测试
        >>> results = run_flash_attention2_no_pad_benchmark(
        ...     batch=2, n_heads=16, head_dim=64, 
        ...     max_seq_len_list=[512, 1024, 2048]
        ... )
        >>> print(f"Flash Attention平均加速比: {np.mean(results['standard_times']) / np.mean(results['flash_times']):.2f}x")
    """
    # =============================================================================
    # 第一部分：算法精度验证
    # =============================================================================
    device = "cuda"
    # 计算注意力缩放因子，包含log2转换以适配triton的exp2函数
    sm_scale = 1.0 / math.sqrt(head_dim) * 1.4426950408889634  # 1/sqrt(d) * log2(e)
    max_seq_len = max_seq_len_list[0]  # 使用第一个序列长度进行精度验证

    # 创建测试数据：模拟真实的变长序列批次
    # 总token数 = batch * max_seq_len，但实际序列长度可能不同
    shape = (batch * max_seq_len, n_heads, head_dim)
    q = torch.randn(shape, device=device, dtype=torch.float16)
    k = torch.randn(shape, device=device, dtype=torch.float16)
    v = torch.randn(shape, device=device, dtype=torch.float16)
    
    # 设置变长序列：不同批次有不同的序列长度
    b_seq_len = torch.tensor([512, 1024, 512, 1024], dtype=torch.int32, device="cuda")
    b_start_loc = torch.tensor([0, 512, 1536, 2048], dtype=torch.int32, device="cuda")

    # 运行Flash Attention实现
    triton_output = flash_attention2_no_pad(
        q, k, v, sm_scale, b_start_loc, b_seq_len, max_seq_len
    )
    
    # 运行标准实现作为参考
    torch_output = standard_attention_prefill(
        q, k, v, b_start_loc, b_seq_len, sdpa=False
    )
    
    # 计算并报告精度差异
    max_diff = torch.max(torch.abs(torch_output - triton_output))
    print(f"Flash Attention与标准实现的最大绝对误差: {max_diff}")
    print(f"相对误差: {max_diff / torch.max(torch.abs(torch_output)) * 100:.6f}%")

    # =============================================================================
    # 第二部分：性能基准测试
    # =============================================================================
    flash_times = []      # Flash Attention执行时间列表
    standard_times = []   # 标准实现执行时间列表
    iterations = 50       # 每个测试的迭代次数

    print("\n开始性能基准测试...")
    
    # 遍历不同的序列长度进行性能测试
    for seq_len in max_seq_len_list:
        print(f"\n测试序列长度: {seq_len}")
        
        # 为当前序列长度创建测试数据
        shape = (batch * seq_len, n_heads, head_dim)
        q = torch.randn(shape, device=device, dtype=torch.float16)
        k = torch.randn(shape, device=device, dtype=torch.float16)
        v = torch.randn(shape, device=device, dtype=torch.float16)

        # 构造批次信息：假设每个批次的序列长度相等
        b_start_loc = torch.tensor(
            [0, seq_len, 2 * seq_len, 3 * seq_len], dtype=torch.int32, device="cuda"
        )
        b_seq_len = torch.full((batch,), seq_len, device=device, dtype=torch.int32)

        # === Flash Attention性能测试 ===
        # 预热：确保GPU内核加载和缓存预热
        _ = flash_attention2_no_pad(q, k, v, sm_scale, b_start_loc, b_seq_len, seq_len)
        torch.cuda.synchronize()  # 等待GPU完成所有操作
        
        # 开始计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        # 多次执行以获得稳定的平均时间
        for _ in range(iterations):
            _ = flash_attention2_no_pad(
                q, k, v, sm_scale, b_start_loc, b_seq_len, seq_len
            )
        
        end_event.record()
        torch.cuda.synchronize()
        flash_time = start_event.elapsed_time(end_event) / iterations  # 平均执行时间
        flash_times.append(flash_time)

        # === 标准Attention性能测试 ===
        # 预热
        _ = standard_attention_prefill(q, k, v, b_start_loc, b_seq_len)
        torch.cuda.synchronize()
        
        # 计时
        start_event.record()
        for _ in range(iterations):
            _ = standard_attention_prefill(q, k, v, b_start_loc, b_seq_len)
        end_event.record()
        torch.cuda.synchronize()
        standard_time = start_event.elapsed_time(end_event) / iterations
        standard_times.append(standard_time)

        # 输出当前测试结果
        speedup = standard_time / flash_time
        print(f"  Flash Attention: {flash_time:.3f} ms")
        print(f"  标准实现: {standard_time:.3f} ms") 
        print(f"  加速比: {speedup:.2f}x")

    # =============================================================================
    # 第三部分：生成性能对比图表（可选）
    # =============================================================================
    try:
        import matplotlib.pyplot as plt
        
        # 创建性能对比图表
        plt.figure(figsize=(10, 6))
        plt.plot(max_seq_len_list, flash_times, marker="o", linewidth=2, 
                label="Flash Attention v2", color='blue')
        plt.plot(max_seq_len_list, standard_times, marker="s", linewidth=2,
                label="标准PyTorch实现", color='red')
        
        # 设置图表标签和标题
        plt.xlabel("序列长度 (KV缓存长度)", fontsize=12)
        plt.ylabel("平均执行时间 (ms)", fontsize=12)
        plt.title("预填充阶段性能对比", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # 添加性能提升标注
        for i, (seq_len, flash_time, std_time) in enumerate(zip(max_seq_len_list, flash_times, standard_times)):
            speedup = std_time / flash_time
            plt.annotate(f'{speedup:.1f}x', 
                        xy=(seq_len, flash_time), 
                        xytext=(10, 10), 
                        textcoords='offset points',
                        fontsize=9, 
                        color='blue')
        
        # 保存图表
        plt.tight_layout()
        plt.savefig("./flashattentionv2_nopad_benchmark.png", dpi=300, bbox_inches='tight')
        print(f"\n性能对比图表已保存为: flashattentionv2_nopad_benchmark.png")
        
    except ImportError:
        print("\n警告: matplotlib未安装，跳过图表生成")

    # 返回测试结果
    return {
        "max_seq_len_list": max_seq_len_list,  # 测试的序列长度列表
        "flash_times": flash_times,            # Flash Attention执行时间
        "standard_times": standard_times,      # 标准实现执行时间
        "average_speedup": sum(s/f for s, f in zip(standard_times, flash_times)) / len(flash_times)  # 平均加速比
    }


# =============================================================================
# 主程序入口：执行验证与性能测试
# =============================================================================
if __name__ == "__main__":
    """
    主程序入口，当直接运行该脚本时执行基准测试
    
    执行流程：
    1. 运行Flash Attention的精度验证
    2. 进行多种序列长度的性能基准测试  
    3. 输出详细的测试报告
    4. 生成性能对比图表（如果matplotlib可用）
    
    测试配置：
    - 批次大小: 4
    - 注意力头数: 32  
    - 头维度: 128
    - 测试序列长度: [1024, 2048, 4096]
    
    预期结果：
    - Flash Attention在长序列上应显示显著的性能优势
    - 精度误差应在可接受范围内（通常 < 1e-3）
    - 内存使用应显著低于标准实现
    """
    print("=" * 80)
    print("Flash Attention v2 NoP 基准测试")
    print("=" * 80)
    
    # 执行基准测试
    stats = run_flash_attention2_no_pad_benchmark()
    
    # 输出测试总结
    print("\n" + "=" * 80)
    print("测试总结:")
    print(f"平均加速比: {stats['average_speedup']:.2f}x")
    print(f"测试序列长度: {stats['max_seq_len_list']}")
    print(f"Flash Attention时间 (ms): {[f'{t:.3f}' for t in stats['flash_times']]}")
    print(f"标准实现时间 (ms): {[f'{t:.3f}' for t in stats['standard_times']]}")
    print("=" * 80)
