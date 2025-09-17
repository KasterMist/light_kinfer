"""
Flash Decoding 实现模块

本模块实现了基于Triton的Flash Decoding算法，专门优化大语言模型的解码(Decode)阶段。
Flash Decoding通过分阶段处理和分块计算，显著提高了长序列解码的效率。

核心思想：
1. 两阶段计算：将注意力计算分为Stage1(分块计算)和Stage2(结果归约)
2. 分块处理：将长的KV缓存分成多个分区并行处理
3. 在线归约：使用数值稳定的在线算法合并各分区结果
4. 内存优化：避免存储完整的注意力矩阵，降低内存占用

应用场景：
- 大语言模型的自回归解码
- 长上下文对话系统
- 批量推理优化

算法优势：
- 支持超长序列(>100K tokens)的高效处理
- 显著降低解码阶段的计算延迟
- 保持与标准注意力完全相同的数学结果
"""

import triton, torch           # Triton GPU编程框架和PyTorch深度学习框架
import triton.language as tl  # Triton DSL语言，用于编写GPU内核
from torch.amp import custom_fwd  # 自动混合精度训练的自定义前向传播装饰器


@triton.jit
def _flash_decoding_stage1_kernel(
    Q,                          # Query张量指针，形状[batch_size, num_heads, head_dim]
    K,                          # Key缓存张量指针，形状[total_tokens, num_kv_heads, head_dim]
    V,                          # Value缓存张量指针，形状[total_tokens, num_kv_heads, head_dim]
    qk_scale,                   # 注意力缩放因子，通常为1/sqrt(head_dim)
    b_req_tokens_table,         # 批次token索引表，记录每个批次在KV缓存中的token位置
    B_Seqlen,                   # 每个批次的序列长度
    num_kv_groups,              # KV分组数量，用于Grouped Query Attention
    Mid_O,                      # 中间输出张量，存储各分区的注意力结果
    Mid_O_LogExpSum,            # 中间log-exp-sum张量，存储各分区的归一化因子
    # 张量步长参数
    stride_req_to_tokens_b,     # token表在批次维度的步长
    stride_req_to_tokens_s,     # token表在序列维度的步长
    q_bs_stride,                # Q张量批次维度步长
    q_heads_stride,             # Q张量头维度步长
    q_dim_stride,               # Q张量特征维度步长
    k_bs_stride,                # K张量批次维度步长
    k_heads_stride,             # K张量头维度步长
    k_dim_stride,               # K张量特征维度步长
    v_bs_stride,                # V张量批次维度步长
    v_heads_stride,             # V张量头维度步长
    v_dim_stride,               # V张量特征维度步长
    mido_batch_stride,          # 中间输出批次维度步长
    mido_heads_stride,          # 中间输出头维度步长
    mido_partitions_stride,     # 中间输出分区维度步长
    mido_dim_stride,            # 中间输出特征维度步长
    mido_les_batch_stride,      # 中间log-exp-sum批次维度步长
    mido_les_heads_stride,      # 中间log-exp-sum头维度步长
    mido_les_partitions_stride, # 中间log-exp-sum分区维度步长
    # 编译时常量
    BLOCK_SEQ: tl.constexpr,    # 序列分块大小，默认128
    BLOCK_N: tl.constexpr,      # 内部处理块大小，默认32
    BLOCK_DMODEL: tl.constexpr, # 模型维度块大小
):
    """
    Flash Decoding 第一阶段内核：分块注意力计算
    
    这是Flash Decoding算法的核心部分，负责将长序列的KV缓存分成多个分区，
    每个分区独立计算与Query的注意力，产生中间结果供第二阶段归约使用。
    
    算法流程：
    1. 确定当前线程块负责的分区范围
    2. 将分区内的KV序列进一步分成小块处理
    3. 对每个小块计算注意力分数和加权输出
    4. 使用在线softmax算法累积结果
    5. 存储分区的中间输出和归一化因子
    
    关键优化：
    - 分块处理避免一次性加载整个KV缓存
    - 在线softmax保证数值稳定性
    - 并行处理多个分区提高GPU利用率
    """
    # 获取当前程序块的三维索引：(批次，头，序列分区)
    batch_pid = tl.program_id(0)        # 当前处理的批次索引
    head_pid = tl.program_id(1)         # 当前处理的注意力头索引
    seq_block_pid = tl.program_id(2)    # 当前处理的序列分区索引
    kv_head_pid = head_pid // num_kv_groups  # 对应的KV头索引（用于GQA）

    # 获取当前批次的序列信息
    cur_batch_seq_len = tl.load(B_Seqlen + batch_pid)                              # 当前批次的实际序列长度
    cur_req_start_loc = tl.load(b_req_tokens_table + stride_req_to_tokens_b * batch_pid)  # 当前批次在KV缓存中的起始位置

    # 计算当前分区要处理的序列范围
    cur_batch_partition_start_index = seq_block_pid * BLOCK_SEQ                     # 分区起始索引
    cur_batch_partition_end_index = tl.minimum(                                    # 分区结束索引（不超过序列长度）
        cur_batch_seq_len, cur_batch_partition_start_index + BLOCK_SEQ
    )

    # 计算当前分区需要处理的小块数量
    # 如果分区为空（起始>=结束），则块数为0；否则按BLOCK_N大小向上取整
    num_blocks = tl.where(
        cur_batch_partition_end_index - cur_batch_partition_start_index <= 0,
        0,                                                              # 分区为空时不处理
        (cur_batch_partition_end_index - cur_batch_partition_start_index + BLOCK_N - 1)
        // BLOCK_N,                                                     # 向上取整计算块数
    )

    # 初始化内存访问的偏移向量
    # 这些偏移向量用于并行访问张量的不同位置
    offs_n = cur_batch_partition_start_index + tl.arange(0, BLOCK_N)   # KV序列维度的偏移[start, start+1, ..., start+BLOCK_N-1]
    offs_d = tl.arange(0, BLOCK_DMODEL)                                # 特征维度的偏移[0, 1, ..., BLOCK_DMODEL-1]

    # 计算Q和K张量的内存访问偏移
    # 这些偏移量用于定位张量中的具体元素位置
    q_offs = batch_pid * q_bs_stride + head_pid * q_heads_stride + offs_d * q_dim_stride  # Q的偏移：定位到当前批次、当前头、所有特征维度
    k_offs = kv_head_pid * k_heads_stride + offs_d[None, :] * k_dim_stride               # K的偏移：定位到对应KV头、所有特征维度（序列维度稍后动态计算）

    # 加载Query向量（在解码阶段，Q的序列长度为1）
    q_ptrs = Q + q_offs     # 计算Q张量的内存地址
    q = tl.load(q_ptrs)     # 加载Query数据，形状：[BLOCK_DMODEL]

    # 初始化在线softmax算法的状态变量
    # 在线softmax用于在遍历KV块的过程中动态维护softmax的分子和分母
    # 这种方法避免了存储完整的注意力矩阵，显著降低内存使用
    d_i = 0.0                                                           # 归一化因子（softmax分母的累积值）
    m_i = -float("inf")                                                 # 当前最大值（用于数值稳定的softmax计算）
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)                   # 累积的注意力输出

    # 主循环：遍历当前分区内的所有KV小块
    # 这是Flash Decoding的核心计算循环，每次处理BLOCK_N个KV位置
    for start_n in range(0, num_blocks, 1):
        # 计算当前小块在分区内的绝对位置索引
        offs_n_new = offs_n + start_n * BLOCK_N                        # 当前小块的绝对位置索引
        
        # 从token索引表中获取实际的KV缓存位置
        # 这是Flash Decoding的关键创新：通过索引表间接访问KV缓存
        # 支持非连续存储的KV缓存（如PagedAttention中的分页存储）
        k_loc = tl.load(
            b_req_tokens_table + stride_req_to_tokens_b * batch_pid + offs_n_new,
            mask=offs_n_new < cur_batch_partition_end_index,            # 确保不越界访问
            other=0.0,                                                  # 越界位置填充0
        )
        
        # 计算KV张量的实际内存地址
        # k_loc包含了实际的token位置，k_offs包含了头和特征维度的偏移
        k_ptrs = k_loc[:, None] * k_bs_stride + k_offs
        
        # 创建有效性掩码，防止访问超出分区边界的数据
        k_mask = offs_n_new < cur_batch_partition_end_index             # 形状：[BLOCK_N]

        # 从KV缓存中加载当前小块的Key和Value数据
        # 使用掩码确保只加载有效数据，无效位置填充0
        k = tl.load(K + k_ptrs, mask=k_mask[:, None], other=0.0)       # 形状：[BLOCK_N, BLOCK_DMODEL]
        v = tl.load(V + k_ptrs, mask=k_mask[:, None], other=0.0)       # 形状：[BLOCK_N, BLOCK_DMODEL]

        # 计算注意力分数：qk^T / sqrt(head_dim)
        # q的形状：[BLOCK_DMODEL]，k的形状：[BLOCK_N, BLOCK_DMODEL]
        # 通过广播和逐元素乘法计算qk^T，结果形状：[BLOCK_N]
        qk = tl.sum(q[None, :] * k, axis=1)  # [BLOCK_N]
        qk *= qk_scale                       # 应用缩放因子
        qk = tl.where(k_mask, qk, float("-inf"))  # 掩码无效位置为负无穷，确保softmax后为0

        # 在线softmax算法核心：数值稳定的softmax计算
        # 1. 更新全局最大值，防止指数运算溢出
        current_max = tl.max(qk)                # 当前块的最大注意力分数
        m_ij = tl.maximum(m_i, current_max)     # 更新全局最大值
        p = tl.exp(qk - m_ij)                   # 计算当前块的softmax分子项（未归一化）

        # 2. 更新归一化项（softmax分母）
        # 需要对之前累积的结果进行重新缩放以保持数值稳定性
        alpha = tl.exp(m_i - m_ij)              # 之前结果的重新缩放因子
        d_i = alpha * d_i + tl.sum(p, axis=0)  # 更新归一化项累积值

        # 3. 更新注意力输出累积器
        # 将当前块的加权输出累加到总输出中，同时重新缩放之前的结果
        acc = alpha * acc + tl.sum(p[:, None] * v, axis=0)  # [BLOCK_DMODEL]
        # 等价写法：acc = acc * alpha + tl.dot(p, v)  # [BLOCK_DMODEL]

        # 4. 更新全局最大值记录
        m_i = m_ij

    # 存储分区计算结果到中间张量
    # 只有当分区内包含有效数据时才进行存储
    need_store = tl.where(num_blocks == 0, 0, 1)    # 检查是否需要存储
    
    # 计算中间输出张量的存储地址偏移
    off_mid_o = (
        batch_pid * mido_batch_stride                # 批次维度偏移
        + head_pid * mido_heads_stride               # 注意力头维度偏移
        + seq_block_pid * mido_partitions_stride     # 分区维度偏移
        + offs_d * mido_dim_stride                   # 特征维度偏移
    )

    # 计算中间log-exp-sum张量的存储地址偏移
    off_mid_o_les = (
        batch_pid * mido_les_batch_stride            # 批次维度偏移
        + head_pid * mido_les_heads_stride           # 注意力头维度偏移
        + seq_block_pid * mido_les_partitions_stride # 分区维度偏移
    )

    # 存储当前分区的计算结果
    # 使用循环确保只在需要时进行存储操作
    for _ in range(0, need_store, 1):
        # 存储归一化后的注意力输出（分区内的最终结果）
        tl.store(Mid_O + off_mid_o, acc / d_i)
        # 存储log-sum-exp值，用于后续跨分区的归一化
        # m_i + log(d_i) 就是当前分区的 log(sum(exp(qk_scores - max_score)))
        tl.store(Mid_O_LogExpSum + off_mid_o_les, m_i + tl.log(d_i))


@torch.no_grad()
def flash_decode_stage1(
    q,                          # Query张量，形状：[batch_size, num_heads, head_dim]
    k,                          # Key缓存张量，形状：[total_tokens, num_kv_heads, head_dim]  
    v,                          # Value缓存张量，形状：[total_tokens, num_kv_heads, head_dim]
    qk_scale,                   # 注意力缩放因子，通常为1/sqrt(head_dim)
    b_req_tokens_table,         # 批次token索引表，形状：[batch_size, max_seq_len]
    b_seq_len,                  # 每个批次的实际序列长度，形状：[batch_size]
    max_actual_seq_len,         # 当前批次中的最大实际序列长度
    mid_o,                      # 中间输出张量，形状：[batch, heads, num_partitions, head_dim]
    mid_o_logexpsum,            # 中间log-exp-sum张量，形状：[batch, heads, num_partitions]
    PARTITION_SIZE,             # 分区大小，通常为128或256
):
    """
    Flash Decoding 第一阶段：分区并行注意力计算
    
    功能描述：
    - 将长序列的KV缓存分割成多个固定大小的分区
    - 每个分区独立计算与Query的注意力，产生中间结果
    - 使用在线softmax算法保证数值稳定性
    - 为第二阶段的结果归约准备中间数据
    
    算法原理：
    1. 分区策略：将长度为N的KV序列分成ceil(N/PARTITION_SIZE)个分区
    2. 并行计算：每个GPU线程块负责一个分区的注意力计算
    3. 在线softmax：在分区内使用在线算法累积softmax结果
    4. 中间存储：保存分区输出和归一化因子供后续使用
    
    参数说明：
    - Mid_O输出形状：[batch, heads, ceil(max_seq_len/PARTITION_SIZE), head_dim]
    - Mid_O_LogExpSum输出形状：[batch, heads, ceil(max_seq_len/PARTITION_SIZE)]
    
    性能特点：
    - 内存访问：通过分区减少单次内存访问量
    - 并行度：三维并行网格(batch, heads, partitions)
    - 数值稳定：在线softmax防止数值溢出
    """
    # 设置内部块大小，必须能整除分区大小以确保完整覆盖
    BLOCK_N_SIZE = 16

    # 验证分区大小和块大小的兼容性
    assert PARTITION_SIZE % BLOCK_N_SIZE == 0, (
        "PARTITION_SIZE 必须是 BLOCK_N_SIZE 的倍数，确保分区可以被完整分块处理"
    )

    # 获取输入张量的维度信息
    # 在解码阶段，q张量的第一维实际上就是batch_size（因为seq_len=1）
    batchs, num_heads, head_dim = q.shape
    
    # 配置三维并行网格：相比标准FlashAttention增加了序列分区维度
    # 这使得可以并行处理不同的序列分区，提高长序列的处理效率
    grid = (
        batchs,                                                                      # 批次维度并行
        num_heads,                                                                   # 注意力头维度并行  
        triton.cdiv(max_actual_seq_len + PARTITION_SIZE - 1, PARTITION_SIZE),       # 序列分区维度并行
    )
    
    # 计算KV分组数量，用于Grouped Query Attention (GQA)
    # 在GQA中，多个Query头可能共享同一组Key-Value头
    num_kv_groups = q.shape[1] // k.shape[1]  # num_q_heads // num_k_heads

    # 调用第一阶段内核进行分区并行计算
    _flash_decoding_stage1_kernel[grid](
        q,                              # Query张量
        k,                              # Key缓存张量
        v,                              # Value缓存张量
        qk_scale,                       # 注意力缩放因子
        b_req_tokens_table,             # token索引表
        b_seq_len,                      # 序列长度表
        num_kv_groups,                  # KV分组数量
        mid_o,                          # 中间输出张量
        mid_o_logexpsum,                # 中间log-exp-sum张量
        *b_req_tokens_table.stride(),   # token索引表的步长参数
        *q.stride(),                    # Query张量的步长参数
        *k.stride(),                    # Key张量的步长参数
        *v.stride(),                    # Value张量的步长参数  
        *mid_o.stride(),                # 中间输出张量的步长参数
        *mid_o_logexpsum.stride(),      # 中间log-exp-sum张量的步长参数
        BLOCK_SEQ=PARTITION_SIZE,       # 分区大小编译时常量
        BLOCK_N=BLOCK_N_SIZE,           # 内部块大小编译时常量
        BLOCK_DMODEL=head_dim,          # 模型维度编译时常量
        num_warps=1,                    # 每个线程块的warp数量
        num_stages=2,                   # 流水线阶段数
    )


@triton.jit
def _flash_decoding_stage2_kernel(
    Mid_O,                      # 第一阶段的中间输出张量，形状：[batch, heads, num_partitions, head_dim]
    Mid_O_LogExpSum,            # 第一阶段的中间log-exp-sum，形状：[batch, heads, num_partitions]
    Ouput,                      # 最终注意力输出张量，形状：[batch, heads, head_dim]
    # 中间输出张量的步长参数
    mido_batch_stride,          # 中间输出批次维度步长
    mido_heads_stride,          # 中间输出头维度步长
    mido_partitions_stride,     # 中间输出分区维度步长
    mido_dim_stride,            # 中间输出特征维度步长
    # 中间log-exp-sum张量的步长参数
    mido_les_batch_stride,      # 中间log-exp-sum批次维度步长
    mido_les_heads_stride,      # 中间log-exp-sum头维度步长
    mido_les_partitions_stride, # 中间log-exp-sum分区维度步长
    # 输出张量的步长参数
    o_bs_stride,                # 输出张量批次维度步长
    o_heads_stride,             # 输出张量头维度步长
    o_dim_stride,               # 输出张量特征维度步长
    B_Seqlen,                   # 每个批次的序列长度，形状：[batch_size]
    # 编译时常量
    BLOCK_DMODEL: tl.constexpr, # 模型维度大小
    BLOCK_SEQ: tl.constexpr,    # 分区大小
):
    """
    Flash Decoding 第二阶段内核：跨分区结果归约
    
    功能描述：
    - 将第一阶段产生的多个分区结果合并成最终的注意力输出
    - 使用在线softmax算法确保跨分区的数值稳定性
    - 每个线程块处理一个(batch, head)对的所有分区
    
    算法原理：
    1. 在线归约：逐个处理各分区的中间结果
    2. 数值稳定：动态维护全局最大值和归一化因子
    3. 最终归一化：合并所有分区后进行最终的softmax归一化
    
    数学原理：
    设第i个分区的输出为O_i，log-exp-sum为L_i = m_i + log(d_i)
    最终输出 = Σ(exp(L_i - L_max) * O_i) / Σ(exp(L_i - L_max))
    其中L_max = max(L_1, L_2, ..., L_n)
    """
    # 获取当前线程块的二维索引：(批次，注意力头)
    batch_pid = tl.program_id(0)        # 当前处理的批次索引
    head_pid = tl.program_id(1)         # 当前处理的注意力头索引
    
    # 获取当前批次的实际序列长度
    cur_batch_seq_len = tl.load(B_Seqlen + batch_pid)

    # 初始化特征维度的偏移向量
    offs_d = tl.arange(0, BLOCK_DMODEL)  # [0, 1, 2, ..., BLOCK_DMODEL-1]

    # 计算中间输出张量的基础访问偏移（不包括分区维度）
    offs_part_v = batch_pid * mido_batch_stride + head_pid * mido_heads_stride + offs_d

    # 计算中间log-exp-sum张量的基础访问偏移
    offs_part_max = batch_pid * mido_les_batch_stride + head_pid * mido_les_heads_stride

    # 构造访问中间张量的指针
    part_v_ptrs = Mid_O + offs_part_v           # 中间输出的基础指针
    part_max_ptrs = Mid_O_LogExpSum + offs_part_max  # 中间log-exp-sum的基础指针

    # 初始化在线softmax归约的状态变量
    d_i = 0.0                                   # 归一化因子累积值
    m_i = -float("inf")                         # 全局最大值
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)  # 最终输出累积器

    # 计算需要处理的分区数量
    num_partitions = (cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    # 主循环：遍历所有分区进行归约
    for block_seq_n in range(0, num_partitions, 1):
        # 加载当前分区的中间输出和log-exp-sum值
        part_v = tl.load(part_v_ptrs + block_seq_n * mido_partitions_stride)  # [BLOCK_DMODEL]
        part_max = tl.load(part_max_ptrs + block_seq_n)  # 标量，注意分区维度步长为1
        part_max = tl.load(part_max_ptrs + block_seq_n)  # 标量，注意分区维度步长为1

        # 在线softmax归约的核心步骤：
        # 1. 更新全局最大值，确保数值稳定性
        m_ij = tl.maximum(part_max, m_i)            # 新的全局最大值
        
        # 2. 计算重新缩放因子
        # alpha用于重新缩放之前累积的结果，p用于缩放当前分区的结果
        alpha = tl.exp(m_i - m_ij)                  # 之前结果的缩放因子
        p = tl.exp(part_max - m_ij)                 # 当前分区的缩放因子

        # 3. 更新累积的注意力输出
        # 使用在线softmax的更新公式：新结果 = 缩放的旧结果 + 缩放的新结果
        acc = alpha * acc + p * part_v

        # 4. 更新归一化因子（softmax的分母）
        # 同样使用在线更新公式
        d_i = alpha * d_i + p

        # 5. 更新全局最大值记录
        m_i = m_ij

    # 计算最终输出的存储地址偏移
    offs_out = (
        batch_pid * o_bs_stride +                   # 批次维度偏移
        head_pid * o_heads_stride +                 # 注意力头维度偏移  
        offs_d * o_dim_stride                       # 特征维度偏移
    )
    
    # 存储最终的注意力输出：归一化后的累积结果
    tl.store(Ouput + offs_out, acc / d_i)


@torch.no_grad()
def flash_decode_stage2(
    mid_o,                      # 第一阶段的中间输出，形状：[batch, heads, num_partitions, head_dim]
    mid_o_logexpsum,            # 第一阶段的log-exp-sum值，形状：[batch, heads, num_partitions]
    atten_output,               # 最终注意力输出张量，形状：[batch, heads, head_dim]
    b_seq_len,                  # 每个批次的实际序列长度，形状：[batch_size]
    PARTITION_SIZE,             # 分区大小，与第一阶段保持一致
):
    """
    Flash Decoding 第二阶段：跨分区归约合并
    
    功能描述：
    - 将第一阶段产生的多个分区中间结果合并成最终输出
    - 使用在线softmax算法保证数值稳定的跨分区归约
    - 产生与标准注意力计算数学等价的结果
    
    算法流程：
    1. 配置二维并行网格：(batch_size, num_heads)
    2. 每个线程块处理一个(batch, head)对的所有分区
    3. 使用在线softmax逐个合并分区结果
    4. 输出最终归一化的注意力结果
    
    性能特点：
    - 内存效率：避免存储完整的注意力矩阵
    - 数值稳定：动态维护全局最大值防止溢出
    - 并行度：每个(batch, head)对独立处理
    """
    # 获取张量维度信息
    batchs, num_heads, HEAD_DIM = mid_o.shape[0], mid_o.shape[1], mid_o.shape[-1]
    
    # 配置二维并行网格：每个线程块处理一个(batch, head)对
    grid = (batchs, num_heads)

    # 调用第二阶段内核进行跨分区归约
    _flash_decoding_stage2_kernel[grid](
        mid_o,                      # 中间输出张量
        mid_o_logexpsum,            # 中间log-exp-sum张量
        atten_output,               # 最终输出张量
        *mid_o.stride(),            # 中间输出张量的步长参数
        *mid_o_logexpsum.stride(),  # 中间log-exp-sum张量的步长参数
        *atten_output.stride(),     # 输出张量的步长参数
        b_seq_len,                  # 序列长度信息（TODO: 支持PagedAttention）
        BLOCK_DMODEL=HEAD_DIM,      # 模型维度编译时常量
        BLOCK_SEQ=PARTITION_SIZE,   # 分区大小编译时常量
        num_warps=4,                # 每个线程块的warp数量
        num_stages=2,               # 流水线阶段数
    )


@torch.no_grad()
@custom_fwd(cast_inputs=torch.float16, device_type="cuda")
def flash_decoding(
    q,                          # Query张量，形状：[batch_size, num_heads, head_dim]
    k_cache,                    # Key缓存张量，形状：[total_tokens, num_kv_heads, head_dim]
    v_cache,                    # Value缓存张量，形状：[total_tokens, num_kv_heads, head_dim]
    qk_scale,                   # 注意力缩放因子，通常为1/sqrt(head_dim)
    b_req_tokens_table,         # 批次token索引表，形状：[batch_size, max_seq_len]
    b_seq_len,                  # 每个批次的实际序列长度，形状：[batch_size]
    max_actual_seq_len,         # 当前批次中的最大实际序列长度
):
    """
    Flash Decoding 主函数：高效的大语言模型解码阶段注意力计算
    
    功能描述：
    - 针对解码阶段（单token生成）的注意力机制优化实现
    - 通过两阶段算法处理超长序列的KV缓存（支持>100K tokens）
    - 相比标准注意力显著降低内存占用和计算延迟
    
    算法原理：
    1. 第一阶段：将长KV序列分区并行计算注意力
    2. 第二阶段：使用在线softmax合并各分区结果
    3. 数值稳定：全程使用在线算法避免数值溢出
    4. 内存优化：避免存储完整的注意力矩阵
    
    输入张量要求：
    - q: 解码阶段的Query（seq_len=1）
    - k_cache/v_cache: 累积的历史Key/Value缓存
    - 支持Grouped Query Attention (GQA)
    
    性能优势：
    - 长序列处理：O(N)内存复杂度，支持超长上下文
    - 解码优化：专门针对单token生成场景优化
    - 数值稳定：在线softmax保证计算精度
    
    返回：
    - atten_output: 注意力输出，形状与q相同
    """
    # 验证输入张量维度的兼容性
    assert q.shape[-1] == k_cache.shape[-1] == v_cache.shape[-1], \
        "Query, Key, Value的head_dim必须相同"
    
    # 设置分区大小：根据GPU内存容量调整
    # 较新的GPU（3090Ti以上）可以设置为256，提高并行度
    PARTITION_SIZE = 128  
    
    # 获取输入维度信息（解码阶段q的seq_len=1）
    batchs, num_heads, head_dim = q.shape

    # 计算所需的最大分区数量
    max_num_partitions = (max_actual_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE

    # 分配中间结果存储张量
    # mid_o: 存储每个分区的注意力输出结果
    mid_o = torch.empty(
        (batchs, num_heads, max_num_partitions, head_dim),
        dtype=torch.float32,  # 使用float32保证数值精度
        device=q.device,
    )
    
    # mid_o_logexpsum: 存储每个分区的log-sum-exp值，用于后续归一化
    mid_o_logexpsum = torch.empty(
        (batchs, num_heads, max_num_partitions), 
        dtype=torch.float32, 
        device=q.device
    )

    # 第一阶段：分区并行注意力计算
    # 将长序列KV缓存分成多个分区，每个分区独立计算与Query的注意力
    flash_decode_stage1(
        q,                      # Query张量
        k_cache,                # Key缓存
        v_cache,                # Value缓存
        qk_scale,               # 缩放因子
        b_req_tokens_table,     # token索引表
        b_seq_len,              # 序列长度
        max_actual_seq_len,     # 最大序列长度
        mid_o,                  # 中间输出
        mid_o_logexpsum,        # 中间归一化因子
        PARTITION_SIZE,         # 分区大小
    )

    # 第二阶段：跨分区结果归约
    # 使用在线softmax算法将各分区的中间结果合并成最终输出
    atten_output = torch.empty_like(q)  # 分配输出张量

    flash_decode_stage2(
        mid_o,                  # 第一阶段的中间输出
        mid_o_logexpsum,        # 第一阶段的归一化因子
        atten_output,           # 最终输出张量
        b_seq_len,              # 序列长度信息
        PARTITION_SIZE          # 分区大小
    )

    return atten_output


# --------------------------------------
# 标准 Attention Decode 实现（纯 PyTorch版本）
# 用于对比验证和性能基准测试
# --------------------------------------
def _naive_attention(q, k, v):
    """
    朴素的注意力计算实现
    
    参数：
    - q: Query张量，形状：[1, num_heads, head_dim]
    - k: Key张量，形状：[seq_len, num_heads, head_dim]  
    - v: Value张量，形状：[seq_len, num_heads, head_dim]
    
    返回：
    - output: 注意力输出，形状：[1, num_heads, head_dim]
    
    算法：
    1. 计算注意力分数：Q @ K^T / sqrt(head_dim)
    2. 应用softmax归一化
    3. 计算加权输出：softmax(scores) @ V
    """
    import math

    head_dim = q.shape[-1]
    # 调整张量维度以适应矩阵乘法
    q = q.transpose(0, 1)  # (num_heads, 1, head_dim)
    k = k.transpose(0, 1)  # (num_heads, seq_len, head_dim)
    v = v.transpose(0, 1)  # (num_heads, seq_len, head_dim)
    
    # 计算缩放的注意力分数
    scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
    
    # 应用softmax得到注意力权重
    scores = torch.nn.functional.softmax(scores.float(), dim=-1).to(q.dtype)
    
    # 计算加权输出并恢复原始维度
    output = (
        torch.matmul(scores, v).transpose(0, 1).contiguous()
    )  # (1, num_heads, head_dim)
    return output


def torch_attention_with_kvcache(q, k_cache, v_cache, b_start_loc, b_seq_len):
    """
    使用KV缓存的标准注意力计算（批处理版本）
    
    参数：
    - q: Query张量，形状：[batch_size, num_heads, head_dim]
    - k_cache: Key缓存，形状：[total_tokens, num_heads, head_dim]
    - v_cache: Value缓存，形状：[total_tokens, num_heads, head_dim]
    - b_start_loc: 每个批次在KV缓存中的起始位置，形状：[batch_size]
    - b_seq_len: 每个批次的序列长度，形状：[batch_size]
    
    返回：
    - out: 注意力输出，形状与q相同
    
    功能：
    为批次中的每个样本独立计算注意力，主要用于验证Flash Decoding的正确性
    """
    out = torch.empty_like(q)  # 分配输出张量
    Z = q.shape[0]  # 批次大小
    
    # 逐个处理批次中的每个样本
    for i in range(Z):
        # 确定当前样本在KV缓存中的范围
        start = b_start_loc[i]              # 起始位置
        end = start + b_seq_len[i]          # 结束位置
        
        # 提取当前样本的数据
        q_i = q[i : i + 1]                  # 当前样本的Query，形状：(1, num_heads, head_dim)
        k_i = k_cache[start:end]            # 当前样本的Key序列，形状：(seq_len, num_heads, head_dim)
        v_i = v_cache[start:end]            # 当前样本的Value序列，形状：(seq_len, num_heads, head_dim)
        
        # 计算当前样本的注意力输出
        o_i = _naive_attention(q_i, k_i, v_i)
        
        # 存储结果
        out[i : i + 1] = o_i
    return out


# ----------------------------------
# 性能对比及曲线绘制函数封装（含 Warm up）
# ----------------------------------
def plot_performance_comparison(token_sizes, warmup_iterations=10, test_iterations=50):
    """
    性能对比测试函数：Flash Decoding vs 标准 Attention
    
    功能描述：
    对不同token长度下的Flash Decoding与标准Attention进行性能测试，
    并绘制性能对比曲线，用于分析算法在不同序列长度下的性能表现。

    参数说明：
    - token_sizes: list[int]，要测试的不同KV缓存长度
    - warmup_iterations: int，预热迭代次数，用于稳定GPU状态
    - test_iterations: int，正式测试迭代次数，用于计算平均性能
    
    测试流程：
    1. 为每个token_size构造测试数据
    2. 预热GPU内核以获得稳定的性能测量
    3. 测量Flash Decoding和标准Attention的执行时间
    4. 绘制性能对比曲线并保存
    
    输出：
    - 控制台打印各长度下的性能数据
    - 保存性能对比图表到flashdecoding_benchamrk.png
    """
    # 导入绘图库（在函数内导入避免全局依赖）
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("警告：matplotlib未安装，无法绘制性能曲线")
        return

    # 测试环境配置
    device = torch.device("cuda")
    batch = 4               # 批次大小
    num_heads = 32          # 注意力头数量
    head_dim = 64           # 每个头的维度
    qk_scale = 1.0 / (head_dim**0.5)  # 注意力缩放因子
    
    # 构造固定的Query张量（解码阶段seq_len=1）
    q = torch.randn(batch * 1, num_heads, head_dim, device=device)

    # 存储性能测试结果
    flash_times = []        # Flash Decoding的执行时间
    standard_times = []     # 标准Attention的执行时间

    # 遍历不同的token长度进行测试
    for tokens in token_sizes:
        print(f"\n测试 token size: {tokens}")
        
        # 构造测试用的KV缓存数据
        k_cache = torch.randn(batch * tokens, num_heads, head_dim, device=device)
        v_cache = torch.randn(batch * tokens, num_heads, head_dim, device=device)
        
        # 构造token索引表（用于Flash Decoding的间接内存访问）
        b_req_tokens_table = torch.arange(
            0, tokens, device=device, dtype=torch.int32
        ).repeat(batch, 1)
        
        # 构造批次信息（用于标准Attention的批处理）
        b_start_loc = torch.tensor(
            [0, tokens, 2 * tokens, 3 * tokens], 
            dtype=torch.int32, 
            device="cuda"
        )  # 假设batch=4
        b_seq_len = torch.full((batch,), tokens, device=device, dtype=torch.int32)
        max_actual_seq_len = tokens

        # === Flash Decoding性能测试 ===
        # 预热阶段：让GPU内核充分"热身"以获得稳定的性能数据
        for _ in range(warmup_iterations):
            _ = flash_decoding(
                q, k_cache, v_cache, qk_scale,
                b_req_tokens_table, b_seq_len, max_actual_seq_len,
            )
        
        # 正式性能测试
        torch.cuda.synchronize()  # 确保所有CUDA操作完成
        flash_start = torch.cuda.Event(enable_timing=True)
        flash_end = torch.cuda.Event(enable_timing=True)
        
        flash_start.record()
        for _ in range(test_iterations):
            _ = flash_decoding(
                q, k_cache, v_cache, qk_scale,
                b_req_tokens_table, b_seq_len, max_actual_seq_len,
            )
        flash_end.record()
        torch.cuda.synchronize()
        
        # 计算Flash Decoding的平均执行时间
        flash_avg = flash_start.elapsed_time(flash_end) / test_iterations
        flash_times.append(flash_avg)
        print(f"Flash Decoding 平均时间: {flash_avg:.3f} ms")

        # === 标准Attention性能测试 ===
        # 预热阶段
        for _ in range(warmup_iterations):
            _ = torch_attention_with_kvcache(
                q, k_cache, v_cache, b_start_loc, b_seq_len
            )
        
        # 正式性能测试
        torch.cuda.synchronize()
        std_start = torch.cuda.Event(enable_timing=True)
        std_end = torch.cuda.Event(enable_timing=True)
        
        std_start.record()
        for _ in range(test_iterations):
            _ = torch_attention_with_kvcache(
                q, k_cache, v_cache, b_start_loc, b_seq_len
            )
        std_end.record()
        torch.cuda.synchronize()
        
        # 计算标准Attention的平均执行时间
        std_avg = std_start.elapsed_time(std_end) / test_iterations
        standard_times.append(std_avg)
        print(f"Standard Attention 平均时间: {std_avg:.3f} ms")

    # === 绘制性能对比曲线 ===
    plt.figure(figsize=(10, 6))
    plt.plot(token_sizes, flash_times, marker="o", linewidth=2, label="Flash Decoding", color='blue')
    plt.plot(token_sizes, standard_times, marker="s", linewidth=2, label="Standard Attention", color='red')
    
    # 图表美化
    plt.xlabel("Token Size (KV Cache Length)", fontsize=12)
    plt.ylabel("Average Execution Time (ms)", fontsize=12)
    plt.title("Performance Comparison: Flash Decoding vs Standard Attention", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数坐标更好地显示性能差异
    
    # 保存图表
    plt.tight_layout()
    plt.savefig("./flashdecoding_benchamrk.png", dpi=300, bbox_inches='tight')
    print(f"\n性能对比图表已保存至: ./flashdecoding_benchamrk.png")


# -------------------------------
# 验证输出和调用性能对比函数
# -------------------------------
def main():
    """
    主测试函数：验证Flash Decoding的正确性并进行性能对比
    
    功能描述：
    1. 验证算法正确性：比较Flash Decoding与标准Attention的输出差异
    2. 性能基准测试：测试不同序列长度下的执行时间
    3. 生成性能对比图表：可视化性能差异
    
    测试流程：
    - 构造标准化的测试数据
    - 运行Flash Decoding和标准Attention
    - 验证数值精度（允许小的浮点误差）
    - 执行多种序列长度的性能测试
    - 生成性能分析报告
    """
    # 设置随机种子以确保测试结果的可重现性
    torch.manual_seed(0)
    device = torch.device("cuda")

    # 测试参数配置
    batch = 4                    # 批次大小
    num_heads = 32               # 注意力头数量
    head_dim = 64                # 每个头的维度大小
    max_tokens = 2048            # 每个请求序列的最大token长度
    qk_scale = 1.0 / (head_dim**0.5)  # 注意力缩放因子

    # 构造测试数据
    # 注意：输入张量的形状为[batch * seq_len, num_heads, head_dim]，这是三维格式，
    # 以兼容Flash Decoding内核的内存布局要求
    print("构造测试数据...")
    q = torch.randn(batch * 1, num_heads, head_dim, device=device)  # Query（解码阶段seq_len=1）
    k_cache = torch.randn(batch * max_tokens, num_heads, head_dim, device=device)  # Key缓存
    v_cache = torch.randn(batch * max_tokens, num_heads, head_dim, device=device)  # Value缓存
    
    # 构造每个请求的KV token分配的内存块索引
    # 这个索引表告诉Flash Decoding如何在KV缓存中找到每个请求的数据
    b_req_tokens_table = torch.arange(
        0, max_tokens * batch, device=device, dtype=torch.int32
    ).view(batch, max_tokens)
    
    # 构造序列长度信息
    b_seq_len = torch.full((batch,), max_tokens, device=device, dtype=torch.int32)
    
    # 构造起始位置信息（用于标准Attention的批处理）
    b_start_loc = torch.tensor(
        [0, max_tokens, 2 * max_tokens, 3 * max_tokens],
        dtype=torch.int32,
        device="cuda",
    )  # 假设batch=4的情况

    # === 算法正确性验证 ===
    print("\n执行算法正确性验证...")
    
    # 运行Flash Decoding
    flash_out = flash_decoding(
        q, k_cache, v_cache, qk_scale, 
        b_req_tokens_table, b_seq_len, max_tokens
    )
    
    # 运行标准Attention作为基准
    standard_out = torch_attention_with_kvcache(
        q, k_cache, v_cache, b_start_loc, b_seq_len
    )
    
    # 输出维度信息
    print("Flash Decoding输出形状:", flash_out.shape)
    print("Standard Attention输出形状:", standard_out.shape)
    
    # 数值精度验证
    # 由于浮点运算的精度限制，允许小的数值误差（相对误差1e-3，绝对误差1e-3）
    if torch.allclose(flash_out, standard_out, atol=1e-3, rtol=1e-3):
        print("✅ 验证通过: Flash Decoding输出与标准Attention在数值精度范围内一致。")
    else:
        # 如果验证失败，输出详细的误差信息用于调试
        diff = (flash_out - standard_out).abs().max().item()
        mean_diff = (flash_out - standard_out).abs().mean().item()
        print(f"❌ 验证失败：最大误差为 {diff:.6f}，平均误差为 {mean_diff:.6f}")
        print("这可能是由于数值精度、算法实现细节或编译优化差异造成的。")

    # === 性能基准测试 ===
    print("\n开始性能基准测试...")
    
    # 定义要测试的不同token长度
    # 从小规模到大规模，测试Flash Decoding在不同序列长度下的性能表现
    token_numbers = [64, 128, 256, 512, 1024, max_tokens]
    
    # 执行性能对比测试（包含预热和多次测试以获得稳定结果）
    plot_performance_comparison(
        token_numbers, 
        warmup_iterations=10,    # 预热次数，让GPU达到稳定状态
        test_iterations=50       # 测试次数，用于计算平均性能
    )
    
    print("\n测试完成！")
    print("详细的性能数据已显示在上方，性能对比图表已保存。")


if __name__ == "__main__":
    main()
