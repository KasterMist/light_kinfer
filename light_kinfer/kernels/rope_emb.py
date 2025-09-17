"""
RoPE (Rotary Position Embedding) 旋转位置编码的Triton实现

RoPE是一种创新的位置编码方法，被广泛应用于现代大型语言模型中，
如GPT-NeoX、PaLM、LLaMA等。相比传统的绝对位置编码，RoPE具有更好的性能。

核心思想：
RoPE将位置信息通过旋转矩阵的形式融入到query和key的表示中，
使得不同位置的向量在高维空间中保持特定的几何关系。

数学原理：
对于位置m的向量x，RoPE变换为：
[x₁']   [cos(mθ₁) -sin(mθ₁)] [x₁]
[x₂'] = [sin(mθ₁)  cos(mθ₁)] [x₂]

其中θᵢ = 10000^(-2i/d)，d是head dimension，i是维度索引。

实现方式：
1. 将head_dim分为两部分：前半部分和后半部分
2. 前半部分与cos相乘，减去后半部分与sin的乘积
3. 后半部分与cos相乘，加上前半部分与sin的乘积

优势：
- 相对位置感知：模型能够学习到相对位置关系
- 外推能力：可以处理比训练时更长的序列
- 计算效率：相比绝对位置编码，计算开销更小
- 几何特性：在高维空间中保持旋转不变性

应用场景：
- 现代Transformer模型的位置编码
- 长序列处理任务
- 需要位置感知的自回归生成
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _triton_rope_emb(
    q_ptr,                  # Query张量指针
    q_row_stride,           # Query张量行步幅
    k_ptr,                  # Key张量指针  
    k_row_stride,           # Key张量行步幅
    cos,                    # 余弦表指针
    cos_b_stride,           # 余弦表batch维度步幅
    cos_s_stride,           # 余弦表sequence维度步幅
    sin,                    # 正弦表指针
    sin_b_stride,           # 正弦表batch维度步幅
    sin_s_stride,           # 正弦表sequence维度步幅
    sl,                     # 序列长度 (sequence length)
    bs: tl.constexpr,       # 批次大小 (batch size)
    n_qh: tl.constexpr,     # Query头数量
    n_kh: tl.constexpr,     # Key头数量
    hd: tl.constexpr,       # 头维度 (head dimension)
    pad_n_qh: tl.constexpr, # 填充后的Query头数量 (2的幂次)
    pad_n_kh: tl.constexpr, # 填充后的Key头数量 (2的幂次)
    pad_hd: tl.constexpr,   # 填充后的头维度 (2的幂次)
    BLOCK_SIZE: tl.constexpr, # 线程块大小
):
    """
    RoPE旋转位置编码的Triton GPU内核
    
    这个内核同时处理Query和Key张量的RoPE变换，实现高效的位置编码。
    
    变换公式：
    对于每个位置的向量 [x₁, x₂, ..., xₙ]，分为前半部分和后半部分：
    - 前半部分: x₁, x₂, ..., x_{n/2}
    - 后半部分: x_{n/2+1}, x_{n/2+2}, ..., xₙ
    
    变换：
    - new_前半部分 = 前半部分 * cos - 后半部分 * sin
    - new_后半部分 = 后半部分 * cos + 前半部分 * sin
    
    实现特点：
    - 使用向量化操作，提高计算效率
    - 支持不同的Query和Key头数量 (GQA支持)
    - 边界掩码处理，支持任意头维度
    - 内存访问优化，减少数据传输开销
    """
    # 获取当前程序的ID，每个程序处理一个(batch, position)对
    pid = tl.program_id(0)
    batch_id = pid // sl        # 计算batch索引
    cos_row_idx = pid % sl      # 计算sequence位置索引

    # 定位到当前程序负责的Q、K数据的起始位置
    q_ptr += pid * q_row_stride
    k_ptr += pid * k_row_stride

    # 定位到对应位置的cos、sin值
    # cos和sin的形状是(batch_size, seq_len, head_dim//2)
    cos_ptr = cos + batch_id * cos_b_stride + cos_row_idx * cos_s_stride
    sin_ptr = sin + batch_id * sin_b_stride + cos_row_idx * sin_s_stride

    # 加载当前位置的cos和sin值
    # 只需要head_dim//2个值，因为RoPE是成对旋转的
    cos_offsets = tl.arange(0, pad_hd // 2)
    cos_mask = cos_offsets < hd // 2  # 边界掩码
    cos_row = tl.load(cos_ptr + cos_offsets, mask=cos_mask, other=0)
    sin_row = tl.load(sin_ptr + cos_offsets, mask=cos_mask, other=0)

    # 计算Query和Key张量中前半部分的偏移量
    # 形状: (头数, 头维度//2)
    first_half_q_offsets = (
        tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_half_k_offsets = (
        tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )

    # 创建边界掩码，确保不访问超出实际尺寸的内存
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, pad_hd // 2)[None, :] < hd // 2
    )
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, pad_hd // 2)[None, :] < hd // 2
    )

    # 加载Query和Key的前半部分数据
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(
        sin_row.dtype
    )
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(
        sin_row.dtype
    )

    # 计算后半部分的偏移量（前半部分偏移量 + head_dim//2）
    second_half_q_offsets = first_half_q_offsets + (hd // 2)
    second_half_k_offsets = first_half_k_offsets + (hd // 2)
    second_q_mask = first_q_mask  # 后半部分使用相同的掩码
    second_k_mask = first_k_mask

    # 加载Query和Key的后半部分数据
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(
        sin_row.dtype
    )
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(
        sin_row.dtype
    )

    # 应用RoPE旋转变换
    # 新的前半部分 = 前半部分 * cos - 后半部分 * sin
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    
    # 新的后半部分 = 后半部分 * cos + 前半部分 * sin
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

    # 对Key张量应用相同的变换
    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)


def rope_emb_forward(q, k, cos, sin, batch_size, seq_len):
    """
    RoPE旋转位置编码的前向传播主函数
    
    这个函数协调GPU内核的执行，为Query和Key张量应用旋转位置编码。
    
    参数：
        q (torch.Tensor): Query张量
            形状: (batch_size * seq_len, n_q_heads, head_dim)
            包含所有位置的query向量
            
        k (torch.Tensor): Key张量  
            形状: (batch_size * seq_len, n_k_heads, head_dim)
            包含所有位置的key向量
            
        cos (torch.Tensor): 预计算的余弦表
            形状: (batch_size, seq_len, head_dim // 2)
            每个位置对应的cos值
            
        sin (torch.Tensor): 预计算的正弦表
            形状: (batch_size, seq_len, head_dim // 2) 
            每个位置对应的sin值
            
        batch_size (int): 批次大小
        seq_len (int): 序列长度
        
    返回：
        tuple: (rotated_q, rotated_k)
            - rotated_q: 应用RoPE后的Query张量
            - rotated_k: 应用RoPE后的Key张量
            
    实现细节：
        1. 验证输入张量的形状一致性
        2. 计算Triton内核的最优配置参数
        3. 确保所有张量内存连续，优化GPU访问
        4. 启动GPU内核进行并行RoPE变换
        
    性能优化：
        - 使用2的幂次填充，提高内存访问效率
        - 自适应warp配置，根据head_dim调整并行度
        - 内存连续性检查，避免不必要的数据拷贝
        - 单内核同时处理Q和K，减少内核启动开销
        
    注意事项：
        - cos和sin表需要预先计算好
        - 支持GQA (Grouped Query Attention)，Q和K可以有不同的头数
        - head_dim必须是偶数，因为RoPE需要成对旋转
    """
    N, n_qh, HEAD_DIM = q.shape    # 获取Query张量的形状信息
    _, n_kh, _ = k.shape           # 获取Key张量的头数信息
    
    # 验证输入一致性
    assert N == batch_size * seq_len, f"Expected N={batch_size * seq_len}, got {N}"
    assert HEAD_DIM % 2 == 0, f"Head dimension must be even for RoPE, got {HEAD_DIM}"

    # 计算填充尺寸，确保为2的幂次以优化GPU计算
    pad_hd = triton.next_power_of_2(HEAD_DIM)      # 填充头维度
    pad_n_qh = triton.next_power_of_2(n_qh)        # 填充Query头数
    pad_n_kh = triton.next_power_of_2(n_kh)        # 填充Key头数
    BLOCK_SIZE = max(pad_n_qh, pad_n_kh)           # 选择较大的作为块大小

    # 根据头维度大小自适应调整warp数量
    # 更大的头维度需要更多的并行度来充分利用GPU资源
    if HEAD_DIM >= 128:
        num_warps = 8   # 大头维度使用更多warp
    else:
        num_warps = 4   # 小头维度使用较少warp

    # 确保所有张量在GPU内存中连续存储
    # 这对于Triton内核的高效执行至关重要
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    # 启动Triton GPU内核
    # 网格大小为N，即每个(batch, position)位置启动一个程序
    _triton_rope_emb[(N,)](
        q,                    # Query张量
        q.stride(0),          # Query的行步幅
        k,                    # Key张量
        k.stride(0),          # Key的行步幅
        cos,                  # 余弦表
        cos.stride(0),        # 余弦表的batch步幅
        cos.stride(1),        # 余弦表的sequence步幅
        sin,                  # 正弦表
        sin.stride(0),        # 正弦表的batch步幅
        sin.stride(1),        # 正弦表的sequence步幅
        seq_len,              # 序列长度
        batch_size,           # 批次大小
        n_qh,                 # Query头数
        n_kh,                 # Key头数
        HEAD_DIM,             # 头维度
        pad_n_qh,             # 填充后的Query头数
        pad_n_kh,             # 填充后的Key头数
        pad_hd,               # 填充后的头维度
        BLOCK_SIZE=BLOCK_SIZE, # 线程块大小
        num_warps=num_warps,   # warp数量
        num_stages=1,          # 流水线阶段数
    )
    
    # 返回变换后的Q和K张量
    # 注意：这里是就地修改，返回的是修改后的原张量
    return q, k
