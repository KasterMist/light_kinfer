"""
KV索引更新内核模块

这个模块实现了一个高效的KV缓存索引更新机制，用于在Transformer模型中管理键值缓存。
在多请求并行处理场景下，需要正确地将每个token的KV缓存索引映射到对应的请求位置。

主要功能：
1. 使用Triton加速的GPU内核来高效更新KV索引映射
2. 支持变长序列的批量处理
3. 确保每个token的KV缓存能正确对应到其所属的请求

典型应用场景：
- 大模型推理服务中的动态批处理
- 多轮对话中的KV缓存管理
- 变长序列的高效内存管理
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_update_kv_index(
    req_to_token_indexs,  # 输出张量的指针，形状为 (num_requests, max_seq_len)
                          # 用于存储每个请求中每个位置对应的KV缓存索引
    b_req_idx,           # decode_batch 批次中每个token所属的请求ID，形状为 (num_tokens,)
                         # 例如：[0, 0, 1, 1, 2] 表示前两个token属于请求0，接下来两个属于请求1，最后一个属于请求2
    b_seq_len,           # decode_batch 中每个token对应请求的当前序列长度，形状为 (num_tokens,)
                         # 例如：[5, 6, 3, 4, 2] 表示各token所在请求的当前长度
    select_index,        # decode_batch 中每个token在KV缓存中的全局索引，形状为 (num_tokens,)
                         # 例如：[10, 11, 12, 13, 14] 表示这些token在KV缓存池中的实际位置
    stride_req_to_token_b,  # req_to_token_indexs 在第一个维度（请求维度）的内存步幅
    stride_req_to_token_s,  # req_to_token_indexs 在第二个维度（序列长度维度）的内存步幅
):
    """
    Triton GPU内核：更新KV索引映射
    
    这个内核的核心逻辑是：对于批次中的每个token，根据它所属的请求ID和该请求的当前序列长度，
    将这个token的KV缓存索引存储到req_to_token_indexs张量的正确位置。
    
    具体步骤：
    1. 获取当前线程处理的token索引
    2. 读取这个token所属的请求ID
    3. 读取这个token的KV缓存索引
    4. 读取这个token所在请求的当前序列长度
    5. 计算在req_to_token_indexs中的目标位置：req_to_token_indexs[请求ID][序列长度-1]
    6. 将KV索引存储到目标位置
    
    为什么是序列长度-1？
    因为序列长度表示的是1-based的长度，而数组索引是0-based的，
    所以当前token应该存储在位置[序列长度-1]。
    """
    # 获取当前程序的ID，即当前线程要处理的token索引
    # 每个线程处理一个token
    cur_index = tl.program_id(0)

    # 从 b_req_idx 张量加载当前token所属的请求ID
    # 例如：如果cur_index=2，b_req_idx[2]=1，表示第3个token属于请求1
    cur_req_idx = tl.load(b_req_idx + cur_index)

    # 从 select_index 张量加载当前token在KV缓存中的全局索引
    # 例如：如果select_index[2]=15，表示这个token的KV数据存储在缓存池的第15个位置
    cur_token_index = tl.load(select_index + cur_index)

    # 从 b_seq_len 张量加载当前token所在请求的当前序列长度
    # 例如：如果b_seq_len[2]=5，表示这个token所在的请求当前有5个token
    cur_seq_len = tl.load(b_seq_len + cur_index)

    # 计算在req_to_token_indexs张量中的目标位置偏移量
    # 目标位置：req_to_token_indexs[cur_req_idx][cur_seq_len - 1]
    # - cur_req_idx * stride_req_to_token_b：定位到具体的请求行
    # - (cur_seq_len - 1) * stride_req_to_token_s：定位到该请求中的具体位置
    # 使用cur_seq_len - 1是因为当前token是该请求的第cur_seq_len个token（从1开始计数）
    # 所以在0-based索引中应该存储在位置cur_seq_len - 1
    dest_offset = (
        req_to_token_indexs
        + cur_req_idx * stride_req_to_token_b
        + (cur_seq_len - 1) * stride_req_to_token_s
    )

    # 将当前token的KV缓存索引存储到计算出的目标位置
    # 这样，req_to_token_indexs[cur_req_idx][cur_seq_len - 1] = cur_token_index
    tl.store(dest_offset, cur_token_index)

    return


@torch.no_grad()
def update_kv_index(req_to_token_indexs, b_req_idx, b_seq_len, select_index):
    """
    更新KV索引映射的主函数
    
    这个函数的核心作用是维护一个从"请求-序列位置"到"KV缓存索引"的映射关系。
    在大模型推理服务中，多个请求会被批量处理，每个请求可能有不同的序列长度，
    而所有token的KV缓存会存储在一个连续的缓存池中。这个函数负责建立正确的索引映射关系。
    
    参数说明：
        req_to_token_indexs (torch.Tensor): 
            输出张量，用于存储KV索引映射关系
            形状: (num_requests, max_seq_len)
            含义: req_to_token_indexs[i][j] 表示第i个请求的第j个位置对应的KV缓存索引
            
        b_req_idx (torch.Tensor): 
            当前批次中每个token所属的请求ID
            形状: (num_tokens,)
            含义: b_req_idx[i] 表示第i个token属于哪个请求
            
        b_seq_len (torch.Tensor): 
            当前批次中每个token对应请求的序列长度
            形状: (num_tokens,)
            含义: b_seq_len[i] 表示第i个token所在请求的当前序列长度
            
        select_index (torch.Tensor): 
            当前批次中每个token在KV缓存池中的全局索引
            形状: (num_tokens,)
            含义: select_index[i] 表示第i个token的KV数据在缓存池中的位置
    
    函数行为：
        对于批次中的每个token i，将其KV缓存索引select_index[i]存储到
        req_to_token_indexs[b_req_idx[i]][b_seq_len[i] - 1]位置。
    
    使用场景：
        1. 大模型推理服务中的动态批处理
        2. 支持变长序列的KV缓存管理
        3. 多轮对话中的上下文管理
    
    简单例子：
        假设我们有3个请求，最大序列长度为4：
        
        # 输入数据
        req_to_token_indexs = torch.zeros(3, 4, dtype=torch.int32)  # 3个请求，最大长度4
        b_req_idx = torch.tensor([0, 0, 1, 2, 2])     # 5个token分别属于请求0,0,1,2,2
        b_seq_len = torch.tensor([1, 2, 1, 1, 2])     # 各token所在请求的当前长度
        select_index = torch.tensor([10, 11, 12, 13, 14])  # KV缓存中的索引
        
        # 调用函数
        update_kv_index(req_to_token_indexs, b_req_idx, b_seq_len, select_index)
        
        # 结果解释：
        # token 0: 属于请求0，是该请求的第1个token，KV索引为10
        #         存储到 req_to_token_indexs[0][0] = 10
        # token 1: 属于请求0，是该请求的第2个token，KV索引为11  
        #         存储到 req_to_token_indexs[0][1] = 11
        # token 2: 属于请求1，是该请求的第1个token，KV索引为12
        #         存储到 req_to_token_indexs[1][0] = 12
        # token 3: 属于请求2，是该请求的第1个token，KV索引为13
        #         存储到 req_to_token_indexs[2][0] = 13
        # token 4: 属于请求2，是该请求的第2个token，KV索引为14
        #         存储到 req_to_token_indexs[2][1] = 14
        
        # 最终结果：
        # req_to_token_indexs = [
        #     [10, 11,  0,  0],  # 请求0: 有2个token，KV索引分别为10,11
        #     [12,  0,  0,  0],  # 请求1: 有1个token，KV索引为12  
        #     [13, 14,  0,  0],  # 请求2: 有2个token，KV索引分别为13,14
        # ]
    
    注意事项：
        1. 所有输入张量必须在GPU上且数据类型兼容
        2. b_req_idx、b_seq_len、select_index的第一个维度大小必须相同
        3. req_to_token_indexs的第一个维度应该大于等于b_req_idx中的最大值+1
        4. 该函数使用@torch.no_grad()装饰器，不计算梯度，适用于推理阶段
    """
    # 获取当前批次的token数量
    seq_len = b_seq_len.shape[0]

    # 安全性检查：确保所有输入张量在第一个维度上的大小相同
    # 这是必要的，因为每个token都需要有对应的请求ID、序列长度和KV索引
    assert (
        b_seq_len.shape[0] == select_index.shape[0]
        and b_req_idx.shape[0] == b_seq_len.shape[0]
    ), f"输入张量大小不匹配: b_req_idx={b_req_idx.shape}, b_seq_len={b_seq_len.shape}, select_index={select_index.shape}"

    # 定义Triton内核的执行网格
    # 使用1D网格，每个线程处理一个token
    # seq_len个线程并行执行，每个线程处理批次中的一个token
    grid = (seq_len,)

    # 定义每个block使用的warp数量
    # warp是GPU中的基本执行单元，通常包含32个线程
    # 对于简单的内存操作，1个warp就足够了
    num_warps = 1

    # 启动Triton内核进行并行计算
    # 传递所有必要的参数给GPU内核
    _fwd_kernel_update_kv_index[grid](
        req_to_token_indexs,              # 输出张量的指针
        b_req_idx,                        # 请求索引张量的指针
        b_seq_len,                        # 序列长度张量的指针
        select_index,                     # KV索引张量的指针
        req_to_token_indexs.stride(0),    # 第一个维度的内存步幅（请求维度）
        req_to_token_indexs.stride(1),    # 第二个维度的内存步幅（序列长度维度）
        num_warps=num_warps,              # 每个block的warp数量
        num_stages=1,                     # 流水线阶段数，1表示不使用流水线
    )
    return
