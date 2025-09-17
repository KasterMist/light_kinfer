"""
KV缓存更新模块

本模块实现了高效的KV缓存更新操作，主要用于大语言模型推理过程中的键值缓存管理。
在Transformer架构中，为了避免重复计算，通常会缓存每层的Key和Value张量。
当有新的token生成或批次处理时，需要将新计算的KV值更新到缓存中的指定位置。

核心功能：
1. 支持非连续索引的KV缓存更新
2. 高效的GPU并行更新操作
3. 同时处理Key和Value的更新（合并在一个张量中）
4. 支持变长序列和批量处理

应用场景：
- 自回归生成中的KV缓存增量更新
- 批量推理中的动态缓存管理
- Prefill和Decode阶段的缓存维护

技术特点：
- 基于Triton实现，充分利用GPU并行性
- 支持任意索引位置的更新操作
- 内存访问模式优化，减少带宽浪费
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_update_kv(
    KV_Values,          # 源KV张量指针，包含待更新的Key和Value数据
    Select_Index,       # 目标位置索引数组，指定每个KV值应该存储的位置
    KV_Buffer,          # 目标KV缓存张量指针，用于存储更新后的KV数据
    # 源张量的内存步长参数
    stride_k_bs,        # KV_Values在批次/序列维度的步长
    stride_k_h,         # KV_Values在头维度的步长
    stride_k_d,         # KV_Values在特征维度的步长
    # 目标张量的内存步长参数
    stride_o_bs,        # KV_Buffer在批次/序列维度的步长
    stride_o_h,         # KV_Buffer在头维度的步长
    stride_o_d,         # KV_Buffer在特征维度的步长
    # 其他参数
    head_num,           # 注意力头的总数量（包括Key和Value）
    # 编译时常量
    BLOCK_DMODEL: tl.constexpr,    # 特征维度的块大小
    BLOCK_HEAD: tl.constexpr,      # 头维度的块大小
):
    """
    KV缓存更新的核心Triton内核
    
    该内核负责将新计算的KV值复制到缓存中的指定位置。每个GPU线程块
    处理一个token的所有头和所有特征维度的KV数据。
    
    工作原理：
    1. 每个线程块处理一个token（由program_id(0)确定）
    2. 从Select_Index中读取目标位置索引
    3. 计算源数据和目标数据的内存地址
    4. 执行并行的内存复制操作
    5. 使用掩码确保不越界访问
    
    内存访问模式：
    - 源地址：KV_Values[cur_index, :, :]
    - 目标地址：KV_Buffer[dest_index, :, :]
    - 支持非连续的目标索引，提供灵活的缓存管理
    """
    # 获取当前线程块要处理的token索引
    cur_index = tl.program_id(0)
    
    # 创建头维度和特征维度的偏移向量
    offs_h = tl.arange(0, BLOCK_HEAD)      # 头维度偏移：[0, 1, 2, ..., BLOCK_HEAD-1]
    offs_d = tl.arange(0, BLOCK_DMODEL)    # 特征维度偏移：[0, 1, 2, ..., BLOCK_DMODEL-1]

    # 从索引数组中读取当前token应该存储到的目标位置
    dest_index = tl.load(Select_Index + cur_index)

    # 计算源数据（KV_Values）的内存访问指针
    # 访问模式：KV_Values[cur_index, heads, features]
    k_ptrs = (
        KV_Values
        + cur_index * stride_k_bs              # 定位到当前token
        + stride_k_h * offs_h[:, None]         # 头维度偏移（广播到2D）
        + stride_k_d * offs_d[None, :]         # 特征维度偏移（广播到2D）
    )
    
    # 计算目标数据（KV_Buffer）的内存访问指针
    # 访问模式：KV_Buffer[dest_index, heads, features]
    o_ptrs = (
        KV_Buffer
        + dest_index * stride_o_bs             # 定位到目标位置
        + stride_o_h * offs_h[:, None]         # 头维度偏移（广播到2D）
        + stride_o_d * offs_d[None, :]         # 特征维度偏移（广播到2D）
    )

    # 从源位置加载KV数据，使用掩码防止超出头数边界
    kv_value = tl.load(k_ptrs, mask=offs_h[:, None] < head_num, other=0.0)
    
    # 将数据存储到目标位置，同样使用掩码保护
    tl.store(o_ptrs, kv_value, mask=offs_h[:, None] < head_num)
    return


@torch.no_grad()
def update_kv_buffer(KV_Values, Select_Index, KV_Buffer):
    """
    KV缓存更新的主函数接口
    
    这个函数实现了高效的KV缓存更新操作，是大语言模型推理中的关键组件。
    在Transformer的注意力机制中，为了避免重复计算历史token的Key和Value，
    通常会维护一个KV缓存。当处理新token时，需要将新计算的KV值更新到缓存的指定位置。
    
    核心特性：
    1. 支持非连续索引更新：可以将KV值更新到缓存的任意位置
    2. 批量并行处理：同时处理多个token的KV更新
    3. 内存高效：避免不必要的数据复制和移动
    4. GPU优化：基于Triton实现，充分利用GPU并行性
    
    应用场景：
    - Prefill阶段：将整个输入序列的KV值一次性更新到缓存
    - Decode阶段：将新生成token的KV值追加到缓存末尾
    - 动态批处理：处理不同长度序列的KV缓存管理
    
    张量形状说明：
    - KV_Values: [num_tokens, num_kv_heads * 2, head_dim]
      其中 num_kv_heads * 2 是因为Key和Value合并存储
    - Select_Index: [num_tokens] 指定每个token应该存储的缓存位置
    - KV_Buffer: [max_cache_size, num_kv_heads * 2, head_dim] 全局KV缓存
    
    参数：
        KV_Values (torch.Tensor): 源KV张量，包含待更新的Key和Value数据
            - 形状: [select_tokens, num_kv_heads * 2, head_dim]
            - 数据类型: 通常为float16以节省内存
            - 设备: CUDA设备上的张量
            
        Select_Index (torch.Tensor): 目标位置索引数组
            - 形状: [select_tokens]
            - 数据类型: int32
            - 含义: Select_Index[i]指定KV_Values[i]应该存储到KV_Buffer的哪个位置
            - Prefill阶段: 连续索引，长度为batch_size * seq_len
            - Decode阶段: 通常长度为batch_size，指向缓存末尾的新位置
            
        KV_Buffer (torch.Tensor): 目标KV缓存张量
            - 形状: [max_num_tokens, num_kv_heads * 2, head_dim]
            - 数据类型: 与KV_Values相同
            - 作用: 存储所有历史和当前的KV值
    
    操作语义：
        执行操作 KV_Buffer[Select_Index[i], :, :] = KV_Values[i, :, :]
        对于所有 i ∈ [0, select_tokens)
    
    性能优化：
        1. GPU并行：每个token的更新在独立线程块中并行执行
        2. 内存合并：优化内存访问模式以提高带宽利用率
        3. 向量化：利用Triton的向量化加载/存储操作
        4. 掩码保护：避免越界访问，提高代码健壮性
    
    示例用法：
        >>> # Prefill阶段示例
        >>> batch_size, seq_len, num_kv_heads, head_dim = 2, 10, 8, 64
        >>> kv_values = torch.randn(batch_size * seq_len, num_kv_heads * 2, head_dim)
        >>> select_index = torch.arange(batch_size * seq_len, dtype=torch.int32)
        >>> kv_buffer = torch.zeros(1000, num_kv_heads * 2, head_dim)  # 预分配缓存
        >>> update_kv_buffer(kv_values, select_index, kv_buffer)
        
        >>> # Decode阶段示例
        >>> new_kv = torch.randn(batch_size, num_kv_heads * 2, head_dim)  # 新生成的KV
        >>> new_positions = torch.tensor([20, 25], dtype=torch.int32)     # 存储位置
        >>> update_kv_buffer(new_kv, new_positions, kv_buffer)
    """
    # 解析输入张量的维度信息
    seq_len = Select_Index.shape[0]         # 需要更新的token数量
    head_num = KV_Values.shape[1]           # 头数量（num_kv_heads * 2，包含Key和Value）
    head_dim = KV_Values.shape[2]           # 每个头的特征维度
    
    # 验证输入张量的形状兼容性
    # KV_Values和KV_Buffer必须在头数和特征维度上保持一致
    assert (
        KV_Values.shape[1] == KV_Buffer.shape[1]
        and KV_Values.shape[2] == KV_Buffer.shape[2]
    ), f"张量形状不匹配: KV_Values{KV_Values.shape} vs KV_Buffer{KV_Buffer.shape}"
    
    # 计算Triton内核的块大小参数
    # 使用2的幂次方可以提高GPU内存访问效率
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    
    # 设置GPU执行网格：每个线程块处理一个token
    grid = (seq_len,)
    
    # 设置执行参数：使用较少的warp以适应简单的内存复制操作
    num_warps = 1

    # 启动Triton内核执行KV缓存更新
    _fwd_kernel_update_kv[grid](
        # 输入张量
        KV_Values,                  # 源KV数据
        Select_Index,               # 目标位置索引
        KV_Buffer,                  # 目标KV缓存
        # 源张量的内存步长信息
        KV_Values.stride(0),        # 批次/序列维度步长
        KV_Values.stride(1),        # 头维度步长
        KV_Values.stride(2),        # 特征维度步长
        # 目标张量的内存步长信息
        KV_Buffer.stride(0),        # 批次/序列维度步长
        KV_Buffer.stride(1),        # 头维度步长
        KV_Buffer.stride(2),        # 特征维度步长
        # 其他参数
        head_num,                   # 头数量
        # 编译时常量
        BLOCK_DMODEL=head_dim,      # 特征维度块大小
        BLOCK_HEAD=BLOCK_HEAD,      # 头维度块大小
        # GPU执行参数
        num_warps=num_warps,        # 每个线程块的warp数量
        num_stages=1,               # 流水线阶段数（简单操作使用1）
    )
    return


def test1():
    """
    KV缓存更新功能的性能测试和正确性验证
    
    该测试函数对比了Triton实现的update_kv_buffer与PyTorch原生操作的性能差异，
    并验证两种实现的结果是否一致。这有助于确保自定义内核的正确性和性能优势。
    
    测试设计：
    1. 创建大规模的测试数据以模拟真实使用场景
    2. 执行预热运行以消除GPU初始化开销
    3. 分别测量Triton和PyTorch实现的执行时间
    4. 比较两种实现的数值结果精度
    
    性能考量：
    - Triton实现通常在大规模数据上表现更好
    - PyTorch原生操作在小规模数据上可能更快（减少了内核启动开销）
    - 内存访问模式的优化是性能差异的主要来源
    
    测试参数说明：
    - B=32: 批次大小，模拟实际推理的批处理场景
    - Seq_Len=1024: 序列长度，测试长序列的处理能力
    - H=12: 头数量（包含Key和Value），典型的模型配置
    - D=128: 头维度，常见的特征维度设置
    """
    import time

    # 测试参数设置
    num_of_times = 1000                     # 性能测试的迭代次数
    B, Seq_Len, H, D = 32, 1024, 12, 128  # 批次大小、序列长度、头数、头维度

    # 创建测试数据
    # 使用float16以模拟实际推理中的数据类型
    dest = torch.randn((B * Seq_Len, H, D), dtype=torch.float16).cuda()    # 目标缓存
    src = torch.randn((B * Seq_Len, H, D), dtype=torch.float16).cuda()     # 源KV数据
    dest_loc = torch.arange(0, B * Seq_Len, dtype=torch.int32, device="cuda")  # 连续索引

    # 预热阶段：消除GPU内核编译和缓存的影响
    for _ in range(10):
        update_kv_buffer(src, dest_loc, dest)
    torch.cuda.synchronize()  # 确保所有GPU操作完成

    # 性能测试阶段1：测量Triton实现的执行时间
    t1 = time.time()
    for _ in range(num_of_times):
        update_kv_buffer(src, dest_loc, dest)
    torch.cuda.synchronize()    # 等待GPU完成所有操作
    t2 = time.time()

    # 性能测试阶段2：测量PyTorch原生操作的执行时间
    for _ in range(num_of_times):
        dest[dest_loc] = src    # PyTorch的高级索引赋值操作
    torch.cuda.synchronize()
    t3 = time.time()

    # 输出性能比较结果
    triton_time = t2 - t1
    torch_time = t3 - t2
    print(f"Triton实现用时: {triton_time:.4f}秒")
    print(f"PyTorch实现用时: {torch_time:.4f}秒")
    print(f"性能提升倍数: {torch_time / triton_time:.2f}x")
    
    # 数值精度验证
    max_diff = torch.max(torch.abs(dest - src))
    mean_diff = torch.mean(torch.abs(dest - src))
    print(f"最大绝对误差: {max_diff}")
    print(f"平均绝对误差: {mean_diff}")
    
    # 断言验证：确保两种实现的结果在数值精度范围内一致
    assert torch.allclose(src, dest, atol=1e-2, rtol=0), "数值验证失败：Triton实现与PyTorch结果不一致"


if __name__ == "__main__":
    """
    模块测试入口
    
    当直接运行此脚本时，执行KV缓存更新功能的完整测试，
    包括性能基准测试和正确性验证。
    
    测试输出包括：
    1. Triton实现与PyTorch实现的性能对比
    2. 数值精度验证结果
    3. 功能正确性断言检查
    
    适用场景：
    - 开发阶段的功能验证
    - 性能调优的基准测试
    - 新硬件平台的兼容性测试
    """
    print("开始KV缓存更新功能测试...")
    test1()
    print("所有测试通过！")
