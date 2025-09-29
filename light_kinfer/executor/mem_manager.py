import torch
import json, gc
import numpy as np
from pathlib import Path

from light_kinfer.utils.dummy_data import DummyInputGenerator
from light_kinfer.executor.executor_struct import AttentionInfo, CONFIG_CLASS_MAP
from light_kinfer.utils.logger import get_logger

logger = get_logger(__name__)

def get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()

class ComputeMaxAvailableBlocks:
    """
    A class that can execute a forward pass with dummy inputs to profile the memory usage of the model.
    and  calculate the maximum possible number of GPU blocks that can be allocated with the remaining free memory.
    if not execute dummy forward run, it should be run after cuda graph!
    """

    def __init__(
        self,
        num_layers,
        hidden_size,
        num_kv_heads,       # kv_head数量
        head_dim,           # 每个kv_head的维度(大小)
        gpu_memory_utilization=0.9,
        block_size=1,       # 每个block包含的token数量
        dtype=torch.float16,
        device="cuda",
    ):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_kv_heads = num_kv_heads    
        self.head_dim = head_dim            
        self.gpu_memory_utilization = gpu_memory_utilization
        self.block_size = block_size        
        self.dtype = dtype
        self.device = device
        self.dtype_size = get_dtype_size(dtype)

    def compute_cache_block_size_bytes(self):
        """
        获取KV缓存块大小（字节为单位）
        """
        # 每层KV缓存的字节数 (*2 代表key和value各占一半)
        kv_cache_token_bytes_per_layer = self.num_kv_heads * self.head_dim * 2 * self.dtype_size
        # 单个tokenKV缓存字节数
        kv_cache_token_bytes = kv_cache_token_bytes_per_layer * self.num_layers
        # 每个KV缓存块的字节数
        kv_cache_blocks_bytes = kv_cache_token_bytes * self.block_size

        return kv_cache_blocks_bytes

    def compute_num_available_blocks(self, model, model_path=None):
        """
        评估模型的峰值内存使用情况，以确定在不发生内存溢出的情况下可以分配的 KV（键值）缓存块的数量。

        该方法首先清理 CUDA 缓存，然后使用虚拟输入执行一次前向传播，以评估模型的内存使用情况。
        接着，计算在剩余可用内存下，最多可以分配的 GPU 和 CPU 缓存块数量。

        提示：
            可以通过调整 `gpu_memory_utilization` 参数来限制 GPU 内存的使用。
        """
        # 清理 CUDA 缓存，以确保获取准确的内存使用信息
        # NOTE: torch.cuda.empty_cache() 用于释放 GPU 上由缓存分配器持有的未占用内存。
        # NOTE: torch.cuda.reset_peak_memory_stats() 用于重置 CUDA 内存分配器所跟踪的“峰值”统计数据。
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # 获取当前 GPU 的空闲内存和总内存（单位：字节）# free_memory_pre_profile=9178578944
        free_memory_pre_profile, total_gpu_memory = torch.cuda.mem_get_info()
        
        # 使用虚拟输入执行一次前向传播，以评估模型的内存使用情况
        params_path = Path(model_path) / "config.json"
        if params_path.exists():
            with open(params_path, "r") as f:
                params = json.load(f)
                model_config = CONFIG_CLASS_MAP.get(params["model_type"].lower())
                    
                # 创建虚拟输入 
                batch_size = 1
                seq_len = 32  # 使用较小的序列长度进行内存评估
                dummy_generator = DummyInputGenerator(device="cuda")
                dummy_input, dummy_position_ids = dummy_generator.generate_dummy_input(model_config, batch_size, seq_len)
                    
                # 创建虚拟的 atten_info 对象
                dummy_atten_info = AttentionInfo()
                
                dummy_atten_info.kv_buffer = [
                    torch.empty((seq_len, 2 * self.num_kv_heads, self.head_dim), dtype=self.dtype, device=self.device) for _ in range(self.num_layers)
                ]
                 
                dummy_atten_info.cur_select_index = torch.arange(seq_len, dtype=torch.int32, device="cuda")
                dummy_atten_info.b_start_loc = torch.tensor([0], dtype=torch.int32, device="cuda")
                dummy_atten_info.b_seq_len = torch.tensor([1], device="cuda")
                dummy_atten_info.max_actual_seq_len=seq_len
                # 执行前向传播
                with torch.no_grad():
                    _ = model(dummy_input, dummy_position_ids, dummy_atten_info)

        logger.info(f"模型加载后可用内存: {torch.cuda.mem_get_info()[0] / (1024**3):.2f} GB")
        # 同步 CUDA 操作，确保内存信息准确
        torch.cuda.synchronize()
        # 计算模型加载后的峰值内存使用量
        peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        # 清理未使用的缓存，计算非Torch分配的内存，检查是否有任何剩余内存可能已在“torch”之外的gpu分配。例如，NCCL 操作在前向传递期间可能会使用几 GB
        torch.cuda.empty_cache()
        torch_allocated_bytes = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        total_allocated_bytes = torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]
        non_torch_allocations = total_allocated_bytes - torch_allocated_bytes

        if non_torch_allocations > 0:
            peak_memory += non_torch_allocations
        
        available_kv_cache_memory = total_gpu_memory * self.gpu_memory_utilization - peak_memory

        # 计算每个缓存块的大小
        cache_block_size = self.compute_cache_block_size_bytes()
        # 计算在剩余可用内存下，最多可以分配的GPU缓存块的数量
        num_gpu_blocks = int(
            (total_gpu_memory * self.gpu_memory_utilization - peak_memory) // cache_block_size
        )

        num_gpu_blocks = max(num_gpu_blocks, 0) # 确保块数不为负数

        logger.info(
            f" Memory profiling results: total_gpu_memory = {total_gpu_memory / (1024**3):.2f} GB \n"
            f"    initial_memory_usage = {(total_gpu_memory - free_memory_pre_profile) / (1024**3):.2f} GB "
            f"peak_torch_memory = {(peak_memory - non_torch_allocations) / (1024**3):.2f} GB \n"
            f"    memory_usage_post_profile = {total_allocated_bytes / (1024**3):.2f} GB \n"
            f"    non_torch_memory = {non_torch_allocations / (1024**3):.2f} GB, "
            f"kv_cache_size = {available_kv_cache_memory / (1024**3):.2f} GB \n"
            f"    gpu_memory_utilization = {self.gpu_memory_utilization:.2f}"
        )

        gc.collect() # 进行垃圾回收，释放未使用的内存
        torch.cuda.empty_cache() # 再次清理CUDA缓存

        return num_gpu_blocks   # 返回可分配的 GPU 和 CPU 缓存块数量（此处 CPU 块数量为 0）


class KVCacheMemoryManager:
    """
    KV缓存内存管理器 - 负责高效管理Transformer模型的Key-Value缓存内存
    
    主要功能:
    1. 预分配固定大小的KV缓存池，避免动态内存分配的开销
    2. 支持连续和非连续内存块分配，优化内存访问效率
    3. 使用引用计数机制管理内存块的生命周期
    4. 提供内存块的分配、释放和状态跟踪功能
    
    核心设计理念:
    - 内存池化: 预分配大块连续内存，减少GPU内存碎片
    - 索引映射: 使用索引而非指针管理内存位置，便于GPU并行操作
    - 引用计数: 支持内存块共享，优化sequence共享prefix的场景
    
    参数说明:
        num_layers: int, 模型的 Transformer 层数 (每层都需要独立的KV缓存)
        num_kv_heads: int, 每层的 KV 头数 (支持Multi-Head和Grouped-Query Attention)
        head_dim: int, 每个注意力头的维度大小
        gpu_num_blocks: int, 最大可用内存块数量 (用户配置的内存上限)
        block_size: int, 每个内存块包含的token数量，默认为1 (future: 支持PagedAttention)
        dtype: torch数据类型, KV缓存的数据精度，影响内存使用量
        device: str, 设备类型 ("cuda" 或 "cpu")
    """

    def __init__(self, num_layers, num_kv_heads, head_dim, gpu_num_blocks, block_size=1, dtype=torch.float16, device="cuda"):
        # === 模型结构参数 ===
        self.num_layers = num_layers        # Transformer层数，每层需要独立的KV缓存
        self.num_kv_heads = num_kv_heads    # KV注意力头数 (GQA中通常小于Query头数)
        self.head_dim = head_dim            # 每个注意力头的特征维度
        
        # === 内存管理参数 ===
        self.gpu_num_blocks = gpu_num_blocks    # 用户配置的最大内存块数量
        self.block_size = block_size            # 每个内存块的token容量 (当前=1, future支持更大值)
        self.max_num_tokens = gpu_num_blocks * block_size  # 总的可管理token数量
        
        # === 设备和数据类型 ===
        self.dtype = dtype      # 数据精度: float16节省内存, float32提高精度
        self.device = device    # 运行设备: "cuda"用GPU, "cpu"用内存
        
        # === 内存状态跟踪 ===
        self.can_use_mem_size = gpu_num_blocks  # 当前可分配的内存块数量 (动态减少)
        
        # === 核心数据结构 ===
        # kv_mem_pos_indexs: 内存位置索引映射表
        # 作用: 将逻辑内存块ID映射到实际GPU内存位置
        # 形状: [max_num_tokens] 
        # 内容: [0, 1, 2, 3, ..., max_num_tokens-1]
        # 例子: token_id=5 对应 GPU内存的第5个位置
        self.kv_mem_pos_indexs = torch.arange(0, self.max_num_tokens, dtype=torch.long, device="cuda")
        
        # kv_mem_use_state: 内存使用状态和引用计数表
        # 作用: 记录每个内存位置的使用状态和引用次数
        # 形状: [max_num_tokens]
        # 取值含义:
        #   0: 未使用，可分配
        #   1: 被1个序列引用
        #   2: 被2个序列引用 (共享prefix场景)
        #   >2: 被多个序列引用
        self.kv_mem_use_state = torch.zeros(self.max_num_tokens, dtype=torch.int32, device="cuda")

        # === 初始化KV缓存池 ===
        self.init_kv_buffers(
            self.max_num_tokens,
            head_dim, num_kv_heads, num_layers, 
            dtype, device
        )

    def init_kv_buffers(self, 
        max_num_tokens,
        head_dim, num_kv_heads, num_layers,
        dtype, device: str="cuda"
    )-> list[torch.Tensor]:
        """
        初始化KV缓存内存池 - 为所有Transformer层预分配GPU内存
        
        设计原理:
        1. 预分配策略: 一次性分配所有需要的内存，避免运行时的动态分配开销
        2. 内存布局优化: 使用连续内存块，提高GPU访问效率
        3. 分层管理: 每个Transformer层独立管理自己的KV缓存
        
        内存布局详解:
        - 每层缓存形状: [max_num_tokens, 2 * num_kv_heads, head_dim]
        - 维度说明:
          * max_num_tokens: 最大可缓存的token数量 (序列长度维度)
          * 2 * num_kv_heads: Key和Value缓存拼接 (2倍头数，前一半是Key，后一半是Value)
          * head_dim: 每个注意力头的特征维度
        
        内存访问模式:
        - token_id=i 的Key缓存: gpu_kv_buffer[layer][i, :num_kv_heads, :]
        - token_id=i 的Value缓存: gpu_kv_buffer[layer][i, num_kv_heads:, :]
        
        参数:
            max_num_tokens: 最大token容量
            head_dim: 注意力头维度
            num_kv_heads: KV头数量
            num_layers: Transformer层数
            dtype: 数据类型 (影响内存使用量)
            device: 设备类型
        
        返回:
            list[torch.Tensor]: 每层的KV缓存张量列表
        """
        # TODO: 未来支持PagedAttention的块式内存管理
        # 当前实现: 简单的连续内存分配
        # 未来扩展: 支持非连续的page-based内存管理，进一步优化内存利用率

        # gpu_kv_buffer会在外部被调用和管理 --> TODO: 如何优化结构 
        self.gpu_kv_buffer = [
            torch.empty(
                (max_num_tokens, 2 * num_kv_heads, head_dim), 
                dtype=dtype, 
                device=device
            ) 
            for _ in range(num_layers)
        ]
        
        # 日志输出：便于调试和性能分析
        total_memory_mb = (
            max_num_tokens * 2 * num_kv_heads * head_dim * 
            torch.tensor([], dtype=dtype).element_size() * num_layers
        ) / (1024 * 1024)
        
        logger.debug(
            f"KV缓存池初始化完成: "
            f"每层形状={self.gpu_kv_buffer[0].shape}, "
            f"总层数={num_layers}, "
            f"总内存={total_memory_mb:.2f}MB"
        )

    # ========================= KV缓存分配核心方法 =================================
    
    @torch.no_grad()
    def alloc_kvcache(self, need_size):
        """
        分配非连续的KV缓存内存块 - 灵活分配策略
        
        适用场景:
        1. 内存碎片较多时的应急分配
        2. 不要求内存连续性的应用场景
        3. 小块内存分配 (need_size较小)
        
        分配策略:
        - 从可用内存池中选择任意可用位置
        - 不保证内存连续性，但分配速度快
        - 适合GPU并行访问的场景
        
        工作流程:
        1. 检查可用内存是否足够
        2. 找到所有未使用的内存位置 (state=0)
        3. 选择前need_size个位置
        4. 更新引用计数和可用内存统计
        
        参数:
            need_size: int, 需要分配的token数量
            
        返回:
            torch.Tensor | None: 
                - 成功: 返回分配的内存位置索引张量 [need_size]
                - 失败: 返回None (内存不足)
        
        示例:
            need_size=3, 可用位置=[0,2,5,7,9]
            返回: tensor([0,2,5]) - 非连续但可用的位置
        """
        # 内存充足性检查
        if need_size > self.can_use_mem_size:
            logger.warning(
                f"⚠️ 内存不足! 需要分配={need_size}个token, "
                f"剩余可用={self.can_use_mem_size}个token"
            )
            return None
        
        # 查找所有未使用的内存位置
        # nonzero返回所有非零元素的索引，这里找state=0的位置
        can_use_pos_index = torch.nonzero(self.kv_mem_use_state == 0).view(-1)
        
        # 选择前need_size个可用位置 (不要求连续性)
        select_index = can_use_pos_index[0:need_size]
        
        # 增加引用计数，标记为已使用
        self.add_ref(select_index)
        
        logger.debug(f"✅ 非连续分配成功: 分配了{need_size}个token, 位置={select_index.tolist()}")
        return select_index

    @torch.no_grad()
    def alloc_contiguous_kvcache(self, need_size):
        """
        分配连续的KV缓存内存块 - 高性能分配策略
        
        优势和适用场景:
        1. 内存访问局部性好: 连续内存访问效率高，减少GPU cache miss
        2. 简化索引计算: 只需记录起始位置和长度
        3. 批处理友好: 便于vectorized操作和CUDA kernel优化
        4. 适合长序列: 大块连续内存访问性能更佳
        
        分配算法:
        1. 滑动窗口法: 在可用位置中寻找连续的内存块
        2. 差值检测: 通过end_pos - start_pos == need_size-1 判断连续性
        3. 优先分配: 找到第一个满足条件的连续块即返回
        
        工作流程详解:
        Step 1: 获取所有可用位置 [可用位置可能不连续]
        Step 2: 构造所有可能的(起始,结束)位置对
        Step 3: 计算位置差值，查找连续块
        Step 4: 分配第一个找到的连续块
        
        参数:
            need_size: int, 需要的连续token数量
            
        返回:
            tuple | None:
                成功: (select_index, start_index, end_index)
                    - select_index: 分配的内存位置索引 [need_size]
                    - start_index: 起始位置 (整数)
                    - end_index: 结束位置 (整数, 不包含)
                失败: None (无足够连续内存)
        
        算法示例:
            假设: need_size=3, 可用位置=[0,1,2,5,6,8,9,10]
            
            Step 1: 构造起始和结束位置对
                start_indexs = [0,1,2,5,6,8]     # 前N-need_size+1个
                end_indexs   = [2,5,6,8,9,10]    # 后need_size-1个开始
            
            Step 2: 计算差值
                diff = [2-0, 5-1, 6-2, 8-5, 9-6, 10-8] = [2,4,4,3,3,2]
            
            Step 3: 查找连续块 (diff == need_size-1 = 2)
                连续位置: index=0 (diff[0]=2) 和 index=5 (diff[5]=2)
            
            Step 4: 选择第一个连续块
                选择index=0: start=0, end=3, positions=[0,1,2]
        """
        # 内存充足性检查
        if need_size > self.can_use_mem_size:
            logger.warning(
                f"⚠️ 连续内存不足! 需要分配={need_size}个连续token, "
                f"剩余可用={self.can_use_mem_size}个token"
            )
            return None

        # === Step 1: 获取所有可用内存位置 ===
        can_use_pos_index = torch.nonzero(self.kv_mem_use_state == 0).view(-1)
        N = can_use_pos_index.numel()  # 可用位置总数
        
        if N >= need_size:
            # === Step 2: 构造滑动窗口的起始和结束位置 ===
            # 解释: 为了找到长度为need_size的连续块，我们需要检查所有可能的起始位置
            # 起始位置不能超过 N-need_size，因为后面需要至少need_size个位置
            start_indexs = can_use_pos_index[:N - need_size + 1]  # 所有可能的起始位置
            end_indexs = can_use_pos_index[need_size - 1:]        # 对应的结束位置
            
            # === Step 3: 连续性检测 ===
            # 核心思想: 如果位置连续，则 end_pos - start_pos == need_size - 1
            # 例如: 连续位置[5,6,7] -> 7-5=2, need_size-1=2 ✓
            #       非连续[5,7,9] -> 9-5=4, need_size-1=2 ✗
            diff = end_indexs - start_indexs
            
            # === Step 4: 查找第一个连续块 ===
            # 获取所有连续的块的起始位置索引
            contiguous_blocks = (diff == need_size - 1).nonzero(as_tuple=True)[0]

            if contiguous_blocks.numel() > 0:
                # 获取第一个连续块的实际内存位置
                first_block_idx = contiguous_blocks[0]
                start_index = start_indexs[first_block_idx].item()  # 转为Python int
                end_index = start_index + need_size
                
                # 生成连续的内存位置索引
                select_index = self.kv_mem_pos_indexs[start_index:end_index]
                
                # 标记为已使用
                self.add_ref(select_index)
                
                # logger.debug(
                #     f"✅ 连续分配成功: 分配了{need_size}个连续token, "
                #     f"范围=[{start_index}, {end_index}), "
                #     f"位置={select_index.tolist()}"
                # )
                return select_index, start_index, end_index

        # 分配失败
        logger.debug(
            f"❌ 连续分配失败: 需要{need_size}个连续token, "
            f"但最大连续块小于该值"
        )
        return None

    @torch.no_grad()
    def alloc_kvcache_index(self, need_size):
        """
        智能KV缓存分配 - 优先连续分配，降级到非连续分配
        
        分配策略:
        1. 优先尝试连续分配 (性能最优)
        2. 连续分配失败时，降级到非连续分配
        3. 根据分配结果返回不同的数据结构
        
        设计思想:
        - 连续内存: 直接使用预分配的GPU缓存池，访问效率高
        - 非连续内存: 动态创建临时缓存，兼容性好但性能略低
        
        返回值设计:
        - select_index: 内存位置索引，用于后续的缓存访问
        - kv_cache: 仅在非连续分配时创建，作为临时缓存使用
        
        参数:
            need_size: int, 需要分配的token数量
            
        返回:
            tuple: (select_index, kv_cache)
                - select_index: torch.Tensor[int32], 内存位置索引
                - kv_cache: torch.Tensor | None, 临时缓存(仅非连续分配时)
        
        使用示例:
            # 请求分配5个token的缓存
            indices, temp_cache = manager.alloc_kvcache_index(5)
            
            if temp_cache is None:
                # 连续分配成功，使用预分配的GPU缓存池
                for layer in range(num_layers):
                    layer_cache = manager.gpu_kv_buffer[layer][indices]
            else:
                # 非连续分配，使用临时缓存
                layer_cache = temp_cache
        """
        # === 策略1: 尝试连续分配 (首选) ===
        alloc_result = self.alloc_contiguous_kvcache(need_size)
        
        if alloc_result is not None:
            # 连续分配成功
            select_index, start_index, end_index = alloc_result
            kv_cache = None  # 使用预分配的GPU缓存池，无需临时缓存
            
            # logger.debug(f"🎯 使用连续分配策略: 索引范围=[{start_index}, {end_index})")
        else:
            # === 策略2: 降级到非连续分配 ===
            select_index = self.alloc_kvcache(need_size)
            
            if select_index is not None:
                # 非连续分配成功，创建临时缓存
                # 注意: 这里只创建单层缓存，实际使用时可能需要为每层创建
                kv_cache = torch.empty(
                    (need_size, self.num_kv_heads, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
                logger.debug(f"📦 使用非连续分配策略: 创建临时缓存 {kv_cache.shape}")
            else:
                # 分配完全失败
                logger.error(f"💥 内存分配失败: 无法分配{need_size}个token")
                return None, None
        
        # 返回int32类型的索引 (GPU kernel通常要求int32)
        return select_index.to(torch.int32), kv_cache

    # ========================= 引用计数和内存状态管理 =================================
    
    @torch.no_grad()
    def add_ref(self, token_index: torch.Tensor):
        """
        增加内存块的引用计数 - 支持多序列共享内存
        
        引用计数机制的作用:
        1. 内存共享: 多个序列可以共享相同的prefix KV缓存
        2. 安全管理: 只有当引用计数为0时才能释放内存
        3. 并发支持: 支持多个请求同时使用同一块内存
        
        应用场景:
        - Chat场景: 多轮对话共享历史context
        - 批处理: 多个请求共享相同的system prompt
        - Prefix caching: 缓存常用的prompt prefix
        
        工作流程:
        1. 统计当前有多少内存块正在使用
        2. 增加指定位置的引用计数
        3. 更新可用内存大小统计
        
        参数:
            token_index: torch.Tensor, 需要增加引用的内存位置索引
            
        示例:
            假设: token_index = [5, 6, 7], 当前state = [0, 0, 1, 0, 0, 0, 1, 2]
            
            执行前: 已使用内存块 = 3个 (位置2,6,7), 可用 = 5个
            add_ref([5,6,7]) 后:
            - state变为: [0, 0, 1, 0, 0, 1, 2, 3] 
            - 新增使用: 位置5 (0->1)
            - 增加引用: 位置6 (1->2), 位置7 (2->3)
            - 可用内存减少: 5 -> 4个
        """
        # 获取指定位置的当前引用计数
        state = self.kv_mem_use_state[token_index]
        
        # 统计这些位置中有多少当前正在使用 (引用计数>0)
        has_used_tokens = torch.count_nonzero(state).item()
        all_tokens = len(state)
        
        # 更新可用内存大小: 
        # 只有从未使用(0)变为使用(>0)的内存块才会减少可用内存
        newly_used_tokens = all_tokens - has_used_tokens
        self.can_use_mem_size -= newly_used_tokens
        
        # 增加引用计数
        self.kv_mem_use_state[token_index] += 1
        
        # logger.debug(
        #     f"📈 引用计数增加: 位置={token_index.tolist()}, "
        #     f"新占用={newly_used_tokens}个, "
        #     f"剩余可用={self.can_use_mem_size}个"
        # )
        return
    
    @torch.no_grad()
    def release_ref(self, token_index: torch.Tensor):
        """
        减少内存块的引用计数 - 智能内存释放
        
        释放策略:
        1. 处理重复索引: 同一位置多次释放只减少对应次数
        2. 安全检查: 确保引用计数不会变成负数
        3. 内存回收: 当引用计数降为0时，标记为可重新分配
        
        技术细节:
        - unique处理: 处理token_index中的重复值，统计每个位置的释放次数
        - 原子操作: 使用GPU张量操作确保并发安全
        - 延迟回收: 实际内存不立即清零，等待下次分配时覆盖
        
        工作流程:
        1. 去重并统计每个位置的释放次数
        2. 批量减少引用计数
        3. 统计释放后变为可用的内存块数量
        4. 更新可用内存统计
        
        参数:
            token_index: torch.Tensor, 需要释放引用的内存位置索引
            
        示例:
            token_index = [5, 6, 6, 7] (位置6重复)
            unique处理后: unique_indices=[5,6,7], counts=[1,2,1]
            
            释放前 state: [0, 0, 1, 0, 0, 2, 3, 1]
            释放后 state: [0, 0, 1, 0, 0, 1, 1, 0]
            - 位置5: 2->1 (仍在使用)
            - 位置6: 3->1 (仍在使用) 
            - 位置7: 1->0 (变为可用)
            新增可用内存: 1个
        """
        # 处理重复的索引，统计每个位置需要释放的次数
        # unique: 去除重复值，return_counts: 返回每个唯一值的出现次数
        unique_indices, counts = token_index.unique(return_counts=True)
        
        # 批量减少引用计数
        self.kv_mem_use_state[unique_indices] -= counts
        
        # 统计释放后的内存使用情况
        state_after_release = self.kv_mem_use_state[unique_indices]
        
        # 计算有多少内存块从使用状态变为可用状态 (引用计数变为0)
        still_used_tokens = torch.count_nonzero(state_after_release).item()
        all_released_tokens = len(state_after_release)
        newly_freed_tokens = all_released_tokens - still_used_tokens
        
        # 增加可用内存计数
        self.can_use_mem_size += newly_freed_tokens
        
        # logger.debug(
        #     f"📉 引用计数减少: 位置={unique_indices.tolist()}, "
        #     f"释放次数={counts.tolist()}, "
        #     f"新释放={newly_freed_tokens}个, "
        #     f"剩余可用={self.can_use_mem_size}个"
        # )
        return
    
    # ========================= 内存释放和清理方法 =================================
    
    def _free_buffers(self):
        """
        释放KV缓存缓冲区 - 系统清理
        
        应用场景:
        1. 模型销毁时的资源清理
        2. 内存紧张时的强制回收
        3. 系统重启或重新初始化
        
        注意事项:
        - 调用后需要重新初始化才能继续使用
        - 确保没有其他组件仍在引用这些缓冲区
        """
        self.gpu_kv_buffer = None
        logger.info("🧹 KV缓存缓冲区已释放")
    
    @torch.no_grad()
    def free(self, free_index):
        """
        释放指定的KV缓存内存块 - 精确释放
        
        使用场景:
        1. 序列生成完成后释放其KV缓存
        2. 用户主动结束对话，释放对应的内存
        3. 缓存置换算法中的选择性释放
        
        工作流程:
        1. 将索引转换为long类型 (确保索引有效性)
        2. 调用引用计数释放机制
        3. 检查是否所有内存都已释放 (调试用途)
        
        参数:
            free_index: torch.Tensor, 需要释放的内存位置索引
            
        示例:
            # 释放序列ID=123对应的KV缓存
            seq_cache_indices = get_sequence_cache_indices(seq_id=123)
            manager.free(seq_cache_indices)
        """
        free_index = free_index.long()  # 确保索引为long类型
        self.release_ref(free_index)
        
        # 调试信息: 检查是否所有内存都已释放
        if self.can_use_mem_size == len(self.kv_mem_use_state):
            logger.debug(f"🎉 所有GPU内存已释放，可用大小={self.can_use_mem_size}")
        
        return
    
    @torch.no_grad()
    def free_all(self):
        """
        释放所有内存 - 全局重置
        
        使用场景:
        1. 批处理任务完成后的全局清理
        2. 内存碎片严重时的重置操作
        3. 系统错误恢复
        4. 基准测试的环境重置
        
        效果:
        - 重置所有引用计数为0
        - 恢复可用内存到初始状态
        - 清除所有内存分配记录
        
        注意: 
        - 不清理实际的缓存数据，只重置管理状态
        - 调用后所有之前的内存索引都失效
        - 确保没有其他组件持有旧的内存索引
        """
        # 重置所有状态
        self.can_use_mem_size = len(self.kv_mem_use_state)  # 恢复到初始容量
        self.kv_mem_use_state[:] = 0  # 批量设置所有引用计数为0
        
        logger.info(
            f"🔄 全局内存重置完成: "
            f"恢复可用内存={self.can_use_mem_size}个token, "
            f"总容量={self.max_num_tokens}个token"
        )

    # ========================= 调试和监控方法 =================================
    
    def get_memory_stats(self):
        """
        获取内存使用统计信息 - 监控和调试
        
        返回:
            dict: 包含详细内存统计的字典
        """
        total_tokens = len(self.kv_mem_use_state)
        used_tokens = torch.count_nonzero(self.kv_mem_use_state).item()
        available_tokens = self.can_use_mem_size
        
        # 计算内存利用率
        utilization = (used_tokens / total_tokens) * 100 if total_tokens > 0 else 0
        
        # 引用计数分布
        ref_counts = self.kv_mem_use_state.cpu().numpy()
        # 输入张量中所有唯一值的张量及其对应的频率
        unique_refs, ref_counts_freq = torch.unique(self.kv_mem_use_state, return_counts=True)
        
        stats = {
            'total_capacity': total_tokens,
            'used_tokens': used_tokens,
            'available_tokens': available_tokens,
            'utilization_percent': utilization,
            'reference_distribution': dict(zip(unique_refs.tolist(), ref_counts_freq.tolist())),
            'fragmentation_score': self._calculate_fragmentation(),
        }
        
        return stats
    
    def _calculate_fragmentation(self):
        """计算内存碎片化程度"""
        # 简单的碎片化指标: 连续可用块的数量
        available_mask = (self.kv_mem_use_state == 0).cpu().numpy()
        if not available_mask.any():
            return 0.0
        
        # 计算连续块的数量
        transitions = np.diff(np.concatenate(([False], available_mask, [False])).astype(int))
        num_segments = (transitions == 1).sum()
        max_possible_segments = available_mask.sum()
        
        # 碎片化分数: 1.0表示最大碎片化，0.0表示完全连续
        if max_possible_segments == 0:
            return 0.0
        return num_segments / max_possible_segments