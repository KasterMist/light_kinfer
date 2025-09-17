import torch
from copy import deepcopy
from typing import Dict
from light_kinfer.executor.executor_struct import AttentionInfo
from light_kinfer.executor.mem_manager import KVCacheMemoryManager

# CUDA Graph优化配置常量
# 批次大小对齐：确保批次大小是8的倍数，优化内存访问模式
_BATCH_SIZE_ALIGNMENT = 8

# 预定义要捕获的批次大小列表：覆盖常用的小批次和对齐的大批次
# 小批次 [1,2,4] 用于单用户或少量并发场景
# 大批次 [8,16,24,...,8192] 用于高并发批处理场景
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [
    _BATCH_SIZE_ALIGNMENT * i for i in range(1, 1025)
]


class CUDAGraphRunner:
    """
    CUDA Graph执行器 - 用于捕获和重放CUDA计算图以提升推理性能
    
    功能:
    - 捕获模型的前向传播计算图，避免重复的CUDA kernel启动开销
    - 重放预捕获的计算图，显著减少CPU-GPU同步时间
    - 专门针对decode阶段的单token生成进行优化
    
    工作原理:
    1. Capture阶段: 记录一次完整的模型前向传播过程
    2. Replay阶段: 直接重放录制的操作序列，跳过Python解释和调度开销
    
    性能优势:
    - 减少CUDA kernel启动延迟
    - 降低CPU-GPU同步开销  
    - 提升高频调用场景的吞吐量
    """
    
    def __init__(self, model):
        """
        初始化CUDA Graph执行器
        
        参数:
            model: 要进行图捕获的PyTorch模型
        """
        self.model = model                          # 待优化的模型
        self._cuda_graph = None                     # CUDA计算图对象
        self._graph_inputs: Dict[str, torch.Tensor] = {}  # 图输入张量缓存
        self._graph_output = None                   # 图输出张量引用

    def capture(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info: AttentionInfo,
    ):
        """
        捕获CUDA计算图 - 记录模型前向传播的完整执行序列
        
        参数:
            input_ids: 输入token序列，形状 [batch_size, seq_len]
            position_ids: 位置编码序列，形状 [batch_size, seq_len]  
            atten_info: 注意力相关信息，包含KV缓存等
            
        实现步骤:
        1. 预热运行: 确保CUDA context和内存分配稳定
        2. 图捕获: 在特殊的graph context中记录操作序列
        3. 缓存管理: 保存输入输出张量的引用，用于后续重放
        """
        assert self._cuda_graph is None, "Already compiled the model"
        # 保存用于捕获的占位符输入，用于后续验证和调试
        self._graph_inputs = [input_ids, position_ids, atten_info]

        # === 第一步: Warm up 预热运行 ===
        # 创建独立的CUDA流，避免与主流的同步问题
        graph_capture_stream = torch.cuda.Stream()
        graph_capture_stream.wait_stream(torch.cuda.current_stream())
        
        # 在独立流中进行预热，确保所有CUDA操作都已初始化
        with torch.cuda.stream(graph_capture_stream):
            _ = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                atten_info=atten_info,
            )
        # 等待预热完成，确保所有操作都已执行
        torch.cuda.current_stream().wait_stream(graph_capture_stream)

        # === 第二步: 捕获计算图 ===
        # 创建CUDA Graph对象，用于记录后续的操作序列
        self._cuda_graph = torch.cuda.CUDAGraph()
        
        # 在graph context中执行模型，所有操作都会被记录
        with torch.cuda.graph(self._cuda_graph):
            self._graph_output = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                atten_info=atten_info,
            )

        # === 第三步: 保存输入输出缓冲区引用 ===
        # 这些张量将在重放时被重用，避免重新分配内存
        self._graph_inputs = {
            "input_ids": input_ids,                              # 输入token序列
            "position_ids": position_ids,                        # 位置编码
            "kv_buffer": atten_info.kv_buffer,                   # KV缓存缓冲区
            "cur_select_index": atten_info.cur_select_index,     # 当前选择的缓存索引
            "b_req_tokens_table": atten_info.b_req_tokens_table, # 批次请求token表
            "b_req_idx": atten_info.b_req_idx,                   # 批次请求索引
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info: AttentionInfo,
    ):
        """
        重放CUDA计算图 - 使用新数据执行预捕获的操作序列
        
        参数:
            input_ids: 新的输入token序列
            position_ids: 新的位置编码序列
            atten_info: 新的注意力信息
            
        返回:
            模型输出张量（logits）
            
        实现原理:
        1. 内存优化: 删除不需要复制的固定张量引用
        2. 数据复制: 将新数据复制到预分配的图输入缓冲区
        3. 图重放: 执行预捕获的操作序列
        4. 返回结果: 直接返回图输出张量
        """
        # === 内存优化: 删除不需要复制的大型张量引用 ===
        # kv_buffer是固定的共享缓冲区，不需要复制，直接使用原始引用
        del atten_info.kv_buffer
        # b_req_tokens_table也是共享的，删除引用避免不必要的内存操作
        del atten_info.b_req_tokens_table
        
        # === 数据更新: 将新输入复制到图的输入缓冲区 ===
        # 使用copy_()进行原地复制，保持张量的内存地址不变
        # 这样图中记录的张量引用仍然有效
        self._graph_inputs["input_ids"].copy_(input_ids)
        self._graph_inputs["position_ids"].copy_(position_ids)
        self._graph_inputs["cur_select_index"].copy_(atten_info.cur_select_index)
        self._graph_inputs["b_req_idx"].copy_(atten_info.b_req_idx)

        # === 图重放: 执行预捕获的完整操作序列 ===
        # 这一步会重放之前捕获的所有CUDA kernels，无需Python解释开销
        self._cuda_graph.replay()

        # 返回图执行的输出结果
        return self._graph_output

    def __call__(self, *args, **kwargs):
        """
        使CUDAGraphRunner对象可直接调用，等价于forward方法
        """
        return self.forward(*args, **kwargs)


class ModelRunner:
    """
    模型运行器 - 管理CUDA Graph优化的模型执行和资源分配
    
    功能:
    - 统一管理多个批次大小的CUDA Graph实例
    - 为decode阶段提供优化的推理执行
    - 管理KV缓存和内存资源
    - 根据输入批次大小自动选择最优执行路径
    
    设计理念:
    - 预捕获常用批次大小的计算图，覆盖大部分推理场景
    - 对于未捕获的批次大小，回退到原始模型执行
    - 统一的接口，对上层调用透明
    """
    
    def __init__(
        self,
        model,
        model_config,
        max_gpu_num_blocks: int,
        kv_mem_manager: KVCacheMemoryManager,
        req_tokens_manager,
        seq_len: int = 1,
        start_pos=8,
    ):
        """
        初始化模型运行器
        
        参数:
            model: 要执行的PyTorch模型
            model_config: 模型配置对象，包含词汇表大小、最大批次等信息
            max_gpu_num_blocks: GPU上KV缓存的最大块数
            kv_mem_manager: KV缓存内存管理器
            req_tokens_manager: 请求token管理器  
            seq_len: decode阶段的序列长度，通常为1
            start_pos: 起始位置，用于位置编码计算
        """
        self.model = model
        self.model_config = model_config
        self.max_gpu_num_blocks = max_gpu_num_blocks
        self.kv_mem_manager = kv_mem_manager           # KV缓存内存管理
        self.req_tokens_manager = req_tokens_manager   # 请求token管理

        # 模型基本参数
        self.vocab_size = self.model_config.vocab_size           # 词汇表大小
        # 关键修复：确保图支持的最大批次不超过req_tokens_manager支持的最大请求数
        max_requests_supported = self.req_tokens_manager.max_can_use_req_size
        self.graph_max_batch_size = min(self.model_config.max_batch_size, max_requests_supported)
        # print(f"Debug: model_config.max_batch_size={self.model_config.max_batch_size}")
        # print(f"Debug: max_requests_supported={max_requests_supported}")
        # print(f"Debug: final graph_max_batch_size={self.graph_max_batch_size}")
        self.max_seq_len = model_config.max_seq_len              # 模型支持的最大序列长度

        # decode阶段特定参数
        self.seq_len = seq_len      # decode阶段每次处理的序列长度（固定为1）
        self.start_pos = start_pos  # 起始位置索引，用于位置编码

        # CUDA Graph管理：存储不同批次大小对应的图执行器
        self.graph_runners = {}  # Dict[int, CUDAGraphRunner] - 批次大小到图执行器的映射

    def build_atten_info(self, batch_size, atten_info, device="cuda"):
        """
        构建decode阶段的注意力信息结构体
        
        参数:
            batch_size: 当前批次大小
            atten_info: 注意力信息对象（将被填充）
            device: 计算设备
            
        返回:
            填充完整的AttentionInfo对象
            
        功能说明:
        - 专门针对decode阶段（seq_len=1）设计
        - 设置KV缓存缓冲区和索引管理
        - 配置批次相关的序列信息
        """
        # KV缓存缓冲区：存储所有层的Key-Value历史信息
        atten_info.kv_buffer = self.kv_mem_manager.gpu_kv_buffer
        
        # 批次请求token表：管理每个请求的token索引映射
        atten_info.b_req_tokens_table = self.req_tokens_manager.b_req_tokens_table

        # 批次请求索引：为当前批次中的每个请求分配唯一ID
        atten_info.b_req_idx = torch.arange(batch_size, device=device)
        
        # 批次序列长度：decode阶段每个序列长度都是1
        atten_info.b_seq_len = torch.ones(
            batch_size, dtype=torch.int32, device="cuda"
        )
        
        # 分配KV缓存索引：为当前批次分配缓存空间
        atten_info.cur_select_index, _ = self.kv_mem_manager.alloc_kvcache_index(
            batch_size
        )
        
        # 最大实际序列长度：start_pos + 1（decode阶段固定值）
        atten_info.max_actual_seq_len = self.start_pos + 1
        
        # 调试信息
        # print(f"Debug: batch_size={batch_size}, start_pos={self.start_pos}")
        # print(f"Debug: b_req_tokens_table shape={atten_info.b_req_tokens_table.shape}")
        # print(f"Debug: cur_select_index shape={atten_info.cur_select_index.shape}")
        
        return atten_info

    def capture_decode_graph(self):
        """
        批量捕获decode阶段的CUDA计算图
        
        功能:
        - 为多个常用批次大小预先捕获计算图
        - 优化内存使用：从大批次到小批次的捕获顺序
        - 为每个批次大小创建独立的图执行器
        
        实现策略:
        1. 筛选有效批次：只捕获不超过最大批次限制的大小
        2. 逆序捕获：先捕获大批次，有助于减少内存碎片
        3. 构造模拟输入：生成随机输入数据进行图捕获
        4. 资源管理：每次捕获后清理KV缓存，避免内存泄露
        """
        # 筛选要捕获的批次大小：确保不超过模型配置的最大批次限制
        batch_size_capture_list = [
            bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= self.graph_max_batch_size
        ]
        
        # 创建注意力信息结构体实例
        atten_info = AttentionInfo()
        print("cuda graph support batch list", batch_size_capture_list)

        # 逆序遍历批次大小：从大到小捕获，优化内存分配模式
        # 大批次优先分配连续内存空间，小批次可以利用剩余空间
        for batch_size in reversed(batch_size_capture_list):
            # === 构造模拟输入数据 ===
            # 生成随机token ID，模拟真实的decode输入
            # 形状: [batch_size, 1] - decode阶段每次只处理1个token
            input_ids = torch.randint(0, self.vocab_size, (batch_size, 1)).cuda()
            
            # 构造位置编码：基于start_pos生成单步位置
            # 形状变化: [1] -> [1, 1] -> [batch_size, 1]
            position_ids = (
                torch.arange(
                    self.start_pos, self.start_pos + 1, device=input_ids.device
                )
                .unsqueeze(0)      # 增加批次维度: [1, 1]
                .expand(batch_size, -1)  # 扩展到目标批次大小: [batch_size, 1]
            )
            
            # 构建注意力信息：设置KV缓存和序列管理信息
            atten_info = self.build_atten_info(batch_size, atten_info)

            # === 执行图捕获 ===
            # 准备图捕获的输入元组
            graph_input = (input_ids, position_ids, atten_info)
            
            # 创建专用的图执行器实例
            graph_runner = CUDAGraphRunner(self.model)

            # 捕获当前批次大小的计算图
            graph_runner.capture(*graph_input)
            
            # 存储图执行器，以批次大小为键
            self.graph_runners[batch_size] = graph_runner

            # === 资源清理 ===
            # 释放当前捕获使用的KV缓存空间，为下一次捕获做准备
            self.kv_mem_manager.free_all()

    def decode(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info: AttentionInfo,
    ):
        """
        执行decode阶段的模型推理，自动选择最优执行路径
        
        参数:
            input_ids: 输入token序列，形状 [batch_size, 1]
            position_ids: 位置编码序列，形状 [batch_size, 1]
            atten_info: 注意力信息结构体
            
        返回:
            logits: 模型输出的词汇概率分布，形状 [batch_size, 1, vocab_size]
            
        执行策略:
        1. 优先使用CUDA Graph：如果当前批次大小已预捕获图，使用图执行器
        2. 回退到原始模型：如果未捕获对应批次的图，使用标准前向传播
        
        性能考虑:
        - 图执行器：低延迟，适合高频调用
        - 原始模型：灵活性好，适合非标准批次大小
        """
        # 获取当前推理的批次大小
        batch_size = input_ids.shape[0]
        
        # 选择执行器：优先使用预捕获的CUDA Graph
        if batch_size in self.graph_runners:
            # 使用对应批次大小的图执行器，享受CUDA Graph的性能优势
            model_executable = self.graph_runners[batch_size]
        else:
            # 当前批次大小未预捕获图，回退到原始模型执行
            # 这种情况下仍能正常工作，但性能略低
            print(
                "Warning: CUDA graph not captured for this batch size, falling back to original model."
            )
            model_executable = self.model

        # 执行推理：无论使用图执行器还是原始模型，接口都是统一的
        logits = model_executable(input_ids, position_ids, atten_info)
        return logits
