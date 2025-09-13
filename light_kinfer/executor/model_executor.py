"""
ModelExecutor - 大模型推理执行器

这个模块是light_kinfer推理框架的核心组件，负责：
1. 模型加载和初始化
2. KV缓存的高效管理
3. 批量请求的并行处理
4. prefill和decode两阶段推理

典型使用流程示例：
    # 1. 初始化执行器
    executor = ModelExecutor.build(
        checkpoints_dir="/path/to/model",
        max_seq_len=2048,
        max_gpu_num_blocks=None,
        device="cuda"
    )
    
    # 2. Prefill阶段：处理输入prompt
    batch_prompts = ["Hello world", "How are you"]
    max_len = max(len(p.split()) for p in batch_prompts)
    actual_lens = [len(p.split()) for p in batch_prompts]
    b_req_idx = torch.tensor([0, 1])
    
    kv_indices, _ = executor.prefill_alloc_kv_cache(
        max_prompt_len=max_len,
        actual_prompt_lens=torch.tensor(actual_lens),
        b_req_idx=b_req_idx
    )
    
    # 进行prefill推理
    input_ids = tokenize(batch_prompts)  # 假设的分词函数
    position_ids = create_position_ids(input_ids)
    logits = executor.forward(input_ids, position_ids)
    
    # 3. Decode阶段：逐步生成新token
    for step in range(max_new_tokens):
        # 分配新token的KV缓存
        kv_indices = executor.decode_alloc_kv_cache(batch_size=len(batch_prompts))
        
        # 获取新token（通常是上一步生成的token）
        new_token_ids = sample_from_logits(logits)  # 假设的采样函数
        
        # 进行decode推理
        logits = executor.forward(new_token_ids, current_positions)
        
        # 检查是否完成生成
        if all_sequences_finished(logits):
            break

主要组件说明：
- KVCacheMemoryManager: 管理GPU上的KV缓存内存池
- ReqTokensManager: 管理请求到token的映射关系  
- AttentionInfo: 包含attention计算所需的所有信息
- update_kv_index: GPU内核，高效更新KV索引映射
"""

import torch
import torch.nn as nn

import json, time
from pathlib import Path

from transformers import LlavaConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from light_kinfer.executor.mem_manager import ComputeMaxAvailableBlocks, KVCacheMemoryManager
from light_kinfer.executor.req_tokens_manager import ReqTokensManager

# from light_kinfer.cuda_graph import ModelRunner
from light_kinfer.executor.executor_struct import AttentionInfo, CONFIG_CLASS_MAP
from light_kinfer.models.model_config import LlamaConfig
from light_kinfer.kernels import update_kv_index
from ..utils.logger import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Registry helpers (avoid long if/elif chains)
# -----------------------------------------------------------------------------

# TODO: understand
class ModelExecutor:
    # 定义类属性
    model_config = None
    model = None
    atten_info = AttentionInfo

    # 通过静态方法 build 将类属性当作默认配置使用
    @staticmethod
    def build(
        checkpoints_dir: str,
        max_seq_len: int,
        max_gpu_num_blocks: None,
        compiled_model: bool = False,
        device: str = "cuda",
    ):
        """
        构建 ModelExecutor 实例, 加载模型、分词器和初始化推理信息结构体 atten_info。

        参数:
            checkpoints_dir (str): 模型检查点目录路径。
            load_model (bool): 是否加载模型权重。
            max_seq_len (int): 最大序列长度。
            device (str): 设备类型（'cuda'或'cpu'）。

        返回:
            ModelExecutor: 初始化后的 ModelExecutor 实例。
        """
        model_config = ModelExecutor._load_model_config(checkpoints_dir, max_seq_len)
        model = ModelExecutor._load_model_weight(model_config, checkpoints_dir, device=device)

        # from transformers import AutoModelForCausalLM, AutoConfig
        # model = AutoModelForCausalLM.from_pretrained(checkpoints_dir)
        # model = model.to(device)
        print(model)

        return ModelExecutor(
            checkpoints_dir, model_config, model, max_gpu_num_blocks, compiled_model, device
        )

    @staticmethod
    def _load_model_config(checkpoints_dir: str, max_seq_len: int):
        cfg_path = Path(checkpoints_dir) / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"{cfg_path} not found")

        params = json.loads(cfg_path.read_text())
        cfg_cls = CONFIG_CLASS_MAP.get(params["model_type"].lower())
        
        if cfg_cls is None:
            raise ValueError(f"Unsupported model_type {params['model_type']!r}")
        
        return cfg_cls.from_dict(params)

    
    @staticmethod
    def _accelerate_load_weight(
        model_config,
        checkpoints_dir,
        device="cuda",
    ):
        with init_empty_weights():
            model = ModelExecutor._initialize_model(model_config, device=device)

        # 假设 model 是使用 init_empty_weights 初始化的空模型
        model = load_checkpoint_and_dispatch(
            model, checkpoints_dir, device_map="auto", dtype=torch.float16
        )

        # 将模型转换为半精度, 并验证抓换
        model.to(device)
        model.half()
        for param in model.parameters():
            assert param.dtype == torch.float16, "Model parameters are not in FP16"
        logger.info("Converted model to half precision (FP16)")

        return model

    @staticmethod
    def _load_model_weight(
        model_config,
        checkpoints_dir,
        device="cuda",
    ):
        start_time = time.time()

        # 初始化模型
        with init_empty_weights():
            model = ModelExecutor._initialize_model(model_config, device=device)
            state_dict = None
        #     from safetensors import safe_open
        #     # 加载safetensors权重文件
        #     with safe_open(Path(checkpoints_dir) / "model.safetensors", framework="pt", device=device) as f:
        #         state_dict = {key: f.get_tensor(key) for key in f.keys()}

        checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
        assert len(checkpoints) > 0, (
            f"no checkpoint files found in {checkpoints_dir}"
        )
        ckpt_path = str(checkpoints[0])
        logger.info(f'Loading checkpoint "{ckpt_path}"')
        # 使用 torch.load 加载权重文件。torch.load 可以根据需要将权重加载到指定的设备上
        state_dict = torch.load(
            ckpt_path, mmap=True, weights_only=True, map_location=device
        )

        model.load_state_dict(
            state_dict, strict=True, assign=True
        )  # 将加载的 state_dict 应用到模型实例中。
        model.eval()
        logger.info(f"Loaded state dict in {time.time() - start_time:.2f}s")

        # # 将模型转换为半精度, 并验证转换
        model.half().to(device)
        for param in model.parameters():
            assert param.dtype == torch.float16, "Model parameters are not in FP16"
        logger.info("Converted model to half precision (FP16)")

        return model

    @staticmethod
    def _initialize_model(model_config, device: str) -> nn.Module:
        """
        根据配置初始化模型并将其移动到指定设备。

        参数:
            model_config (LlamaConfig): 自定义模型的配置参数。
            device (str): 设备类型（'cuda'或'cpu'）。

        返回:
            nn.Module: 初始化后的模型。
        """
        model_type = model_config.model_type.lower()
        logger.info(
            f"Initializing model of type '{model_type}' and moving it to device '{device}'..."
        )
        # from transformers import AutoModelForCausalLM, AutoConfig
        # model = AutoModelForCausalLM.from_config(model_config)
        # model.to(device)

        if model_type == "llama":
            from light_kinfer.models.naive_llama import LlamaModel
            model = LlamaModel(model_config)
        # elif model_type == "qwen2":
        #     from light_kinfer.models.qwen2 import Qwen2Model
        #     model = Qwen2Model(model_config)
        # elif model_type == "qwen3":
        #     from light_kinfer.models.qwen3 import Qwen3Model
        #     model = Qwen3Model(model_config)
        # elif model_type == "llava":
        #     from light_kinfer.models.llava import LlavaLlama
        #     model = LlavaLlama(model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        logger.info(f"The model has been initialized and moved to the device. '{device}'")
        return model

    def __init__(
        self,
        checkpoints_dir: str,
        model_config,
        model,
        max_gpu_num_blocks=None,
        compiled_model=False,
        device="cuda",
    ):
        self.checkpoints_dir = checkpoints_dir
        self.model_config = model_config
        self.device = device
        if isinstance(model_config, LlavaConfig):
            self.llm_config = LlamaConfig.from_dict(model_config.text_config.to_dict())
            print(f"self.llm_config.max_seq_len: {self.llm_config.max_seq_len}")
        else:
            self.llm_config = model_config

        self.max_seq_len = self.llm_config.max_seq_len
        self.model_type = model_config.model_type
        self.model = model
        self.model_runner = None

        if max_gpu_num_blocks:
            self.kv_mem_manager = self._init_mem_manager(max_gpu_num_blocks)
            self.max_gpu_num_tokens = max_gpu_num_blocks
        else:
            max_gpu_num_blocks, self.max_gpu_num_tokens = (
                self._get_max_avaliable_tokens(model,gpu_memory_utilization=0.9, block_size=1)
            )
            self.kv_mem_manager = self._init_mem_manager(
                max_gpu_num_blocks, block_size=1
            )

        self.max_request_num = max_gpu_num_blocks // self.max_seq_len

        self.req_tokens_manager = ReqTokensManager(
            self.max_request_num, self.max_seq_len
        )
        self.atten_info = AttentionInfo()  # 创建 AttentionInfo 实例
        self.atten_info.kv_buffer = self.kv_mem_manager.gpu_kv_buffer
        self.atten_info.b_req_tokens_table = self.req_tokens_manager.b_req_tokens_table

        # TODO apply_cuda_graph 新代码有 bug，已经删去，后续等待修复
        self.compiled_model = False
        if self.compiled_model:
            self.apply_cuda_graph()  # 调用 cuda graph 优化

    def _get_max_avaliable_tokens(self,model, gpu_memory_utilization=0.9, block_size=1):
        avaliable_blocks = ComputeMaxAvailableBlocks(
            num_layers=self.llm_config.num_layers,
            hidden_size=self.llm_config.hidden_size,
            num_heads=self.llm_config.num_heads,
            num_kv_heads=self.llm_config.num_kv_heads,
            head_dim=self.llm_config.head_dim,
            gpu_memory_utilization=gpu_memory_utilization,
            block_size=block_size,
        )
        max_gpu_num_blocks = avaliable_blocks.compute_num_available_blocks(model, model_path=self.checkpoints_dir)
        max_gpu_num_tokens = max_gpu_num_blocks * block_size

        return max_gpu_num_blocks, max_gpu_num_tokens

    def _init_mem_manager(
        self, gpu_num_blocks, block_size=1, dtype=torch.float16, device="cuda"
    ):
        kv_mem_manager = KVCacheMemoryManager(
            num_layers=self.llm_config.num_layers,
            num_kv_heads=self.llm_config.num_kv_heads,
            head_dim=self.llm_config.head_dim,
            gpu_num_blocks=gpu_num_blocks,
            block_size=block_size,
            dtype=dtype,
            device=device,
        )

        return kv_mem_manager

    # TODO: 实现apply_cuda_graph
    def apply_cuda_graph(
        self,
    ):
        """应用 cuda graph 优化
        参数:
            - input_ids: 输入 tokens id 列表, shape: (batch_size, 1)
            - prev_pos: 当前处于第几轮迭代循环, 生成第几个 token
        """
        # self.model_runner = ModelRunner(
        #     self.model,
        #     self.llm_config,
        #     self.max_gpu_num_tokens,
        #     self.kv_mem_manager,
        #     self.req_tokens_manager,
        # )
        # self.model_runner.capture_decode_graph()

    def init_req_to_tokens_table(
        self, b_req_tokens_table, b_req_idx, b_seq_len, alloc_mem_index
    ):
        """
        初始化请求到token索引的映射表（在prefill阶段使用）
        
        这个函数的作用是为每个请求建立从序列位置到KV缓存索引的映射关系。
        在prefill阶段，每个请求的所有token会一次性处理，需要预先分配连续的KV缓存空间。
        
        参数说明：
            b_req_tokens_table (torch.Tensor): 请求到token索引的映射表
                形状: (max_request_num, max_seq_len)
                用途: 存储每个请求中每个位置对应的KV缓存索引
                
            b_req_idx (torch.Tensor): 批次中每个请求的ID
                形状: (batch_size,)
                含义: 标识每个请求在全局请求池中的索引
                
            b_seq_len (torch.Tensor): 每个请求的序列长度
                形状: (batch_size,)
                含义: 每个请求包含的token数量
                
            alloc_mem_index (torch.Tensor): 已分配的连续内存索引
                形状: (total_tokens,)
                含义: prefill阶段为所有token分配的连续KV缓存索引
        
        返回：
            b_start_loc (torch.Tensor): 每个请求在alloc_mem_index中的起始位置
                形状: (batch_size,)
                用途: 记录每个请求的token在连续索引中的起始位置
        
        工作原理：
            1. 遍历每个请求
            2. 根据请求的序列长度，从alloc_mem_index中取出对应数量的连续索引
            3. 将这些索引填充到b_req_tokens_table[req_id, :seq_len]位置
            4. 记录每个请求的起始位置用于后续计算
        
        简单例子：
            假设有2个请求，序列长度分别为3和2：
            
            # 输入
            b_req_idx = [0, 1]           # 请求0和请求1
            b_seq_len = [3, 2]           # 请求0有3个token，请求1有2个token
            alloc_mem_index = [10, 11, 12, 13, 14]  # 分配的连续KV缓存索引
            
            # 处理过程
            # 请求0: 取索引[10, 11, 12]，填充到b_req_tokens_table[0, 0:3]
            # 请求1: 取索引[13, 14]，填充到b_req_tokens_table[1, 0:2]
            
            # 输出
            b_req_tokens_table = [
                [10, 11, 12, 0, 0, ...],  # 请求0的KV索引映射
                [13, 14, 0,  0, 0, ...],  # 请求1的KV索引映射
                ...
            ]
            b_start_loc = [0, 3]         # 请求0从位置0开始，请求1从位置3开始
        """
        # TODO: 性能等待优化 - 当前使用CPU计算，可考虑GPU加速
        start_index = 0  # 在alloc_mem_index中的当前位置
        batch_size = len(b_seq_len)
        
        # 转换为numpy数组以便CPU上快速索引（避免GPU-CPU数据传输开销）
        b_seq_len_numpy = b_seq_len.cpu().numpy()
        b_req_idx_numpy = b_req_idx.cpu().numpy()
        
        # 初始化每个请求在连续索引中的起始位置记录
        b_start_loc = torch.zeros((batch_size,), dtype=torch.int32, device=self.device)
        
        # 遍历批次中的每个请求
        for i in range(batch_size):
            if i > 0:
                # 记录当前请求在alloc_mem_index中的起始位置
                b_start_loc[i] = start_index
                
            cur_seq_len = b_seq_len_numpy[i]  # 当前请求的序列长度
            cur_req_idx = b_req_idx_numpy[i]  # 当前请求的全局ID
            
            # 将分配的连续KV索引填充到请求映射表的对应位置
            # b_req_tokens_table[请求ID, 0:序列长度] = 连续的KV缓存索引
            b_req_tokens_table[cur_req_idx, :cur_seq_len] = alloc_mem_index[
                start_index : start_index + cur_seq_len
            ]
            
            # 更新下一个请求的起始位置
            start_index += cur_seq_len

        return b_start_loc

    def prefill_alloc_kv_cache(
        self,
        max_prompt_len,
        actual_prompt_lens,
        b_req_idx,
        image_batch_size=None,
        debug_mode=False,
    ):
        """
        Prefill阶段的KV缓存分配函数
        
        在大模型推理中，处理分为两个阶段：
        1. Prefill阶段：处理输入的prompt序列，为所有input token计算并缓存KV
        2. Decode阶段：逐个生成新token，每次只需计算一个新token的KV
        
        这个函数负责prefill阶段的KV缓存分配和初始化工作。
        
        参数说明：
            max_prompt_len (int): 批次中最长prompt的长度
                用途: 决定需要分配的最大缓存空间，确保所有请求都能容纳
                
            actual_prompt_lens (torch.Tensor): 每个请求的实际prompt长度
                形状: (batch_size,)
                含义: 批次中每个请求真实的token数量（可能小于max_prompt_len）
                
            b_req_idx (torch.Tensor): 批次中每个请求的全局ID
                形状: (batch_size,)
                用途: 标识每个请求在全局请求池中的位置
                
            image_batch_size (int, optional): 图像批次大小（多模态模型使用）
                用途: 对于视觉-语言模型，需要额外考虑图像token的空间
                
            debug_mode (bool): 是否开启调试模式
                用途: 打印详细的分配信息用于调试
        
        返回：
            tuple: (cur_select_index, num_patch_indexs)
                cur_select_index: 分配的KV缓存索引
                num_patch_indexs: 图像patch的数量（多模态模型）
        
        工作流程：
            1. 计算总需要的token数量（包括图像token）
            2. 一次性分配连续的KV缓存空间
            3. 初始化attention信息结构体
            4. 建立请求到token索引的映射关系
        
        实际例子：
            假设有2个请求的批次处理：
            
            # 输入参数
            max_prompt_len = 15        # 最长prompt有15个token
            actual_prompt_lens = [12, 8]  # 实际长度：请求0有12个token，请求1有8个token
            b_req_idx = [0, 1]         # 请求ID
            
            # 计算过程
            batch_size = 2
            context_num_tokens = 15 * 2 = 30  # 总共需要30个KV缓存位置
            
            # 分配结果（假设分配到索引100-129）
            cur_select_index = [100, 101, 102, ..., 129]  # 连续的30个索引
            
            # attention信息更新
            atten_info.b_seq_len = [12, 8]      # 实际序列长度
            atten_info.max_actual_seq_len = 15   # 最大序列长度
            atten_info.b_start_loc = [0, 15]     # 每个请求的起始位置
            
            # KV索引映射建立
            b_req_tokens_table[0, 0:12] = [100, 101, ..., 111]  # 请求0的映射
            b_req_tokens_table[1, 0:8]  = [115, 116, ..., 122]  # 请求1的映射
        
        注意事项：
            1. 这个函数只在prefill阶段调用一次
            2. 分配的空间按最大长度计算，可能存在浪费但保证安全
            3. 多模态模型需要额外考虑图像patch的空间需求
            4. 所有的索引分配都是连续的，便于高效的内存访问
        """
        num_patch_indexs = None  # 图像patch索引数量，用于多模态模型
        batch_size = len(actual_prompt_lens)
        
        # 设置attention信息中的请求ID
        self.atten_info.b_req_idx = b_req_idx

        # 处理多模态模型的图像token（如LLaVA等视觉-语言模型）
        if image_batch_size is not None:
            # 从模型配置中获取图像相关参数
            image_size = self.model_config.vision_config.image_size      # 图像尺寸
            pathch_size = self.model_config.vision_config.patch_size     # patch尺寸
            number_patchs = image_size // pathch_size                    # 每个维度的patch数量
            
            # 计算图像token数量：patch数量的平方减1（通常减去class token）
            num_patch_indexs = number_patchs * number_patchs - 1
            
            # 更新prompt长度以包含图像token
            max_prompt_len += num_patch_indexs
            actual_prompt_lens += num_patch_indexs
            
            if debug_mode:
                print(f"处理多模态输入 - 图像patch数量: {num_patch_indexs}")

        # 计算prefill阶段需要的总token数量
        # 使用最大长度确保所有请求都有足够空间
        context_num_tokens = max_prompt_len * batch_size
        
        # 一次性分配连续的KV缓存索引
        # 这样做的好处是：
        # 1. 内存访问连续，提高缓存命中率
        # 2. 简化索引管理逻辑
        # 3. 减少内存碎片
        self.atten_info.cur_select_index, _ = self.kv_mem_manager.alloc_kvcache_index(
            context_num_tokens
        )
        
        # 设置attention信息的关键参数
        self.atten_info.b_seq_len = actual_prompt_lens           # 每个请求的实际长度
        self.atten_info.max_actual_seq_len = max_prompt_len      # 批次的最大长度
        
        # 初始化请求到token索引的映射表
        # 这是核心步骤：建立从"请求-位置"到"KV缓存索引"的映射
        self.atten_info.b_start_loc = self.init_req_to_tokens_table(
            self.atten_info.b_req_tokens_table,    # 映射表（输出）
            self.atten_info.b_req_idx,             # 请求ID列表
            self.atten_info.b_seq_len,             # 序列长度列表
            self.atten_info.cur_select_index,      # 分配的连续索引
        )

        # 调试模式：打印详细的分配信息
        if debug_mode:
            print(f"""
            Prefill阶段KV缓存分配详情:
            - 总token数量: {context_num_tokens}
            - 最大prompt长度: {max_prompt_len}
            - 分配的KV索引: {self.atten_info.cur_select_index}
            - 最大实际序列长度: {self.atten_info.max_actual_seq_len}
            - 各请求序列长度: {self.atten_info.b_seq_len}
            - 各请求起始位置: {self.atten_info.b_start_loc}
            """)

        return self.atten_info.cur_select_index, num_patch_indexs

    def decode_alloc_kv_cache(self, batch_size):
        """
        Decode阶段的KV缓存分配函数
        
        在decode阶段，模型会逐个生成新的token。每生成一个新token，就需要：
        1. 为这个新token分配KV缓存空间
        2. 更新请求的序列长度
        3. 更新KV索引映射关系
        
        与prefill阶段不同，decode阶段每次只处理每个请求的一个新token，
        因此分配的是非连续的、分散的KV缓存空间。
        
        参数说明：
            batch_size (int): 当前批次中的请求数量
                含义: 需要为batch_size个新token分配KV缓存空间
                注意: 每个请求在decode阶段每轮只生成一个token
        
        返回：
            cur_select_index (torch.Tensor): 新分配的KV缓存索引
                形状: (batch_size,)
                含义: 每个请求新生成token的KV缓存位置
        
        工作流程：
            1. 为batch_size个新token分配非连续的KV缓存索引
            2. 使用update_kv_index内核更新索引映射表
            3. 增加所有请求的序列长度
            4. 更新全局最大序列长度
        
        详细例子：
            假设当前有2个请求正在decode阶段：
            
            # decode前的状态
            batch_size = 2
            atten_info.b_req_idx = [0, 1]          # 请求ID
            atten_info.b_seq_len = [15, 12]        # 当前序列长度
            atten_info.max_actual_seq_len = 15     # 最大序列长度
            
            # 请求映射表当前状态（简化显示）
            b_req_tokens_table = [
                [100, 101, ..., 114, 0, 0, ...],   # 请求0: 已有15个token的KV索引
                [200, 201, ..., 211, 0, 0, ...],   # 请求1: 已有12个token的KV索引
                ...
            ]
            
            # 调用decode_alloc_kv_cache(2)
            
            # 1. 分配新的KV缓存索引（假设分配到500, 501）
            cur_select_index = [500, 501]
            
            # 2. 调用update_kv_index更新映射表
            # 对于请求0: 将索引500存储到位置[0][15]
            # 对于请求1: 将索引501存储到位置[1][12]
            
            # 3. 更新序列长度
            atten_info.b_seq_len = [16, 13]        # 每个请求长度+1
            atten_info.max_actual_seq_len = 16     # 最大长度+1
            
            # decode后的映射表状态
            b_req_tokens_table = [
                [100, 101, ..., 114, 500, 0, ...], # 请求0: 新增索引500
                [200, 201, ..., 211, 501, 0, ...], # 请求1: 新增索引501  
                ...
            ]
            
            # 返回新分配的索引
            return [500, 501]
        
        性能考虑：
            1. decode阶段的分配是非连续的，可能导致内存碎片
            2. 使用GPU内核update_kv_index进行并行索引更新
            3. 每次decode只需分配很少的空间（通常就是batch_size个）
        
        与prefill阶段的区别：
            - prefill: 一次性分配大量连续空间
            - decode: 每次分配少量非连续空间
            - prefill: 处理整个prompt序列
            - decode: 每次只处理一个新token
        """
        # 为当前批次的每个请求分配一个新的KV缓存位置
        # 注意：这里分配的索引可能是非连续的，因为之前可能有其他请求释放了空间
        self.atten_info.cur_select_index, _ = self.kv_mem_manager.alloc_kvcache_index(
            batch_size
        )
        
        # 使用GPU内核更新KV索引映射表
        # 这个操作将新分配的索引正确地放置到每个请求的映射表中
        # 具体位置是：b_req_tokens_table[req_id][当前序列长度]
        update_kv_index(
            self.atten_info.b_req_tokens_table,    # 请求到token的映射表
            self.atten_info.b_req_idx,             # 当前批次的请求ID
            self.atten_info.b_seq_len,             # 当前各请求的序列长度
            self.atten_info.cur_select_index,      # 新分配的KV缓存索引
        )

        # 更新序列长度信息
        # 每个请求都生成了一个新token，所以长度都增加1
        self.atten_info.b_seq_len += 1
        
        # 更新全局最大序列长度
        # 在decode阶段，最大长度会逐步增长
        self.atten_info.max_actual_seq_len += 1

        # 返回新分配的KV缓存索引，供attention计算使用
        return self.atten_info.cur_select_index  # shape [batch_size,]

    def forward(self, input_ids, position_ids, image_tensor=None):
        """
        模型前向推理的统一接口
        
        这是ModelExecutor的核心推理函数，负责调用底层模型进行前向计算。
        该函数会根据模型类型（是否为多模态模型）选择合适的调用方式。
        
        参数说明：
            input_ids (torch.Tensor): 输入的token ID序列
                形状: (batch_size, seq_len) 或 (total_tokens,)
                含义: 要处理的token的数值ID，由分词器生成
                用途: 作为模型的主要文本输入
                
            position_ids (torch.Tensor): 位置编码ID
                形状: 与input_ids相同
                含义: 每个token在序列中的位置信息
                用途: 帮助模型理解token的相对和绝对位置
                
            image_tensor (torch.Tensor, optional): 图像张量（多模态模型使用）
                形状: (batch_size, channels, height, width)
                含义: 预处理后的图像数据
                用途: 为视觉-语言模型提供视觉输入
        
        返回：
            logits (torch.Tensor): 模型输出的logits
                形状: (batch_size, vocab_size) 或 (total_tokens, vocab_size)
                含义: 每个位置上词表中每个token的未归一化概率
                用途: 可通过softmax转换为概率分布，用于token采样
        
        工作原理：
            1. 检查模型类型（是否为多模态模型如LLaVA）
            2. 根据模型类型调用相应的forward方法
            3. 传递attention信息（包含KV缓存索引等）
            4. 返回计算结果
        
        使用例子：
            # 纯文本模型推理
            input_ids = torch.tensor([[1, 2, 3, 4]])      # 1个请求，4个token
            position_ids = torch.tensor([[0, 1, 2, 3]])   # 对应的位置ID
            logits = executor.forward(input_ids, position_ids)
            # logits.shape = (1, vocab_size)
            
            # 多模态模型推理（如LLaVA）
            input_ids = torch.tensor([[1, 2, 3, 4]])      # 文本输入
            position_ids = torch.tensor([[0, 1, 2, 3]])   # 位置信息
            image_tensor = torch.randn(1, 3, 224, 224)    # 图像输入
            logits = executor.forward(input_ids, position_ids, image_tensor)
            # logits.shape = (1, vocab_size)
        
        attention信息的作用：
            self.atten_info包含了推理所需的关键信息：
            - b_req_tokens_table: 请求到token的KV索引映射
            - cur_select_index: 当前使用的KV缓存索引
            - b_seq_len: 各请求的序列长度
            - max_actual_seq_len: 最大序列长度
            - kv_buffer: KV缓存缓冲区
            
            这些信息帮助模型正确地：
            1. 找到每个token对应的KV缓存位置
            2. 实现高效的attention计算
            3. 支持变长序列的批量处理
        
        模型类型区别：
            - 普通语言模型: 只需要文本输入，直接进行transformer计算
            - 多模态模型: 需要额外处理图像输入，通常包含视觉编码器
        
        注意事项：
            1. 调用此函数前必须先调用prefill_alloc_kv_cache或decode_alloc_kv_cache
            2. attention信息必须正确初始化
            3. 对于多模态模型，image_tensor不能为None
            4. 返回的logits需要进一步处理（如采样）才能得到最终的token
        """
        # 根据模型类型选择合适的forward调用方式
        if self.model_type == "llava":
            # 多模态模型（如LLaVA）：需要处理图像和文本输入
            # 图像会先通过视觉编码器处理，然后与文本特征融合
            logits = self.model.forward(
                input_ids,              # 文本token序列
                position_ids,           # 位置编码
                self.atten_info,        # attention相关信息（KV缓存等）
                image_tensor            # 图像输入张量
            )
        else:
            # 纯文本模型：标准的transformer forward
            # 只需要处理文本输入和位置信息
            logits = self.model.forward(
                input_ids,              # 文本token序列  
                position_ids,           # 位置编码
                self.atten_info         # attention相关信息（KV缓存等）
            )
            
        return logits