from typing import Optional
import torch
from typing import Optional, TypedDict, Generator
from light_kinfer.executor.model_executor import ModelExecutor
from light_kinfer.utils.file_interface import get_model_name_from_path
from light_kinfer.kernels.softmax_split import softmax_split

from transformers import AutoTokenizer

class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: list[str]   # not required
    logprobs: list[float] # not required

@torch.inference_mode()
def sample_top_p(probs, p):
    """
    执行 Top-p (Nucleus) 采样, 从概率分布中采样下一个词。

    参数：
        probs (torch.Tensor): 概率分布张量，形状为 `[batch_size, vocab_size]`。
        p (float): 累积概率阈值，取值范围在 0 到 1 之间。
    返回：
        torch.Tensor: 采样得到的词索引，形状为 `[batch_size, 1]`。

    说明：
        Top-p 采样算法: 选择概率累积和超过阈值 p 的最小集合，将这些词的概率重新归一化后进行采样。
    """
    # 对概率分布进行降序排序。probs_sort: 排序后的概率值，形状与 probs 相同。probs_idx: 排序后的索引，用于映射回原始词汇表。
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # 计算排序后概率的累积和. 返回的 probs_sum 是累积概率分布。
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # 保留累积概率未超过阈值 p 的词汇的概率，其余词汇的概率被置为 0.0。
    mask = (
        probs_sum - probs_sort > p
    )  # 创建掩码，对于每个位置，计算累积概率（不包括当前词）是否超过阈值 p。
    probs_sort[mask] = 0.0  # 将累积概率超过阈值 p 的词的概率置零。

    # 对剩余的概率重新归一化, 确保总和为 1。
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # 从重新归一化的概率分布中采样下一个词. 返回的 next_token 是采样得到的词在排序后概率分布中的索引。
    next_token_sorted_idx = torch.multinomial(probs_sort, num_samples=1)
    # 在 probs_idx 的最后一维（dim=-1）中，使用 next_token_sorted_idx 作为索引，提取对应的值。沿着 dim=1（列）进行索引提取
    # NOTE: torch.gather 函数按照给定的索引张量 index，从输入张量中收集 (获取) 数据，并返回一个与索引张量形状一致的张量。
    next_token = torch.gather(probs_idx, -1, index=next_token_sorted_idx)

    return next_token  # 返回采样得到的下一个词的索引


class GenerateStreamText:
    """
    流式文本生成类 - 实现LLaMA模型的流式推理和文本生成
    
    功能:
    - 加载预训练的LLaMA模型和tokenizer
    - 执行流式文本生成，实时输出生成的token
    - 支持批量处理多个输入序列
    - 优化内存管理，支持大规模序列生成
    
    核心特性:
    1. 流式输出: 每生成一个token就立即返回，无需等待完整序列
    2. 温度采样: 支持可控的随机性文本生成
    3. Top-p采样: 核采样策略，平衡质量和多样性
    4. KV缓存管理: 高效的注意力缓存机制
    5. 批量处理: 同时处理多个输入序列
    """

    def __init__(
        self,
        checkpoints_dir: str,
        tokenizer_path: str,
        max_gpu_num_blocks=None,
        max_seq_len=1024,
        compiled_model=False,
        device="cuda",
    ):
        """
        初始化流式文本生成器
        
        参数:
            checkpoints_dir: 模型检查点目录路径
            tokenizer_path: tokenizer文件路径
            max_gpu_num_blocks: GPU上的最大内存块数，用于KV缓存管理
            max_seq_len: 支持的最大序列长度
            compiled_model: 是否使用编译优化的模型
            device: 运行设备 ("cuda" 或 "cpu")
        """
        self.checkpoints_dir = checkpoints_dir

        # 构建模型执行器，负责模型加载和推理
        self.model_executor = ModelExecutor.build(
            checkpoints_dir=checkpoints_dir,
            max_gpu_num_blocks=max_gpu_num_blocks,
            max_seq_len=max_seq_len,
            compiled_model=compiled_model,
            device=device,
        )
        
        # 加载对应的tokenizer
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        self.model_config = self.model_executor.model_config
        self.device = device

    def load_tokenizer(self, pretrained_model_name_or_path):
        """
        根据模型类型加载相应的tokenizer
        
        参数:
            pretrained_model_name_or_path: tokenizer的路径或模型名称
            
        返回:
            加载好的tokenizer对象
            
        实现原理:
        - 检测模型类型（如llava多模态模型）
        - 选择合适的tokenizer配置（fast/slow模式）
        - 启用远程代码信任以支持自定义tokenizer
        """
        model_name = get_model_name_from_path(pretrained_model_name_or_path)

        if "llava" in model_name.lower():
            # LLaVA多模态模型使用慢速tokenizer，避免兼容性问题
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, use_fast=False, trust_remote_code=True
            )
        else:
            # 标准文本模型使用快速tokenizer，提升性能
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, use_fast=True, trust_remote_code=True
            )

        return tokenizer

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt_tokens: list[list[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        echo: bool = False,
        device="cuda",
    ) -> Generator[tuple[list[str], Optional[list[float]]], None, None]:
        """
        核心流式生成函数 - 基于prompt逐token生成文本并实时输出
        
        参数：
            prompt_tokens: 已分词的输入序列列表，每个元素是一个token ID列表
            max_gen_len: 最大生成长度
            temperature: 温度参数，控制生成的随机性 (0=确定性, >0=随机性)
            top_p: 核采样阈值，控制采样的词汇范围
            echo: 是否在输出中包含原始prompt
            device: 计算设备
            
        yield输出：
            Generator[list[str], None, None]: 每次yield一个批次的生成文本列表
            
        实现原理:
        1. 初始化阶段: 准备tokens张量、掩码、状态跟踪
        2. Prefill阶段: 一次性处理所有prompt tokens
        3. Decode阶段: 逐个生成新token，每生成一个就立即输出
        4. 内存管理: 动态分配和释放KV缓存
        """
        # === 1. 批次和序列长度初始化 ===
        bsz = len(prompt_tokens)  # 批次大小
        max_prompt_len = max(len(t) for t in prompt_tokens)  # 最长prompt长度
        assert max_prompt_len <= self.model_config.max_seq_len
        
        # 计算总的序列长度：prompt + 生成部分，不超过模型最大长度
        total_len = min(self.model_config.max_seq_len, max_gen_len + max_prompt_len)
        
        # 记录每个序列的实际prompt长度，用于后续的掩码和位置管理
        actual_prompt_lens = torch.tensor(
            [len(t) for t in prompt_tokens], dtype=torch.long, device=device
        )
        
        # 确定填充token ID，优先使用pad_token，否则使用eos_token
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        # === 2. 张量和状态初始化 ===
        # 预分配完整的tokens张量，初始化为padding token
        # 形状: [batch_size, total_len] - 包含prompt和将要生成的部分
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        
        # 输入文本掩码：True表示实际输入内容，False表示padding
        input_text_mask = tokens != pad_id
        
        # EOS状态跟踪：记录每个序列是否已生成结束token
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        
        prev_pos = 0  # 上一次处理的位置
        
        # 流式输出位置跟踪：记录每个序列已经输出到的位置
        # 如果echo=True，从0开始输出（包含prompt）；否则从prompt结尾开始
        last_yielded_pos = [
            len(prompt_tokens[i]) if not echo else 0 for i in range(bsz)
        ]

        # === 3. 填充prompt tokens到预分配的张量中 ===
        for k, t in enumerate(prompt_tokens):
            # 将每个prompt的token填入对应的行，从位置0开始
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

        # === 4. KV缓存管理初始化 ===
        b_req_idx = torch.arange(bsz, device=self.device)  # 批次索引
        all_select_index_list = []  # 记录所有分配的KV缓存索引，用于最后释放
        
        # Prefill阶段的KV缓存分配：为prompt序列分配缓存空间
        prefill_select_index, _ = self.model_executor.prefill_alloc_kv_cache(
            max_prompt_len, actual_prompt_lens, b_req_idx
        )
        all_select_index_list.append(prefill_select_index)

        # === 5. Prefill阶段：处理完整的prompt ===
        input_ids = tokens[:, :max_prompt_len]  # 提取prompt部分 [batch_size, max_prompt_len]
        
        # === 6. Decode循环：逐个token生成阶段 ===
        for cur_pos in range(max_prompt_len, total_len):
            # 6.1 准备当前推理的输入
            # 获取从上一个位置到当前位置的token序列
            # Prefill阶段: input_ids是完整prompt；Decode阶段: input_ids是单个新token
            input_ids = tokens[:, prev_pos:cur_pos]
            batch_size, seq_len = input_ids.shape
            
            # 生成对应的位置编码：从prev_pos开始的连续位置序列
            # 形状变化: [seq_len] -> [1, seq_len] -> [batch_size, seq_len]
            position_ids = (
                torch.arange(prev_pos, prev_pos + seq_len, device=input_ids.device)
                .unsqueeze(0)  # 增加batch维度: [1, seq_len]
                .repeat(batch_size, 1)  # 复制到每个样本: [batch_size, seq_len]
            )

            # 6.2 模型前向推理
            # 输入: input_ids [batch_size, seq_len], position_ids [batch_size, seq_len]
            # 输出: logits [batch_size, seq_len, vocab_size] - 每个位置的词汇概率分布
            logits = self.model_executor.forward(input_ids, position_ids)
            
            # 为decode阶段分配新的KV缓存空间
            decode_select_index = self.model_executor.decode_alloc_kv_cache(bsz)
            all_select_index_list.append(decode_select_index)

            # 6.3 下一个token采样
            # 只关注序列最后一个位置的logits（即下一个要生成的token）
            # logits[:, -1]形状: [batch_size, vocab_size]
            if temperature > 0:
                # 随机采样模式：使用温度调节和top-p采样
                # 温度缩放：temperature越大，分布越平滑，随机性越强
                probs = softmax_split(logits[:, -1] / temperature)
                # Top-p核采样：从累积概率达到top_p的最小词汇集合中采样
                next_token = sample_top_p(probs, top_p)
            else:
                # 确定性模式：直接选择概率最大的token
                next_token = torch.argmax(logits[:, -1], dim=-1)

            # 6.4 更新token序列
            input_ids = next_token  # 形状: [batch_size, 1] - 新生成的token
            
            # 构建掩码：判断当前位置是否需要更新（非prompt部分）
            # input_text_mask[:, cur_pos]: 当前位置的掩码，True表示原始输入
            # ~input_text_mask[:, cur_pos]: 取反，True表示需要生成的位置
            mask = ~input_text_mask[:, cur_pos]  # [batch_size]
            
            # 条件更新：只在需要生成的位置填入新token，保持prompt不变
            # torch.where(condition, x, y): 条件为True选择x，否则选择y
            tokens[:, cur_pos] = torch.where(
                mask, next_token.reshape(-1), tokens[:, cur_pos]
            )

            # 6.5 EOS检测：判断是否生成了结束token
            # 逻辑：当前位置是生成部分 AND 生成的token是EOS token
            eos_reached = eos_reached | (
                mask & (next_token == self.tokenizer.eos_token_id)
            )
            prev_pos = cur_pos  # 更新处理位置

            # 6.6 流式输出处理：实时生成文本并yield
            batch_outputs = []  # 收集当前批次的所有输出
            for i in range(bsz):
                # 确定当前样本的输出范围
                start = last_yielded_pos[i]  # 上次已输出的位置
                end = cur_pos + 1            # 当前位置+1（包含新生成的token）
                
                if start < end:
                    # 有新内容需要输出
                    # 提取新生成的token序列并转换为Python列表
                    token = tokens[i, start:end].tolist()
                    
                    # 解码token为文本，跳过特殊标记以提高可读性
                    text = self.tokenizer.decode(
                        token, skip_special_tokens=True
                    )
                    batch_outputs.append(text)
                    
                    # 更新已输出位置
                    last_yielded_pos[i] = end
                else:
                    # 没有新内容，添加空字符串保持批次一致性
                    batch_outputs.append("")

            # 6.7 流式输出：yield当前批次的所有生成文本
            yield batch_outputs

            # 6.8 提前终止条件：所有序列都生成了EOS token
            if eos_reached.all():
                break

        # === 7. 内存清理：释放KV缓存资源 ===
        # 合并所有分配的缓存索引并释放，避免内存泄漏
        all_select_indexs = torch.concat(all_select_index_list)
        self.model_executor.kv_mem_manager.release_ref(all_select_indexs)

    def text_completion_stream(
        self,
        prompts: list[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        echo: bool = False,
    ) -> Generator[list[CompletionPrediction], None, None]:
        """
        高级文本补全流式接口 - 将字符串prompt转换为流式文本生成
        
        参数:
            prompts: 输入的文本prompt列表
            temperature: 采样温度，控制随机性
            top_p: 核采样阈值
            max_gen_len: 最大生成长度，默认为模型最大序列长度-1
            echo: 是否在输出中包含原始prompt
            
        yield输出:
            Generator[list[CompletionPrediction], None, None]: 
            每次yield包含所有prompt的补全结果，支持增量更新
            
        实现原理:
        1. 文本预处理: 将字符串prompt转换为token序列
        2. 流式生成调用: 调用底层generate_stream方法
        3. 结果聚合: 将流式输出累积为完整的补全结果
        4. 格式化输出: 按照CompletionPrediction格式返回
        """
        # 设置默认生成长度：模型最大长度减去1个位置（留给特殊token）
        if max_gen_len is None:
            max_gen_len = self.model_config.max_seq_len - 1

        # 文本tokenization：将字符串转换为token ID序列
        # add_special_tokens=True 确保添加必要的开始/结束标记
        prompt_tokens = [
            self.tokenizer.encode(x, add_special_tokens=True) for x in prompts
        ]

        # 调用底层流式生成方法
        stream = self.generate_stream(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            echo=echo,
        )

        # 初始化每个prompt的补全结果容器
        # CompletionPrediction格式: {"generation": str, "tokens": list[str]}
        completions = [{"generation": "", "tokens": []} for _ in prompts]
        
        # 流式处理：逐步累积生成的文本
        for batch_outputs in stream:
            # 更新每个prompt的生成结果
            for i, text in enumerate(batch_outputs):
                # 累积拼接新生成的文本片段
                completions[i]["generation"] += text
            
            # yield当前状态的完整结果副本
            # 使用copy()避免引用问题，确保每次yield的都是独立状态
            yield completions.copy()