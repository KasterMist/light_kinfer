import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import json

from typing import Optional, Tuple, Union
from pathlib import Path

from light_kinfer.models.model_config import LlamaConfig

# -------------------- RMSNorm -------------------- #

class LlamaRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    RMSNorm 公式:
        1. 计算均方根: rms = sqrt(mean(x^2) + eps)
        2. 归一化: x_norm = x / rms
        3. 缩放: output = weight * x_norm
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) # 预分配权重位置
        self.eps = eps  # variance epsilon, 避免除零错误

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)  # 提升精度
        variance = x.pow(2).mean(-1, keepdim=True) + self.eps  # 对最后一个维度计算均值, [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, 1]
        x = x * torch.rsqrt(variance)  # 归一化, torch.rsqrt为计算平方根的倒数
        return self.weight * x.to(input_dtype)  # 恢复原始数据类型并缩放

# --------------------- Rotary Embeddings --------------------- #

class LlamaRotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings
    公式:
        1. 计算频率: freq = 1 / (10000^(2i/dim))
        2. 计算位置编码: pos_enc = [sin(pos * freq), cos(pos * freq)]
        3. 应用位置编码:
            x_rotated = x * cos(pos_enc) + rotate_half(x) * sin(pos_enc)
    """
    # dim: 词嵌入维度
    # max_position_embeddings: 最大位置编码长度
    # base: 频率基数
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 计算频率 公式: freq = 1 / (10000^(2i/dim)), 在dim处，每两个维度使用相同的频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # 注册为buffer, 不作为模型参数

    def forward(self, x, seq_len = None):
        # x: [batch_size, seq_len, num_heads, head_dim]
        if seq_len is None:
            seq_len = x.shape[1]
        
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype) # 功能: 生成位置索引, [seq_len], WARNING: 仍然会有重复回答问题的可能性，建议使用int64
        # t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq) # 功能: 生成位置索引, [seq_len] ERROR: 会导致重复回答问题！
        freqs = torch.outer(t, self.inv_freq) # 功能: 计算位置与频率的外积, [seq_len, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # 复制频率以匹配维度, [seq_len, dim], 后续会调用rotate_half来保证正确性
        cos = emb.cos()
        sin = emb.sin()

        # Return with proper shape: [1, seq_len, 1, head_dim]
        # Example: 
        # cos = [cos(θ0), cos(θ1), cos(θ2), cos(θ3), cos(θ0), cos(θ1), cos(θ2), cos(θ3)]
        # sin = [sin(θ0), sin(θ1), sin(θ2), sin(θ3), sin(θ0), sin(θ1), sin(θ2), sin(θ3)]
        return cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2] # 功能: 取前半部分, ... 表示所有前面的维度 i.e. [x0, x1, x2, x3]
    x2 = x[..., x.shape[-1] // 2 :] # 功能: 取后半部分  i.e. [x4, x5, x6, x7]
    return torch.cat((-x2, x1), dim=-1) # 功能: 交换前后部分并拼接, i.e. [-x4, -x5, -x6, -x7, x0, x1, x2, x3]

def apply_rotary_pos_emb(q, k, cos, sin, position_ids = None):
    """
    Apply rotary position embedding to query and key tensors.
    """
    # q: [batch_size, num_heads, seq_len, head_dim] - shape like [1, 32, 9, 64]
    # k: [batch_size, num_kv_heads, seq_len, head_dim] - shape like [1, 8, 9, 64]  
    # cos, sin: [1, seq_len, 1, head_dim] - shape like [1, 9, 1, 64]

    if position_ids is None:
        # Transpose cos and sin to match q and k dimensions
        # From [1, seq_len, 1, head_dim] to [1, 1, seq_len, head_dim]
        cos = cos.transpose(1, 2)
        sin = sin.transpose(1, 2)
    else:
        # 选择特定位置的cos和sin
        cos = cos.squeeze(0).squeeze(1)  # [seq_len, head_dim]
        sin = sin.squeeze(0).squeeze(1)  # [seq_len, head_dim]
        # example: position_ids.shape = batch_size, seq_len
        cos = cos[position_ids]  # [batch_size, seq_len, head_dim]
        sin = sin[position_ids]  # [batch_size, seq_len, head_dim]
        cos = cos.unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]
        sin = sin.unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]
    # Apply rotary embedding
    # i.e. x = x * cos + rotate_half(x) * sin
    # result = [
    #     x0·cos(θ0) - x4·sin(θ0),  # 第0维和第4维配对，使用频率θ0
    #     x1·cos(θ1) - x5·sin(θ1),  # 第1维和第5维配对，使用频率θ1
    #     x2·cos(θ2) - x6·sin(θ2),  # 第2维和第6维配对，使用频率θ2
    #     x3·cos(θ3) - x7·sin(θ3),  # 第3维和第7维配对，使用频率θ3
    #     x4·cos(θ0) + x0·sin(θ0),  # 第4维和第0维配对，使用频率θ0
    #     x5·cos(θ1) + x1·sin(θ1),  # 第5维和第1维配对，使用频率θ1
    #     x6·cos(θ2) + x2·sin(θ2),  # 第6维和第2维配对，使用频率θ2
    #     x7·cos(θ3) + x3·sin(θ3)   # 第7维和第3维配对，使用频率θ3
    # ]
    # 
    # 这实现了标准的2D旋转变换:
    # [x0'] = [cos(θ0) -sin(θ0)] [x0]     [x4'] = [cos(θ0) -sin(θ0)] [x4]
    # [x4']   [sin(θ0)  cos(θ0)] [x4] ... [x0']   [sin(θ0)  cos(θ0)] [x0]
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed

# --------------------- LLaMA Attention -------------------- #

class LlamaAttention(nn.Module):
    """
    Multi-head attention module
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.num_heads * config.head_dim   # 词嵌入维度
        self.num_heads = config.num_heads       # 注意力头数
        self.head_dim = config.head_dim if config.head_dim is not None else self.hidden_size // self.num_heads    # 每个头的维度
        self.num_key_value_heads = config.num_kv_heads  # key和value的头数, 用于多query单key/value机制
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # key/value头的重复次数
        self.max_position_embeddings = config.max_position_embeddings  # 最大位置编码长度
        self.rope_theta = config.rope_theta  # Rotary Embedding base

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=torch.float16)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, dtype=torch.float16)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, dtype=torch.float16)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=torch.float16)

        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    # 函数功能: 重塑张量形状计算以适配多头注意力机制的计算需求
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """
        输入: [batch_size, seq_len, hidden_size]
        输出: [batch_size, num_heads, seq_len, head_dim]
        """
        # tensor.view(bsz, seq_len, -1, self.head_dim) 中-1会自动计算为num_heads，因为hidden_size = num_heads * head_dim
        # 然后通过transpose交换seq_len和num_heads的位置
        return tensor.view(bsz, seq_len, -1, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,  # hidden_states包含了每个token在当前层的语义理解，形状为 [batch_size, seq_len, hidden_size]
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None, # past_key_value包含了之前生成的key和value，用于自回归生成
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        hidden_states: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, 1, seq_len, seq_len] or None
        position_ids: [batch_size, seq_len] or None
        past_key_value: Tuple of (key, value) each of shape [batch_size, num_kv_heads, past_seq_len, head_dim]
        """
        bsz, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 重塑张量形状以适配多头注意力机制
        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 计算当前序列的长度, 如果有past_key_value, 则加上过去的长度
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # 执行rotary position embeddings
        cos, sin = self.rotary_emb(key_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids=position_ids
        )

        past_key_value = (key_states, value_states) if use_cache else None

        # 实现分组查询注意力(Grouped Query Attention, GQA)机制
        # 假设 num_heads=32, num_kv_heads=8, num_key_value_groups=4
        # key_states: [batch_size, 8, seq_len, head_dim]
        # 经过 repeat_interleave 后: [batch_size, 32, seq_len, head_dim]
        # 重复模式: [k0,k0,k0,k0, k1,k1,k1,k1, ..., k7,k7,k7,k7]
        # TODO: 优化内存占用, 避免 repeat_interleave
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # 注意力权重的计算 Q * K^T / sqrt(d_k)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, seq_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, seq_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, seq_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, seq_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
        
        # 计算softmax时使用float32以提升数值稳定性
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # 与value进行矩阵乘法，作为最终的注意力输出
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, seq_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value


# --------------------- LLaMA MLP --------------------- #

class LlamaMLP(nn.Module):
    """
    Feed-forward network (MLP) module
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size # intermediate size是FFN中间层的维度，通常是hidden_size的4倍

        # SwiGLU(x) = SiLU(Linear1(x)) * Linear2(x) -> SiLU(self.gate_proj(x)) * self.up_proj(x) 
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)    # 用于Gated Linear Unit (GLU)的门控投影
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)      # 用于上采样的线性投影
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)    # 用于下采样的线性投影
        self.act_fn = nn.SiLU()  # SiLU激活函数

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
# --------------------- LLaMA Decoder Layer --------------------- #

class LlamaDecoderLayer(nn.Module):
    """
    Single decoder layer
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        hidden_states: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, 1, seq_len, seq_len] or None
        position_ids: [batch_size, seq_len] or None
        past_key_value: Tuple of (key, value) each of shape [batch_size, num_kv_heads, past_seq_len, head_dim]
        """

        # 计算残差和layernorm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # 残差连接
        hidden_states = residual + hidden_states

        # 全连接前馈网络
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    

# --------------------- LLaMA Model --------------------- #
class LlamaModel(nn.Module):
    """
    由 *config.num_hidden_layers* 层组成的Transformer模型，每一层都是一个 [`LlamaDecoderLayer`]
    这是LLaMA的核心模型，负责将输入token序列转换为隐藏状态表示
    """
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.bos_token_id   # 填充标记索引, 通常与bos_token_id相同
        self.vocab_size = config.vocab_size      # 词汇表大小

        # 词嵌入层, 功能: 将输入的token ID转换为对应的词向量
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # 创建多个解码器层，每层包含自注意力和前馈网络
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_layers)])
        # 最终的归一化层，应用在所有解码器层之后
        self.norm = LlamaRMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        """获取输入嵌入层"""
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """设置输入嵌入层"""
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,          # 输入的token ID张量, 形状为 [batch_size, seq_len]
        attention_mask: Optional[torch.Tensor] = None,     # 注意力掩码张量, 形状为 [batch_size, 1, seq_len, seq_len]
        position_ids: Optional[torch.Tensor] = None,       # 位置ID张量, 形状为 [batch_size, seq_len]
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, # 过去的key和value张量, 用于自回归生成
        inputs_embeds: Optional[torch.Tensor] = None,      # 预计算的输入嵌入张量, 形状为 [batch_size, seq_len, hidden_size]
        use_cache: Optional[bool] = None,                  # 是否返回新的past_key_values
        output_attentions: Optional[bool] = False,         # 是否返回注意力权重
        output_hidden_states: Optional[bool] = False,      # 是否返回所有层的隐藏状态
        return_dict: Optional[bool] = True,                # 是否以字典形式返回输出
    ):
        """
        LlamaModel的前向传播函数
        
        参数说明:
        - input_ids: 输入的token ID序列 [batch_size, seq_len]
        - attention_mask: 注意力掩码，用于忽略填充位置 [batch_size, seq_len]
        - position_ids: 位置编码ID [batch_size, seq_len]
        - past_key_values: 缓存的历史key/value，用于生成阶段的加速
        - inputs_embeds: 预计算的嵌入向量（可选，与input_ids二选一）
        - use_cache: 是否缓存key/value用于下次生成
        - output_attentions: 是否输出注意力权重
        - output_hidden_states: 是否输出所有层的隐藏状态
        - return_dict: 是否以字典形式返回结果
        """
        # 设置默认输出选项，优先使用传入参数，其次使用配置文件设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions if hasattr(self.config, "output_attentions") else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states if hasattr(self.config, "output_hidden_states") else False
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache if hasattr(self.config, "use_cache") else False

        # 1. 输入验证与形状推断
        # 确保input_ids和inputs_embeds不能同时指定
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_len = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_len, _ = inputs_embeds.shape 
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 2. 计算序列长度（包含历史缓存）
        seq_len_with_past = seq_len
        past_key_values_len = 0

        # 如果有历史缓存，计算历史序列长度
        if past_key_values is not None:
            past_key_values_len = past_key_values[0][0].shape[2]  # 从第一层的key张量获取历史长度
            seq_len_with_past = seq_len_with_past + past_key_values_len
        # 3. 生成位置编码ID
        if position_ids is None:
            # 自动生成位置ID：从past_key_values_len开始，长度为seq_len
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_len, seq_len + past_key_values_len, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len)  # 扩展为 [batch_size, seq_len]
        else:
            position_ids = position_ids.view(-1, seq_len).long()

        # 4. 获取输入嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)  # 将token ID转换为嵌入向量
        
        # 5. 准备4D因果注意力掩码
        # 从2D掩码 [batch_size, seq_len] 生成4D掩码 [batch_size, 1, seq_len, seq_len_with_past]
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_len), inputs_embeds, past_key_values_len
        )

        # 6. 初始化隐藏状态
        hidden_states = inputs_embeds

        # 7. 逐层前向传播
        # decode layers
        all_hidden_states = () if output_hidden_states else None  # 存储所有层的隐藏状态
        all_self_attns = () if output_attentions else None        # 存储所有层的注意力权重
        next_decoder_cache = () if use_cache else None           # 存储所有层的key/value缓存

        # 遍历所有解码器层
        for idx, decoder_layer in enumerate(self.layers):
            # 如果需要输出隐藏状态，保存当前层输入
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            # 获取当前层的历史缓存（如果存在）
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            # 执行当前解码器层的前向传播
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果使用缓存，保存当前层的key/value
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            # 如果需要输出注意力权重，保存当前层的注意力
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # 8. 最终处理
        # 应用最终的归一化层
        hidden_states = self.norm(hidden_states)

        # 如果需要输出隐藏状态，添加最后一层的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 准备缓存数据
        next_cache = next_decoder_cache if use_cache else None

        # 9. 返回结果
        # 创建简单的输出对象而不是使用BaseModelOutputWithPast
        class ModelOutput:
            def __init__(self, last_hidden_state, past_key_values=None, hidden_states=None, attentions=None):
                self.last_hidden_state = last_hidden_state    # 最后一层的隐藏状态
                self.past_key_values = past_key_values        # 所有层的key/value缓存
                self.hidden_states = hidden_states            # 所有层的隐藏状态
                self.attentions = attentions                  # 所有层的注意力权重

        return ModelOutput(last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            )

# --------------------- LLaMA For Causal LM --------------------- #

class LlamaForCausalLM(nn.Module):
    """
    带有语言建模头的LLaMA模型
    这是完整的生成式语言模型，在LlamaModel基础上添加了输出投影层
    用于将隐藏状态映射到词汇表概率分布，支持文本生成任务
    """
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)          # 核心的LLaMA模型
        self.vocab_size = config.vocab_size      # 词汇表大小
        # 语言建模头：将隐藏状态映射到词汇表概率
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        """获取输入嵌入层"""
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """设置输入嵌入层"""
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """获取输出嵌入层（语言建模头）"""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """设置输出嵌入层"""
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """设置解码器模型"""
        self.model = decoder

    def get_decoder(self):
        """获取解码器模型"""
        return self.model

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """从预训练模型路径加载模型"""
        model_path = Path(model_path)
        
        # 1. 加载配置文件
        config_path = model_path / "config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # 2. 创建配置对象
        config = LlamaConfig.from_dict(config_dict)
        
        # 3. 创建模型实例
        model = cls(config)
        
        # 4. 加载模型权重
        # 优先尝试safetensors格式（更安全）
        weights_path = model_path / "model.safetensors"
        if weights_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            # 备选方案：尝试pytorch格式
            weights_path = model_path / "pytorch_model.bin"
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location="cpu")
            else:
                raise FileNotFoundError(f"No model weights found in {model_path}")
        
        # 5. 加载权重到模型
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(missing_keys)
        # 6. 处理权重共享情况
        # 如果lm_head.weight缺失且配置要求权重共享，则共享embed_tokens的权重
        if 'lm_head.weight' in missing_keys and config.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight
            
        # 7. 设置数据类型
        if hasattr(config, 'torch_dtype') and config.torch_dtype:
            if config.torch_dtype == "float16":
                model = model.half()
            elif config.torch_dtype == "bfloat16":
                model = model.to(torch.bfloat16)
        
        return model
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,          # 输入的token ID张量, 形状为 [batch_size, seq_len]
        attention_mask: Optional[torch.Tensor] = None,     # 注意力掩码张量, 形状为 [batch_size, 1, seq_len, seq_len]
        position_ids: Optional[torch.Tensor] = None,       # 位置ID张量, 形状为 [batch_size, seq_len]
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, # 过去的key和value张量, 用于自回归生成
        inputs_embeds: Optional[torch.Tensor] = None,      # 预计算的输入嵌入张量, 形状为 [batch_size, seq_len, hidden_size]
        labels: Optional[torch.Tensor] = None,             # 目标标签张量, 用于计算语言模型损失
        use_cache: Optional[bool] = None,                  # 是否返回新的past_key_values
        output_attentions: Optional[bool] = False,         # 是否返回注意力权重
        output_hidden_states: Optional[bool] = False,      # 是否返回所有层的隐藏状态
        return_dict: Optional[bool] = True,                # 是否以字典形式返回输出
    ):
        """
        LlamaForCausalLM的前向传播函数
        
        参数说明:
        - input_ids: 输入的token ID序列 [batch_size, seq_len]
        - attention_mask: 注意力掩码，用于忽略填充位置 [batch_size, seq_len]
        - position_ids: 位置编码ID [batch_size, seq_len]
        - past_key_values: 缓存的历史key/value，用于生成阶段的加速
        - inputs_embeds: 预计算的嵌入向量（可选，与input_ids二选一）
        - labels: 目标标签张量，用于计算语言模型损失 [batch_size, seq_len]
        - use_cache: 是否缓存key/value用于下次生成
        - output_attentions: 是否输出注意力权重
        - output_hidden_states: 是否输出所有层的隐藏状态
        - return_dict: 是否以字典形式返回结果
        """
        # 设置默认输出选项
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions if hasattr(self.config, 'output_attentions') else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states if hasattr(self.config, 'output_hidden_states') else False
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict if hasattr(self.config, 'use_return_dict') else True

        # 1. 调用核心模型进行前向传播
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 2. 获取最后一层的隐藏状态并计算logits
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)    # 映射到词汇表大小 [batch_size, seq_len, vocab_size]
        logits = logits.float()                 # 确保使用float32进行数值稳定性

        # 3. 计算损失（如果提供了标签）
        loss = None
        if labels is not None:
            # 实现标准的语言建模损失：预测下一个token
            # 将序列向左偏移一位：tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()    # 去掉最后一个位置的预测
            shift_labels = labels[..., 1:].contiguous()        # 去掉第一个位置的标签
            
            # 展平张量以便计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)  # [batch_size * seq_len, vocab_size]
            shift_labels = shift_labels.view(-1)                          # [batch_size * seq_len]
            
            # 确保标签在正确的设备上（模型并行支持）
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # 4. 返回结果
        # 创建简单的输出对象而不是使用CausalLMOutputWithPast
        class CausalLMOutput:
            def __init__(self, loss=None, logits=None, past_key_values=None, hidden_states=None, attentions=None):
                self.loss = loss                    # 语言建模损失（如果提供了标签）
                self.logits = logits                # 词汇表概率分布
                self.past_key_values = past_key_values  # key/value缓存
                self.hidden_states = hidden_states      # 所有层的隐藏状态
                self.attentions = attentions            # 所有层的注意力权重

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    """
    创建4D因果注意力掩码
    
    功能: 从2D掩码 `(batch_size, key_length)` 创建4D因果掩码 `(batch_size, 1, query_length, key_length)`
    
    参数:
    - attention_mask: 2D注意力掩码 [batch_size, seq_len] 或 None
    - input_shape: 输入形状元组 (batch_size, seq_len)
    - inputs_embeds: 输入嵌入张量，用于获取设备和数据类型信息
    - past_key_values_length: 历史缓存的序列长度
    
    返回:
    - 4D因果注意力掩码 [batch_size, 1, seq_len, seq_len_with_past]
    
    具体例子:
    假设我们有一个序列 ["Hello", "world", "!"]，seq_len=3，past_key_values_length=0
    
    1. 初始因果掩码（下三角矩阵，True表示可以关注）:
       [[True,  False, False],   # "Hello" 只能看到自己
        [True,  True,  False],   # "world" 可以看到 "Hello" 和自己  
        [True,  True,  True ]]   # "!" 可以看到前面所有token
    
    2. 如果attention_mask=[1, 1, 0]（最后一个位置是padding）:
       [[True,  False, False],   # "Hello" 只能看到自己
        [True,  True,  False],   # "world" 可以看到 "Hello" 和自己
        [True,  True,  False]]   # "!" 不能看到padding位置（自己）
    
    3. 转换为注意力权重掩码（-inf表示被掩蔽，0.0表示可关注）:
       [[  0.0, -inf, -inf],
        [  0.0,  0.0, -inf], 
        [  0.0,  0.0, -inf]]
    
    4. 在生成阶段，如果past_key_values_length=2（已有2个历史token）:
       当前输入是新的1个token，它可以关注所有历史token：
       [[0.0, 0.0, 0.0]]  # 新token可以关注2个历史token + 自己
    """
    batch_size, seq_length = input_shape
    dtype = inputs_embeds.dtype
    device = inputs_embeds.device

    # 1. 处理None的情况，创建默认的全1掩码
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=device)

    # 2. 创建因果掩码（下三角矩阵）
    # 计算包含历史缓存的总序列长度
    seq_length_with_past = seq_length + past_key_values_length
    
    # 创建下三角因果掩码：当前位置只能看到之前（包括自己）的位置
    # torch.tril创建下三角矩阵，上三角部分为False，下三角及对角线为True
    causal_mask = torch.tril(torch.ones((seq_length, seq_length_with_past), dtype=torch.bool, device=device))
    # 扩展维度: [seq_length, seq_length_with_past] -> [batch_size, 1, seq_length, seq_length_with_past]
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, seq_length, seq_length_with_past)
    
    # 3. 应用用户提供的注意力掩码
    if attention_mask.dim() == 2:
        # 处理历史缓存的情况
        if past_key_values_length > 0:
            # 对于生成阶段，需要处理历史key values
            # 创建包含历史部分的完整掩码（历史部分设为全1，当前部分使用用户掩码）
            expanded_attn_mask = torch.ones((batch_size, seq_length_with_past), dtype=attention_mask.dtype, device=device)
            expanded_attn_mask[:, past_key_values_length:] = attention_mask
        else:
            # 训练阶段，直接使用用户掩码
            expanded_attn_mask = attention_mask
            
        # 扩展到4D: [batch_size, seq_length_with_past] -> [batch_size, 1, seq_length, seq_length_with_past]
        expanded_attn_mask = expanded_attn_mask[:, None, None, :].expand(batch_size, 1, seq_length, seq_length_with_past)
        # 将因果掩码与用户掩码进行逻辑AND操作
        causal_mask = causal_mask & expanded_attn_mask.bool()

    # 4. 转换为适合注意力计算的格式
    # 将True/False转换为0.0/-inf，True表示可以关注，False表示需要掩蔽
    inverted_mask = 1.0 - causal_mask.float()  # True->0.0, False->1.0
    # 将需要掩蔽的位置（值为1.0）设置为负无穷，这样softmax后会变成0
    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)