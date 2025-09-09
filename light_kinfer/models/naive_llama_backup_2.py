import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
from typing import Optional, Tuple, List
from pathlib import Path

from .model_config import LlamaConfig
nn.RMSNorm

# class LlamaRMSNorm(nn.Module):
#     """Root Mean Square Layer Normalization"""
    
#     def __init__(self, hidden_size: int, eps: float = 1e-6):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, hidden_states):
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(torch.float32)
#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         return self.weight * hidden_states.to(input_dtype)

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

# class LlamaRotaryEmbedding(nn.Module):
#     """Rotary Position Embedding"""
    
#     def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
#         super().__init__()
#         self.dim = dim
#         self.max_position_embeddings = max_position_embeddings
#         self.base = base
        
#         inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
#         self.register_buffer("inv_freq", inv_freq, persistent=False)

#     def forward(self, x, seq_len=None):
#         # x: [batch_size, seq_len, num_heads, head_dim]
#         if seq_len is None:
#             seq_len = x.shape[1]
            
#         t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
#         freqs = torch.outer(t, self.inv_freq)
#         emb = torch.cat((freqs, freqs), dim=-1)
#         cos = emb.cos()
#         sin = emb.sin()
#         # Return with proper shape: [1, seq_len, 1, head_dim]
#         return cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2)


# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
#     """Apply rotary position embedding to query and key tensors."""
#     # q: [batch_size, num_heads, seq_len, head_dim] - shape like [1, 32, 9, 64]
#     # k: [batch_size, num_kv_heads, seq_len, head_dim] - shape like [1, 8, 9, 64]  
#     # cos, sin: [1, seq_len, 1, head_dim] - shape like [1, 9, 1, 64]
    
#     if position_ids is None:
#         # Transpose cos and sin to match q and k dimensions
#         # From [1, seq_len, 1, head_dim] to [1, 1, seq_len, head_dim]
#         cos = cos.transpose(1, 2)  # [1, 1, seq_len, head_dim]
#         sin = sin.transpose(1, 2)  # [1, 1, seq_len, head_dim]
#     else:
#         # Use specific positions
#         cos = cos.squeeze(0).squeeze(1)  # [seq_len, head_dim]
#         sin = sin.squeeze(0).squeeze(1)  # [seq_len, head_dim]
#         # example: position_ids.shape = batch_size, seq_len
#         cos = cos[position_ids]  # [batch_size, seq_len, head_dim]
#         sin = sin[position_ids]  # [batch_size, seq_len, head_dim]
#         cos = cos.unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]
#         sin = sin.unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]

#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed
# ----

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

    def forward(self, x, seq_len=None):
        # x: [batch_size, seq_len, num_heads, head_dim]
        if seq_len is None:
            seq_len = x.shape[1]

        t = torch.arange(seq_len, device=x.device, dtype=torch.int64) # 功能: 生成位置索引, [seq_len]
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

# class LlamaAttention(nn.Module):
#     """Multi-head attention module"""
    
#     def __init__(self, config: LlamaConfig):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_heads
#         self.head_dim = config.head_dim
#         self.num_key_value_heads = config.num_kv_heads
#         self.num_key_value_groups = self.num_heads // self.num_key_value_heads
#         self.max_position_embeddings = config.max_position_embeddings
#         self.rope_theta = config.rope_theta
        
#         self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
#         self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
#         self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
#         self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
#         self.rotary_emb = LlamaRotaryEmbedding(
#             self.head_dim,
#             max_position_embeddings=self.max_position_embeddings,
#             base=self.rope_theta,
#         )

#     def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
#         return tensor.view(bsz, seq_len, -1, self.head_dim).transpose(1, 2).contiguous()

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Tuple[torch.Tensor]] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
#         bsz, q_len, _ = hidden_states.size()

#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)

#         query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#         key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#         value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

#         kv_seq_len = key_states.shape[-2]
#         if past_key_value is not None:
#             kv_seq_len += past_key_value[0].shape[-2]
            
#         cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

#         if past_key_value is not None:
#             # reuse k, v, self_attention
#             key_states = torch.cat([past_key_value[0], key_states], dim=2)
#             value_states = torch.cat([past_key_value[1], value_states], dim=2)

#         past_key_value = (key_states, value_states) if use_cache else None

#         # repeat k/v heads if n_kv_heads < n_heads
#         key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
#         value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

#         attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

#         if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
#             raise ValueError(
#                 f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
#                 f" {attn_weights.size()}"
#             )

#         if attention_mask is not None:
#             if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
#                 raise ValueError(
#                     f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
#                 )
#             attn_weights = attn_weights + attention_mask

#         # upcast attention to fp32
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#         attn_output = torch.matmul(attn_weights, value_states)

#         if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )

#         attn_output = attn_output.transpose(1, 2).contiguous()
#         attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

#         attn_output = self.o_proj(attn_output)

#         if not output_attentions:
#             attn_weights = None

#         return attn_output, attn_weights, past_key_value

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

# class LlamaMLP(nn.Module):
#     """MLP module"""
    
#     def __init__(self, config: LlamaConfig):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size

#         self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
#         self.act_fn = nn.SiLU()

#     def forward(self, x):
#         down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
#         return down_proj

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

# class LlamaDecoderLayer(nn.Module):
#     """Single decoder layer"""
    
#     def __init__(self, config: LlamaConfig):
#         super().__init__()
#         self.hidden_size = config.hidden_size
#         self.self_attn = LlamaAttention(config=config)
#         self.mlp = LlamaMLP(config)
#         self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Tuple[torch.Tensor]] = None,
#         output_attentions: Optional[bool] = False,
#         use_cache: Optional[bool] = False,
#     ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
#         residual = hidden_states

#         hidden_states = self.input_layernorm(hidden_states)

#         # Self Attention
#         hidden_states, self_attn_weights, present_key_value = self.self_attn(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_value=past_key_value,
#             output_attentions=output_attentions,
#             use_cache=use_cache,
#         )
#         hidden_states = residual + hidden_states

#         # Fully Connected
#         residual = hidden_states
#         hidden_states = self.post_attention_layernorm(hidden_states)
#         hidden_states = self.mlp(hidden_states)
#         hidden_states = residual + hidden_states

#         outputs = (hidden_states,)

#         if output_attentions:
#             outputs += (self_attn_weights,)

#         if use_cache:
#             outputs += (present_key_value,)

#         return outputs

# --------------------- LLaMA Model --------------------- #

# class LlamaModel(nn.Module):
#     """
#     Transformer consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
#     """

#     def __init__(self, config: LlamaConfig):
#         super().__init__()
#         self.config = config
#         self.padding_idx = config.bos_token_id  # Use bos_token_id as padding
#         self.vocab_size = config.vocab_size

#         self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
#         self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_layers)])
#         self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#     def get_input_embeddings(self):
#         return self.embed_tokens

#     def set_input_embeddings(self, value):
#         self.embed_tokens = value

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ):
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions if hasattr(self.config, 'output_attentions') else False
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states if hasattr(self.config, 'output_hidden_states') else False
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache

#         # retrieve input_ids and inputs_embeds
#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         elif input_ids is not None:
#             batch_size, seq_length = input_ids.shape
#         elif inputs_embeds is not None:
#             batch_size, seq_length, _ = inputs_embeds.shape
#         else:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")

#         seq_length_with_past = seq_length
#         past_key_values_length = 0

#         if past_key_values is not None:
#             past_key_values_length = past_key_values[0][0].shape[2]
#             seq_length_with_past = seq_length_with_past + past_key_values_length

#         if position_ids is None:
#             device = input_ids.device if input_ids is not None else inputs_embeds.device
#             position_ids = torch.arange(
#                 past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
#             )
#             position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
#         else:
#             position_ids = position_ids.view(-1, seq_length).long()

#         if inputs_embeds is None:
#             inputs_embeds = self.embed_tokens(input_ids)

#         # 4d mask is passed through the layers
#         attention_mask = _prepare_4d_causal_attention_mask(
#             attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
#         )

#         hidden_states = inputs_embeds

#         # decoder layers
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         next_decoder_cache = () if use_cache else None

#         for idx, decoder_layer in enumerate(self.layers):
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)

#             past_key_value = past_key_values[idx] if past_key_values is not None else None

#             layer_outputs = decoder_layer(
#                 hidden_states,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 past_key_value=past_key_value,
#                 output_attentions=output_attentions,
#                 use_cache=use_cache,
#             )

#             hidden_states = layer_outputs[0]

#             if use_cache:
#                 next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)

#         hidden_states = self.norm(hidden_states)

#         # add hidden states from the last decoder layer
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)

#         next_cache = next_decoder_cache if use_cache else None

#         # Return a simple object instead of BaseModelOutputWithPast
#         class ModelOutput:
#             def __init__(self, last_hidden_state, past_key_values=None, hidden_states=None, attentions=None):
#                 self.last_hidden_state = last_hidden_state
#                 self.past_key_values = past_key_values
#                 self.hidden_states = hidden_states
#                 self.attentions = attentions

#         return ModelOutput(
#             last_hidden_state=hidden_states,
#             past_key_values=next_cache,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#         )

class LlamaModel(nn.Module):
    """
    Transformer consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    """
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.bos_token_id   # 填充标记索引, 通常与bos_token_id相同
        self.vocab_size = config.vocab_size      # 词汇表大小

        # 词嵌入层, 功能: 将输入的token ID转换为对应的词向量
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = LlamaRMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
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
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, 1, seq_len, seq_len] or None
        position_ids: [batch_size, seq_len] or None
        past_key_values: Tuple of (key, value) each of shape [batch_size, num_kv_heads, past_seq_len, head_dim]
        inputs_embeds: [batch_size, seq_len, hidden_size]
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions if hasattr(self.config, "output_attentions") else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states if hasattr(self.config, "output_hidden_states") else False
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache if hasattr(self.config, "use_cache") else False

        # 1. 输入验证与形状推断
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_len = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_len, _ = inputs_embeds.shape 
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_len_with_past = seq_len
        past_key_values_len = 0

        # 计算past_key_values的长度
        if past_key_values is not None:
            past_key_values_len = past_key_values[0][0].shape[2]
            seq_len_with_past = seq_len_with_past + past_key_values_len
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_len, seq_len + past_key_values_len, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
        else:
            position_ids = position_ids.view(-1, seq_len).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_len), inputs_embeds, past_key_values_len
        )

        hidden_states = inputs_embeds

        # decode layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        # Return a simple object instead of BaseModelOutputWithPast
        class ModelOutput:
            def __init__(self, last_hidden_state, past_key_values=None, hidden_states=None, attentions=None):
                self.last_hidden_state = last_hidden_state
                self.past_key_values = past_key_values
                self.hidden_states = hidden_states
                self.attentions = attentions

        return ModelOutput(last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            )

# --------------------- LLaMA For Causal LM --------------------- #

class LlamaForCausalLM(nn.Module):
    """Llama Model with a language modeling head"""
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load pretrained model from path"""
        model_path = Path(model_path)
        
        # Load config
        config_path = model_path / "config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create config object
        config = LlamaConfig.from_dict(config_dict)
        
        # Create model
        model = cls(config)
        
        # Load weights
        weights_path = model_path / "model.safetensors"
        if weights_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            # Try pytorch format
            weights_path = model_path / "pytorch_model.bin"
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location="cpu")
            else:
                raise FileNotFoundError(f"No model weights found in {model_path}")
        
        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        # Handle tied embeddings - if lm_head.weight is missing and tie_word_embeddings is True
        if 'lm_head.weight' in missing_keys and config.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight
        # Set dtype if specified
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
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, 1, seq_len, seq_len] or None
        position_ids: [batch_size, seq_len] or None
        past_key_values: Tuple of (key, value) each of shape [batch_size, num_kv_heads, past_seq_len, head_dim]
        inputs_embeds: [batch_size, seq_len, hidden_size]
        labels: [batch_size, seq_len]
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions if hasattr(self.config, 'output_attentions') else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states if hasattr(self.config, 'output_hidden_states') else False
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict if hasattr(self.config, 'use_return_dict') else True

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

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # Return a simple object instead of CausalLMOutputWithPast
        class CausalLMOutput:
            def __init__(self, loss=None, logits=None, past_key_values=None, hidden_states=None, attentions=None):
                self.loss = loss
                self.logits = logits
                self.past_key_values = past_key_values
                self.hidden_states = hidden_states
                self.attentions = attentions

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_length)` from a 2D mask of shape
    `(batch_size, key_length)`
    """
    batch_size, seq_length = input_shape
    dtype = inputs_embeds.dtype
    device = inputs_embeds.device

    # Handle the case where attention_mask is None
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=device)

    # Create causal mask
    # [batch_size, seq_length] -> [batch_size, 1, seq_length, seq_length]
    seq_length_with_past = seq_length + past_key_values_length
    
    causal_mask = torch.tril(torch.ones((seq_length, seq_length_with_past), dtype=torch.bool, device=device))
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, seq_length, seq_length_with_past)
    
    # Apply attention mask
    if attention_mask.dim() == 2:
        # Expand attention_mask to 4D
        if past_key_values_length > 0:
            # For generation, we need to handle past key values
            expanded_attn_mask = torch.ones((batch_size, seq_length_with_past), dtype=attention_mask.dtype, device=device)
            expanded_attn_mask[:, past_key_values_length:] = attention_mask
        else:
            expanded_attn_mask = attention_mask
            
        expanded_attn_mask = expanded_attn_mask[:, None, None, :].expand(batch_size, 1, seq_length, seq_length_with_past)
        causal_mask = causal_mask & expanded_attn_mask.bool()

    # Convert to float and apply masking
    inverted_mask = 1.0 - causal_mask.float()
    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)
