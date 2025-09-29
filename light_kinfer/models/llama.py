import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Union, List
from light_kinfer.executor.executor_struct import *
from light_kinfer.kernels import *
from light_kinfer.models.model_config import LlamaConfig
from light_kinfer.models.RotaryEmbedding import LlamaRotaryEmbedding

class FusedAttention(nn.Module):
    def __init__(self, config: LlamaConfig, cache_k=None, cache_v=None):
        super().__init__()
        self.config = config

        # kv头数可能与q头数不同，如果config没有提及，则默认为相同
        self.num_kv_heads = (
            config.num_heads if config.num_kv_heads is None else config.num_kv_heads
        )
        self.head_dim = config.head_dim if config.head_dim is not None else config.hidden_size // config.num_heads
        self.num_q_heads = config.num_heads
        self.hidden_size = config.hidden_size if config.hidden_size is not None else config.num_heads * self.head_dim

        self.q_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False, dtype=torch.float16
        )
        self.kv_proj_weight = nn.Parameter(
            torch.rand(
                self.num_kv_heads * self.head_dim * 2, self.hidden_size, dtype=torch.float16
            )
        )
        self.o_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False, dtype=torch.float16
        )

    def context_forward(
        self,
        x: torch.Tensor,
        atten_info: AttentionInfo,
        layer_index: int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        qk_scale=None,
    ):
        """
        处理上下文的前向传播函数，主要用于prefill阶段（处理输入序列长度>1的情况）
        
        参数:
            x: 输入张量，形状为 [batch_size, seq_len, hidden_size]
            atten_info: 注意力相关的信息，包含KV缓存和序列信息
            layer_index: 当前层的索引
            position_embeddings: 位置编码，包含cos和sin张量
            qk_scale: 缩放因子，用于缩放注意力分数
        
        返回:
            output: 注意力层的输出，形状为 [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape # 输入形状: prefill阶段为 (B, Seq_Len, Dim); decode阶段为 (B, 1, Dim)
        # 将输入重塑为二维张量，便于进行线性变换
        # 形状变化: [batch_size, seq_len, hidden_size] -> [batch_size*seq_len, hidden_size]
        x = x.view(-1, self.hidden_size)

        # 1. 计算 Q K V 投影(即与对应的权重相乘)并且 reshape 它们的尺寸, 方便后续做 self-attention
        # Query投影: [batch_size*seq_len, hidden_size] -> [batch_size*seq_len, num_q_heads*head_dim]
        xq = self.q_proj(x)

        # 分离K和V的投影权重，每个权重形状为 [num_kv_heads*head_dim, hidden_size]
        k_proj_weight, v_proj_weight = torch.split(
            self.kv_proj_weight, self.num_kv_heads * self.head_dim, dim=0
        )

        # Key投影: [batch_size*seq_len, hidden_size] -> [batch_size*seq_len, num_kv_heads*head_dim]
        xk = F.linear(x, k_proj_weight)
        # Value投影: [batch_size*seq_len, hidden_size] -> [batch_size*seq_len, num_kv_heads*head_dim]
        xv = F.linear(x, v_proj_weight)

        # 2. 应用旋转位置编码到 Q 和 K, 将 xk, xv 合并, 并写入KV缓存

        # 重塑为多头注意力的形状: [batch_size*seq_len, num_heads, head_dim]
        xq = xq.view(-1, self.num_q_heads, self.head_dim)
        xk = xk.view(-1, self.num_kv_heads, self.head_dim)
        xv = xv.view(-1, self.num_kv_heads, self.head_dim)

        # 获取位置编码的cos和sin值
        cos, sin = position_embeddings
        # 应用旋转位置编码（RoPE），输出形状保持不变
        xq, xk = rope_emb_forward(xq, xk, cos, sin, batch_size, seq_len)
        
        # 将K和V沿着头维度拼接，形状: [batch_size*seq_len, 2*num_kv_heads, head_dim]
        combined_kv = torch.cat([xk, xv], dim=-2)

        # 更新KV缓存，将当前层的K、V值存储到缓存中
        update_kv_buffer(
            combined_kv, atten_info.cur_select_index, atten_info.kv_buffer[layer_index]
        )

        # 3. self-attention计算: flashattention优化版本，计算 softmax(QK^T/√d) * V
        # 使用flash attention进行高效的注意力计算，无需显式计算注意力矩阵
        # 输出形状: [batch_size*seq_len, num_q_heads, head_dim]
        output = flash_attention2_no_pad(
            xq,
            xk,
            xv,
            qk_scale,
            atten_info.b_start_loc,    # 批次中每个序列的起始位置
            atten_info.b_seq_len,     # 批次中每个序列的长度
            seq_len,                  # 最大序列长度
        )

        # 将输出重塑回原始的三维形状: [batch_size, seq_len, hidden_size]
        output = output.view(batch_size, seq_len, self.hidden_size)

        # 4. 通过输出投影层进行最终的线性变换
        # 形状: [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        output = self.o_proj(output)
        return output

    def token_forward(
        self,
        x: torch.Tensor,
        atten_info: AttentionInfo,
        layer_index: int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        qk_scale=None,
    ):
        """
        处理单个token的前向传播函数，主要用于decode阶段（生成阶段，seq_len=1）
        
        参数:
            x: 输入张量，形状为 [batch_size, 1, hidden_size] (decode阶段seq_len=1)
            atten_info: 注意力相关的信息，包含KV缓存和序列信息
            layer_index: 当前层的索引
            position_embeddings: 位置编码，包含cos和sin张量
            qk_scale: 缩放因子，用于缩放注意力分数
        
        返回:
            output: 注意力层的输出，形状为 [batch_size, 1, hidden_size]
        """
        batch_size, seq_len, _ = (
            x.shape
        )  # decode阶段: (B, 1, Dim)，seq_len固定为1

        # 将输入重塑为二维张量，便于进行线性变换
        # 形状变化: [batch_size, 1, hidden_size] -> [batch_size, hidden_size]
        x = x.view(-1, self.hidden_size)

        # 1. 计算 Q K V 投影并且 reshape 它们的尺寸, 方便后续做 self-attention
        # Query投影: [batch_size, hidden_size] -> [batch_size, num_q_heads*head_dim]
        xq = self.q_proj(x)
        
        # 一次性计算K和V投影，提高效率
        # KV投影: [batch_size, hidden_size] -> [batch_size, 2*num_kv_heads*head_dim]
        xkv = F.linear(
            x, self.kv_proj_weight.data
        )

        # 2. 应用旋转位置编码到 Q 和 K, 获取 kv 缓冲向量并更新 kv 向量
        # 分离K和V投影结果，每个形状为 [batch_size, num_kv_heads*head_dim]
        xk, xv = torch.split(xkv, self.num_kv_heads * self.head_dim, dim=-1)
        
        # 重塑为多头注意力的形状: [batch_size, num_heads, head_dim]
        xq = xq.view(batch_size, self.num_q_heads, self.head_dim)
        xk = xk.view(batch_size, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, self.num_kv_heads, self.head_dim)

        # 获取位置编码的cos和sin值
        cos, sin = position_embeddings
        # 应用旋转位置编码（RoPE），输出形状保持不变
        xq, xk = rope_emb_forward(xq, xk, cos, sin, batch_size, seq_len)

        # 3. 完成形状变换, 并更新 kv_buffer, 即类似 torch.concat[past_kv_values, kv_values]
        # 将当前token的K和V沿着头维度拼接，形状: [batch_size, 2*num_kv_heads, head_dim]
        combined_kv = torch.cat([xk, xv], dim=-2)
        
        # 更新KV缓存，将当前token的K、V值追加到历史缓存中
        # 这相当于将新的KV与过去的KV进行拼接: [past_k, past_v] + [current_k, current_v]
        update_kv_buffer(
            combined_kv, atten_info.cur_select_index, atten_info.kv_buffer[layer_index]
        )

        # 4. flashdecoding计算: 高效的decode阶段注意力计算 softmax(QK^T/√d) * V
        # 使用优化的flash decoding，利用已缓存的历史K、V与当前Q进行注意力计算
        # 从KV缓存中分别提取K和V：
        # - K缓存: kv_buffer[:, :num_kv_heads, :] 形状为 [max_seq_len, num_kv_heads, head_dim]
        # - V缓存: kv_buffer[:, num_kv_heads:, :] 形状为 [max_seq_len, num_kv_heads, head_dim]
        output = flash_decoding(
            xq,  # 当前Query: [batch_size, num_q_heads, head_dim]
            atten_info.kv_buffer[layer_index][:, : self.num_kv_heads, :],     # 历史+当前的K
            atten_info.kv_buffer[layer_index][:, self.num_kv_heads :, :],     # 历史+当前的V
            qk_scale,                           # 注意力缩放因子
            atten_info.b_req_tokens_table,      # 批次中每个请求的token索引表
            atten_info.b_seq_len,               # 批次中每个序列的实际长度
            atten_info.max_actual_seq_len,      # 当前批次的最大序列长度
        )  # 输出形状: [batch_size, num_q_heads, head_dim]

        # 将输出重塑回原始的三维形状: [batch_size, 1, hidden_size]
        output = output.view(batch_size, seq_len, self.hidden_size)
        
        # 通过输出投影层进行最终的线性变换
        # 形状: [batch_size, 1, hidden_size] -> [batch_size, 1, hidden_size]
        output = self.o_proj(output)
        return output

class FusedMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=torch.float16)

        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=torch.float16)

        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, dtype=torch.float16)

    def forward(self, x):
        return self.down_proj(swiglu_forward(self.gate_proj(x), self.up_proj(x)))

class LlamaDecoderLayer(nn.Module):
    """
    Llama解码器层 - Transformer解码器的单个层实现
    
    功能:
    - 实现标准的Transformer解码器层结构: LayerNorm -> Self-Attention -> LayerNorm -> MLP
    - 使用残差连接和RMSNorm归一化
    - 根据序列长度自动选择prefill或decode模式的注意力计算
    - 支持跳过连接优化，减少内存复制
    
    结构:
    1. Self-Attention模块: 多头自注意力机制
    2. MLP模块: 前馈神经网络 (Gate + Up -> SwiGLU -> Down)
    3. 两个RMSNorm层: 分别用于注意力和MLP的归一化
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim if config.head_dim is not None else self.hidden_size // self.num_heads
        self.rmsnorm_eps = config.rms_norm_eps

        # 注意力层的RMSNorm权重参数，形状: [hidden_size]
        self.attention_norm_weight = nn.Parameter(
            torch.ones(
                self.hidden_size,
            ),
            requires_grad=False,
        )

        # MLP层的RMSNorm权重参数，形状: [hidden_size]
        self.ffn_norm_weight = nn.Parameter(
            torch.ones(
                self.hidden_size,
            ),
            requires_grad=False,
        )

        # 自注意力模块: 包含Q、K、V投影和输出投影
        self.self_attn = FusedAttention(config)
        # MLP模块: 包含gate、up、down三个投影层
        self.mlp = FusedMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        atten_info,
        layer_index: int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        qk_scale=None,
        residual: Optional[torch.Tensor] = None,
    ):
        """
        解码器层的前向传播
        
        参数:
            hidden_states: 输入的隐藏状态，形状 [batch_size, seq_len, hidden_size]
            atten_info: 注意力相关信息，包含KV缓存等
            layer_index: 当前层的索引
            position_embeddings: 位置编码 (cos, sin)
            qk_scale: 注意力缩放因子
            residual: 残差连接的累积值，用于跳过连接优化
            
        返回:
            hidden_states: 处理后的隐藏状态，形状 [batch_size, seq_len, hidden_size]
            residual: 更新后的残差连接值
        """
        # 获取序列长度，用于判断是prefill还是decode阶段
        _, seq_len, _ = hidden_states.shape

        # === 第一部分: Self-Attention ===
        # 1. 注意力前的RMSNorm归一化 + 残差连接
        # skip_rmsnorm同时进行归一化和残差累积，避免额外的内存复制
        # 输入: hidden_states [batch_size, seq_len, hidden_size]
        # 输出: 归一化后的hidden_states, 更新的residual
        hidden_states, residual = skip_rmsnorm(
            hidden_states, residual, self.attention_norm_weight.data, self.rmsnorm_eps
        )

        # 2. 根据序列长度选择不同的注意力计算方式
        if seq_len > 1:
            # Prefill阶段: 处理整个输入序列，使用context_forward
            # 输入: [batch_size, seq_len, hidden_size]
            # 输出: [batch_size, seq_len, hidden_size]
            hidden_states = self.self_attn.context_forward(
                hidden_states, atten_info, layer_index, position_embeddings, qk_scale
            )
        else:
            # Decode阶段: 处理单个token，使用token_forward
            # 输入: [batch_size, 1, hidden_size]  
            # 输出: [batch_size, 1, hidden_size]
            hidden_states = self.self_attn.token_forward(
                hidden_states, atten_info, layer_index, position_embeddings, qk_scale
            )

        # === 第二部分: MLP (Feed-Forward Network) ===
        # 3. MLP前的RMSNorm归一化 + 残差连接
        # 将attention的输出与之前的residual进行归一化和累积
        # 输入: hidden_states [batch_size, seq_len, hidden_size]
        # 输出: 归一化后的hidden_states, 更新的residual
        hidden_states, residual = skip_rmsnorm(
            hidden_states, residual, self.ffn_norm_weight.data, self.rmsnorm_eps
        )
        
        # 4. MLP前馈网络计算
        # 结构: Gate投影 & Up投影 -> SwiGLU激活 -> Down投影
        # 维度变化: [batch_size, seq_len, hidden_size] 
        #          -> [batch_size, seq_len, intermediate_size] (gate & up)
        #          -> [batch_size, seq_len, intermediate_size] (swiglu)
        #          -> [batch_size, seq_len, hidden_size] (down)
        hidden_states = self.mlp.forward(hidden_states)
        
        # 返回处理后的隐藏状态和累积的残差
        return hidden_states, residual

class LlamaModel(nn.Module):
    """
    Llama模型的主体结构 - 完整的Transformer解码器实现
    
    功能:
    - 实现完整的Llama语言模型架构
    - 包含词嵌入、位置编码、多层解码器、输出投影
    - 支持文本生成和语言建模任务
    - 优化了prefill和decode两个阶段的推理性能
    
    架构组成:
    1. 词嵌入层: 将token ID转换为向量表示
    2. 旋转位置编码: RoPE位置编码
    3. 多层解码器: 堆叠的LlamaDecoderLayer
    4. 最终归一化层: 输出前的RMSNorm
    5. 语言模型头: 将隐藏状态映射到词汇表概率
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size        # 词汇表大小
        self.num_layers = config.num_layers        # 解码器层数
        self.head_dim = (
            config.head_dim
            if config.head_dim is not None
            else config.hidden_size // config.num_heads
        )
        # 注意力缩放因子: 1/√d_k，用于缓解梯度消失
        self.qk_scale = 1.0 / (self.head_dim**0.5)
        self.rmsnorm_eps = config.rms_norm_eps     # RMSNorm的数值稳定性参数

        # === 核心组件初始化 ===
        
        # 1. 旋转位置编码模块
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        
        # 2. 词嵌入层: token_id -> 向量表示
        # 形状: [vocab_size, hidden_size]
        self.embed_tokens = nn.Embedding(
            self.vocab_size, config.hidden_size, dtype=torch.float16
        )
        
        # 3. 最终层归一化权重
        # 形状: [hidden_size]，用于输出前的RMSNorm
        self.norm_weight = nn.Parameter(
            torch.ones(
                config.hidden_size,
            ),
            requires_grad=False,
        )

        # 4. 语言模型头: 隐藏状态 -> 词汇表logits
        # 形状: [hidden_size, vocab_size]
        self.lm_head = nn.Linear(
            config.hidden_size, self.vocab_size, bias=False, dtype=torch.float16
        )

        # 5. 解码器层堆叠: 创建num_layers个相同结构的解码器层
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_layers)]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info: AttentionInfo,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        """
        Llama模型的前向传播
        
        参数:
            input_ids: 输入的token ID序列，形状 [batch_size, seq_len]
            position_ids: 位置ID序列，形状 [batch_size, seq_len]  
            atten_info: 注意力相关信息，包含KV缓存、序列长度等
            inputs_embeds: 可选的预计算嵌入，用于多模态模型，形状 [batch_size, seq_len, hidden_size]
            
        返回:
            output: 输出logits，形状 [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        residual = None  # 残差连接的初始值

        # === 1. 输入嵌入处理 ===
        if inputs_embeds is not None:  
            # 多模态模型支持：直接使用预计算的嵌入
            # 形状: [batch_size, seq_len, hidden_size]
            h = inputs_embeds
        else:
            # 标准文本模型：通过词嵌入层获取向量表示
            # 维度变化: [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
            h = self.get_input_embeddings(input_ids) 
        
        # === 2. 动态缩放因子设置 ===
        # 根据序列长度调整注意力缩放因子，优化不同阶段的数值稳定性
        if seq_len > 1:
            # Prefill阶段: 使用更大的缩放因子，ln(2) ≈ 1.442695
            qk_scale = self.qk_scale * 1.4426950408889634
        else:
            # Decode阶段: 使用标准缩放因子
            qk_scale = self.qk_scale
        
        # === 3. 位置编码计算 ===
        # 计算旋转位置编码的cos和sin值
        # 输出: (cos, sin) 每个形状为 [batch_size, seq_len, head_dim]
        position_embeddings = self.rotary_emb(h, position_ids)

        # === 4. 多层解码器处理 ===
        # 依次通过所有解码器层进行特征提取和表示学习
        for i, layer in enumerate(self.layers):
            # 每层的维度变化: [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
            # residual用于跳过连接优化，减少内存开销
            h, residual = layer(
                h,                    # 当前隐藏状态
                atten_info,          # 注意力信息
                i,                   # 层索引
                position_embeddings, # 位置编码
                qk_scale,           # 缩放因子
                residual            # 累积残差
            )
        
        # === 5. 最终处理 ===
        # 最后一次RMSNorm归一化，应用残差连接
        # 维度保持: [batch_size, seq_len, hidden_size]
        h, _ = skip_rmsnorm(h, residual, self.norm_weight.data, self.rmsnorm_eps)
        
        # 语言模型头: 将隐藏状态映射到词汇表上的概率分布
        # 维度变化: [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, vocab_size]
        output = self.lm_head(h)

        return output

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        获取输入token的嵌入表示
        
        参数:
            input_ids: token ID序列，形状 [batch_size, seq_len]
            
        返回:
            嵌入向量，形状 [batch_size, seq_len, hidden_size]
        """
        return self.embed_tokens(input_ids)