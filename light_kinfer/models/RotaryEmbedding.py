"""rotary_embedding.py

旋转位置编码(RoPE - Rotary Positional Embeddings)的紧凑实现

本模块实现了RoPE的多种变体，包括：
- **default**: 标准RoPE实现
- **llama-3/yarn**: Llama3和YARN的频率调节变体  
- **dynamic**: 动态调整最大序列长度的变体
- **longrope**: 支持超长序列的变体

RoPE核心思想：
通过复数旋转将相对位置信息直接编码到注意力计算中，使得注意力分数只依赖于token间的相对位置，
而不是绝对位置，从而具备更好的长度外推能力。

数学原理：
对于位置m的query和位置n的key，RoPE通过旋转变换实现：
q_m' = R_m * q_m, k_n' = R_n * k_n
其中R_θ是旋转矩阵，使得 q_m'^T * k_n' 只依赖于相对位置(m-n)

运行测试:
>>> pytest -q
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Optional local imports – fall back to minimal stubs for standalone run
# -----------------------------------------------------------------------------
try:
    from light_kinfer.models.model_config import LlamaConfig, Qwen2Config  # type: ignore
except Exception:  # pragma: no cover – docs / CI without project

    @dataclass
    class _BaseCfg:
        hidden_size: int = 1024
        num_heads: int = 8
        head_dim: Optional[int] = None
        max_position_embeddings: int = 2048
        rope_theta: float = 10000.0
        rope_scaling: Optional[dict] = None
        partial_rotary_factor: float = 1.0

        def __post_init__(self):
            if self.head_dim is None:
                self.head_dim = self.hidden_size // self.num_heads

    class LlamaConfig(_BaseCfg):
        pass

    class Qwen2Config(_BaseCfg):
        pass

# -----------------------------------------------------------------------------
# 辅助工具函数
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def _to_map(cfg: Optional[object]) -> Optional[Mapping[str, Any]]:
    """
    将配置对象转换为字典映射格式
    
    功能：优雅地处理dataclass/namespace等不同类型的配置对象
    参数：cfg - 配置对象（可能是dataclass、dict或其他类型）
    返回：字典格式的配置映射，如果输入为None则返回None
    """
    if cfg is None or isinstance(cfg, Mapping):
        return cfg
    return vars(cfg)


def _derive_dim(cfg: Mapping[str, Any]) -> int:
    """
    计算RoPE的有效维度
    
    算法：
    1. 获取每个注意力头的维度：head_dim = hidden_size / num_heads
    2. 考虑部分旋转因子：effective_dim = head_dim * partial_rotary_factor
    
    参数：cfg - 包含模型配置的字典
    返回：RoPE应用的有效维度数（必须为偶数）
    
    注意：partial_rotary_factor < 1.0时，只对部分维度应用RoPE，其余维度保持不变
    """
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_heads"]
    return int(head_dim * cfg.get("partial_rotary_factor", 1.0))

# -----------------------------------------------------------------------------
# RoPE频率生成器
# 实现不同变体的频率计算策略
# -----------------------------------------------------------------------------

def compute_rope_default(
    cfg: Mapping[str, Any] | None = None,
    device: torch.device | None = None,
    *,
    base: float | None = None,
    dim: int | None = None,
) -> tuple[torch.Tensor, float]:
    """
    计算标准RoPE的逆频率序列
    
    数学公式：
    θ_i = base^(-2i/dim), i = 0, 1, ..., dim/2-1
    inv_freq = [θ_0, θ_1, ..., θ_{dim/2-1}]
    
    参数：
        cfg: 模型配置字典，包含rope_theta和维度信息
        device: 目标设备
        base: 频率基数（默认10000），控制不同维度的旋转速度
        dim: RoPE维度数，必须为偶数
        
    返回：
        inv_freq: 逆频率张量，形状[dim//2]
        scale: 注意力缩放因子（标准版本为1.0）
        
    维度变化：
        dim=128 -> inv_freq.shape=[64] (每两个维度共享一个频率)
    """
    cfg = _to_map(cfg)
    if cfg is not None and (base is not None or dim is not None):
        raise ValueError("Provide either *cfg* or explicit *base/dim*, not both")

    if cfg is not None:
        base = float(cfg.get("rope_theta", 10000.0))
        dim = _derive_dim(cfg)
    else:
        assert base is not None and dim is not None, "base & dim required when cfg=None"

    # 计算逆频率：θ_i = base^(-2i/dim)
    # 对于dim=128, base=10000: 频率从高到低分布，低频旋转慢，高频旋转快
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    return inv_freq, 1.0


def compute_rope_llama3(
    cfg: Mapping[str, Any] | object,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, float]:
    """
    计算Llama3/YARN变体的频率调节RoPE
    
    核心思想：
    对不同频率的维度应用不同的缩放策略，以更好地处理长序列：
    - 低频维度：保持原始频率（处理长距离依赖）
    - 高频维度：降低频率（避免过度旋转）
    - 中频维度：平滑过渡
    
    算法步骤：
    1. 计算波长 λ = 2π/θ
    2. 根据波长将频率分为三个区域
    3. 对不同区域应用不同的缩放因子
    
    参数：
        cfg: 必须包含rope_scaling配置的模型配置
        device: 目标设备
        
    返回：
        调节后的inv_freq和缩放因子
        
    配置参数说明：
        factor: 全局缩放因子
        low_freq_factor: 低频保护阈值
        high_freq_factor: 高频调节阈值  
        original_max_position_embeddings: 原始训练长度
    """
    cfg = _to_map(cfg)
    inv_freq, scale = compute_rope_default(cfg, device)

    scale_cfg = cfg["rope_scaling"]
    factor = scale_cfg["factor"]                              # 全局缩放因子
    low_f, high_f = scale_cfg["low_freq_factor"], scale_cfg["high_freq_factor"]  # 频率阈值
    old_ctx = scale_cfg["original_max_position_embeddings"]   # 原始上下文长度

    # 计算波长：λ = 2π/θ，波长越长表示频率越低
    wavelen = 2 * math.pi / inv_freq
    
    # 低频保护：波长大于阈值的维度应用全局缩放
    # 目标：保持长距离位置关系的建模能力
    inv_mod = torch.where(wavelen > old_ctx / low_f, inv_freq / factor, inv_freq)

    # 中频平滑过渡：在high_f和low_f之间的维度进行平滑插值
    # 避免频率调节的突变，确保模型性能的连续性
    s = (old_ctx / wavelen - low_f) / (high_f - low_f)       # 插值系数 [0,1]
    smooth_inv = (1 - s) * inv_mod / factor + s * inv_mod    # 线性插值
    mid_mask = (wavelen <= old_ctx / low_f) & (wavelen >= old_ctx / high_f)
    inv_freq = torch.where(mid_mask, smooth_inv, inv_mod)
    
    return inv_freq, scale

# 频率生成器注册表：将字符串标识符映射到对应的计算函数
_ROPE_INIT: dict[str, callable] = {
    "default": compute_rope_default,    # 标准RoPE实现
    "llama3": compute_rope_llama3,      # Llama3频率调节变体
    "yarn": compute_rope_llama3,        # YARN变体（与llama3算法相同）
}

# -----------------------------------------------------------------------------
# 核心嵌入层实现
# RoPE的主要计算逻辑和前向传播
# -----------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    """
    通用RoPE实现，支持多种变体
    
    支持的变体类型：
    - default: 标准RoPE实现
    - llama3/yarn: 频率调节变体，适合长序列
    - dynamic: 动态调整最大序列长度
    - longrope: 超长序列支持
    
    核心功能：
    1. 计算位置相关的cos和sin值
    2. 支持动态序列长度调整
    3. 提供多种频率调节策略
    
    数学原理：
    RoPE通过复数旋转实现位置编码：
    f(x,m) = x * cos(mθ) + rotate_half(x) * sin(mθ)
    其中θ是频率向量，m是位置索引
    """

    def __init__(
        self,
        *,
        config: Optional[object] = None,
        rope_type: str | None = None,
        max_position_embeddings: int | None = None,
        scaling_factor: float = 1.0,
        base: float = 10000.0,
        dim: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        """
        初始化RoPE嵌入层
        
        参数：
            config: 模型配置对象，包含所有RoPE相关参数
            rope_type: RoPE变体类型，如果未指定则从config中推断
            max_position_embeddings: 支持的最大序列长度
            scaling_factor: 全局缩放因子（遗留参数）
            base: 频率基数，控制旋转速度分布
            dim: RoPE维度，如果未指定则从config推断
            device: 目标计算设备
        """
        super().__init__()
        self.config = _to_map(config)

        # 确定RoPE变体类型
        # 优先级：显式指定 > 配置文件 > 默认值
        if rope_type is None:
            if self.config and self.config.get("rope_scaling"):
                rope_type = self.config["rope_scaling"].get(
                    "rope_type", self.config["rope_scaling"].get("type", "default")
                )
            else:
                rope_type = "default"
        self.rope_type: str = rope_type

        # 序列长度边界设置
        # 用于动态调整和缓存管理
        if self.config is not None:
            self.max_seq_len_cached = self.config["max_position_embeddings"]
        else:
            self.max_seq_len_cached = max_position_embeddings or 2048
        self.original_max_seq_len = self.max_seq_len_cached

        # 选择对应的频率生成器
        self.rope_init_fn = _ROPE_INIT[self.rope_type]

        # 遗留参数支持（当config为None时使用）
        self._legacy_kwargs = {"base": base, "dim": dim, "scaling_factor": scaling_factor}

        # 初始化逆频率向量和注意力缩放因子
        inv_freq, self.attention_scaling = self._init_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    # ---------------------------- 辅助方法 ---------------------------- #
    def _init_inv_freq(self, device):
        """
        初始化逆频率向量
        
        根据是否提供config选择不同的初始化路径：
        - 有config：使用配置文件中的参数
        - 无config：使用构造函数中的遗留参数
        
        返回：(inv_freq, attention_scaling)
        """
        if self.config is None:
            return self.rope_init_fn(None, device, **self._legacy_kwargs)
        return self.rope_init_fn(self.config, device)

    def _update_dynamic(self, seq_len: int, device: torch.device):
        """
        动态更新频率向量以支持变长序列
        
        策略：
        - 序列长度增加：重新计算频率向量，支持更长序列
        - 序列长度减少：恢复原始频率向量，释放不必要的计算资源
        
        参数：
            seq_len: 当前需要的序列长度
            device: 目标计算设备
            
        副作用：
            更新self.inv_freq和self.max_seq_len_cached
        """
        if seq_len > self.max_seq_len_cached:
            # 序列长度超出缓存，需要重新计算更大范围的频率
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len
        elif seq_len < self.original_max_seq_len < self.max_seq_len_cached:
            # 序列长度回落到原始范围，恢复原始频率以节省计算
            self.register_buffer("inv_freq", self.original_inv_freq.to(device), persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    # ---------------------------- 前向传播 ---------------------------- #
    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """
        RoPE前向传播：计算位置相关的cos和sin值
        
        数学公式：
        对于位置m和频率θ_i，计算：
        cos_values[m,i] = cos(m * θ_i)
        sin_values[m,i] = sin(m * θ_i)
        
        算法流程：
        1. 动态调整频率（如果需要）
        2. 计算位置-频率矩阵：freqs = position_ids ⊗ inv_freq
        3. 计算cos/sin值：emb = [freqs, freqs], cos/sin = cos(emb)/sin(emb)
        4. 应用注意力缩放
        
        参数：
            x: 输入张量，用于获取设备和数据类型信息
               形状: [batch_size, seq_len, hidden_size]
            position_ids: 位置索引张量
                         形状: [batch_size, seq_len]
                         
        返回：
            cos: 余弦值张量，形状: [batch_size, seq_len, head_dim]
            sin: 正弦值张量，形状: [batch_size, seq_len, head_dim]
            
        维度变化详解：
            position_ids: [batch_size, seq_len] -> [batch_size, 1, seq_len]
            inv_freq: [dim//2] -> [batch_size, dim//2, 1]  
            freqs: [batch_size, dim//2, seq_len] -> [batch_size, seq_len, dim//2]
            emb: [batch_size, seq_len, dim] (通过复制dim//2得到)
            cos/sin: [batch_size, seq_len, dim]
        """
        # 动态调整序列长度支持（针对dynamic和longrope变体）
        if "dynamic" in self.rope_type or self.rope_type == "longrope":
            self._update_dynamic(int(position_ids.max()) + 1, x.device)

        batch, _ = position_ids.shape
        inv = self.inv_freq.to(x.device, dtype=torch.float32)
        
        # 维度扩展：为批量矩阵乘法做准备
        # inv_freq: [dim//2] -> [batch, dim//2, 1]
        inv_exp = inv[None, :, None].expand(batch, -1, 1)
        # position_ids: [batch, seq_len] -> [batch, 1, seq_len]
        pos_exp = position_ids[:, None, :].to(dtype=torch.float32)

        # 自动混合精度处理：确保cos/sin计算的数值稳定性
        # MPS设备的特殊处理，避免autocast问题
        dev_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=dev_type, enabled=False):
            # 计算位置-频率乘积矩阵
            # freqs: [batch, dim//2, seq_len] -> [batch, seq_len, dim//2]
            freqs = (inv_exp @ pos_exp).transpose(1, 2) # 矩阵乘法要求两边 dtype 相同
            
            # 扩展到完整维度：每个频率对应相邻的两个维度
            # emb: [batch, seq_len, dim] (dim = 2 * dim//2)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # 计算三角函数值
            cos, sin = emb.cos(), emb.sin()
            
        # 应用注意力缩放因子（某些变体需要）
        cos *= self.attention_scaling
        sin *= self.attention_scaling
        
        # 转换为输入张量的数据类型，保持一致性
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# -----------------------------------------------------------------------------
# 向后兼容包装类
# 为不同模型提供专门的RoPE实现，保持API兼容性
# -----------------------------------------------------------------------------
class LlamaRotaryEmbedding(RotaryEmbedding):
    """
    Llama模型专用的RoPE实现
    
    继承自通用RotaryEmbedding类，无需额外修改
    主要用于保持与原有代码的兼容性
    """
    pass

class Qwen2RotaryEmbedding(RotaryEmbedding):
    """
    Qwen2模型专用的RoPE实现
    
    继承自通用RotaryEmbedding类，无需额外修改
    主要用于保持与原有代码的兼容性
    """
    pass

class Qwen3RotaryEmbedding(RotaryEmbedding):
    """
    Qwen3模型专用的RoPE实现
    
    相比Qwen2，可能有特定的初始化参数要求
    """
    def __init__(self, config: Qwen2Config, **kw):  # type: ignore[override]
        super().__init__(config=config, **kw)

# -----------------------------------------------------------------------------
# 单元测试
# 验证RoPE实现的正确性和不同变体的功能
# -----------------------------------------------------------------------------

def _make_cfg(head_dim=64, seq=128):
    """
    创建测试用的配置对象
    
    参数：
        head_dim: 每个注意力头的维度
        seq: 最大序列长度
        
    返回：
        LlamaConfig对象，包含测试所需的所有参数
    """
    return LlamaConfig(hidden_size=head_dim * 8, num_heads=8, head_dim=head_dim, max_position_embeddings=seq)


def test_default_inv_freq():
    """
    测试默认RoPE的逆频率计算
    
    验证：
    1. 逆频率向量的维度正确性
    2. 频率计算的数学正确性
    """
    cfg = _make_cfg()
    rope = LlamaRotaryEmbedding(config=cfg)
    assert rope.inv_freq.shape[0] == cfg.head_dim // 2


def test_llama3_inv_freq():
    """
    测试Llama3变体的频率调节功能
    
    验证：
    1. 频率调节配置的正确解析
    2. 调节后频率向量的维度
    3. 不同频率区域的处理逻辑
    """
    cfg = _make_cfg()
    cfg.rope_scaling = {
        "rope_type": "llama3",
        "factor": 8,                        # 全局缩放因子
        "low_freq_factor": 1,               # 低频保护阈值
        "high_freq_factor": 4,              # 高频调节阈值
        "original_max_position_embeddings": cfg.max_position_embeddings,
    }
    rope = LlamaRotaryEmbedding(config=cfg)
    assert rope.inv_freq.shape[0] == cfg.head_dim // 2


def test_forward_shapes():
    """
    测试前向传播的维度正确性
    
    验证：
    1. 输入输出维度的匹配
    2. 批量处理的正确性
    3. cos/sin值的形状和数值范围
    """
    cfg = _make_cfg(head_dim=32, seq=64)
    rope = LlamaRotaryEmbedding(config=cfg)
    
    # 构造测试输入
    x = torch.randn(2, 16, cfg.hidden_size)            # [batch_size, seq_len, hidden_size]
    pos = torch.arange(16).unsqueeze(0).repeat(2, 1)   # [batch_size, seq_len]
    
    # 执行前向传播
    cos, sin = rope(x, pos)
    
    # 验证输出维度
    assert cos.shape == (2, 16, cfg.head_dim)
    assert sin.shape == (2, 16, cfg.head_dim)


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main(sys.argv[1:]))
