"""
配置转换工具模块

该模块用于将 Hugging Face Transformers 库的配置对象转换为
自定义的模型配置对象，主要功能是实现配置格式的标准化转换。
"""

import transformers  # Hugging Face Transformers 库
from transformers import LlavaConfig  # LLaVA 多模态模型配置
import os

# 使用相对导入，无需修改 sys.path
from ..models.model_config import LlamaConfig  # 导入自定义的 Llama 配置类
from ..utils.common import getProjectPath  # 导入项目路径工具函数


def convert_transformers_to_custom_config(
    transformers_config: transformers.LlamaConfig,
) -> LlamaConfig:
    """
    将 Hugging Face Transformers 的 LlamaConfig 转换为自定义的 LlamaConfig
    
    Args:
        transformers_config: Hugging Face Transformers 库中的 LlamaConfig 对象
        
    Returns:
        LlamaConfig: 转换后的自定义 LlamaConfig 对象
    """
    # 将 transformers 配置转换为字典，便于后续处理
    config_dict = transformers_config.to_dict()
    print("transformers.LlamaConfig dict: ", config_dict)

    # 方法一：手动创建自定义 LlamaConfig 对象，逐个设置参数
    # 这种方法提供了更精确的控制，可以设置默认值
    manual_config = LlamaConfig(
        _name_or_path=config_dict.get("_name_or_path"),  # 模型名称或路径
        architectures=config_dict.get("architectures", ["LlamaForCausalLM"]),  # 模型架构列表
        max_position_embeddings=config_dict.get("max_position_embeddings", 4096),  # 最大位置编码长度
        model_type=config_dict.get("model_type", "llama"),  # 模型类型标识
        rms_norm_eps=config_dict.get("rms_norm_eps", 1e-5),  # RMS 归一化的 epsilon 值
        torch_dtype=config_dict.get("torch_dtype", "float16"),  # PyTorch 数据类型
        vocab_size=config_dict.get("vocab_size", 32064),  # 词汇表大小
        hidden_size=config_dict.get("hidden_size", 4096),  # 隐藏层维度
        intermediate_size=config_dict.get("intermediate_size", 11008),  # 前馈网络中间层维度
        num_layers=config_dict.get("num_hidden_layers", 32),  # 隐藏层数量
        num_heads=config_dict.get("num_attention_heads", 32),  # 注意力头数量
        num_kv_heads=config_dict.get("num_key_value_heads", None),  # KV 缓存头数量
    )
    
    # 方法二：使用 from_dict 方法（更简洁，利用别名映射功能）
    # custom_config = LlamaConfig.from_dict(config_dict)
    
    return manual_config


if __name__ == "__main__":
    """
    主程序入口：演示如何使用配置转换功能
    """
    # 加载 transformers 的 LlavaConfig（请替换为实际模型名称）
    # LLaVA 是一个多模态模型，包含文本和视觉两部分配置
    project_path = getProjectPath()
    # model_path 为project_path的上一级目录 加download_models/Llama-3.2-1B-Instruct
    model_path = os.path.abspath(os.path.join(project_path, "..", "download_models", "Llama-3.2-1B-Instruct"))
    transformers_config = LlavaConfig.from_pretrained(model_path)  # 从预训练模型加载配置

    # 转换为自定义配置
    # 注意：这里只转换 LLaVA 模型中的文本部分配置（text_config）
    custom_llama_config = convert_transformers_to_custom_config(
        transformers_config.text_config  # 提取 LLaVA 中的文本配置部分
    )

    # 打印自定义配置对象
    # print(json.dumps(custom_llama_config, indent=4, ensure_ascii=False))  # JSON 格式输出（已注释）
    print(custom_llama_config)  # 直接打印配置对象

"""
示例输出结果：
下面展示了转换后的 LlamaConfig 对象的典型输出格式和参数值

LlamaConfig(architectures=None, attention_bias=False, attention_dropout=0.0, 
bos_token_id=1, eos_token_id=2, head_dim=128, hidden_act='silu', 
initializer_range=0.02, hidden_size=4096, intermediate_size=11008, 
max_position_embeddings=2048, mlp_bias=False, model_type='llama', 
num_heads=32, num_layers=32, num_kv_heads=32, pretraining_tp=1, 
rms_norm_eps=1e-06, rope_scaling=None, rope_theta=10000.0, 
tie_word_embeddings=False, torch_dtype=None, transformers_version='4.40.2', 
use_cache=True, vocab_size=32000, max_batch_size=4, max_seq_len=2048, 
device='cuda')

参数说明：
- hidden_size: 隐藏层维度（4096）
- num_heads: 注意力头数量（32）
- num_layers: Transformer 层数（32）
- vocab_size: 词汇表大小（32000）
- max_seq_len: 最大序列长度（2048）
- device: 计算设备（cuda）
"""
