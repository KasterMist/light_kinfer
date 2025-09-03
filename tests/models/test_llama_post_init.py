#!/usr/bin/env python3
"""
演示 LlamaConfig 中 __post_init__() 的实际调用
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from light_kinfer.models.model_config import LlamaConfig

if __name__ == "__main__":
    print("=== 测试 LlamaConfig 的 __post_init__() 调用 ===")
    
    print("\n1. 创建默认配置:")
    config1 = LlamaConfig()
    print(f"num_heads: {config1.num_heads}")
    print(f"num_kv_heads: {config1.num_kv_heads}")
    print(f"hidden_size: {config1.hidden_size}")
    print(f"head_dim: {config1.head_dim}")
    print(f"intermediate_size: {config1.intermediate_size}")
    
    print("\n2. 使用自定义参数:")
    config2 = LlamaConfig(hidden_size=1024, num_heads=16, num_kv_heads=None)
    print(f"num_heads: {config2.num_heads}")
    print(f"num_kv_heads: {config2.num_kv_heads}")  # 应该被设为16
    print(f"hidden_size: {config2.hidden_size}")
    print(f"head_dim: {config2.head_dim}")  # 应该被计算为64
    print(f"intermediate_size: {config2.intermediate_size}")  # 应该被设为4096
    
    print("\n3. 使用 from_dict 方法:")
    config3 = LlamaConfig.from_dict({
        "num_attention_heads": 8,  # 使用别名
        "hidden_size": 512
    })
    print(f"num_heads: {config3.num_heads}")  # 别名映射后应该是8
    print(f"num_kv_heads: {config3.num_kv_heads}")  # 应该被设为8
    print(f"hidden_size: {config3.hidden_size}")
    print(f"head_dim: {config3.head_dim}")  # 应该被计算为64
    print(f"intermediate_size: {config3.intermediate_size}")  # 应该被设为2048
