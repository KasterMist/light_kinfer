import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from transformers import AutoConfig, AutoModelForCausalLM
from light_kinfer.utils.common import getProjectPath
from pathlib import Path


def test_load_llama_config():
    # 模型路径
    model_path = Path(os.path.abspath(os.path.join(getProjectPath(), "..", "download_models/Llama-3.2-1B-Instruct")))
    
    # 使用 AutoConfig 加载模型配置
    config = AutoConfig.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_config(config)
    
    # 打印模型的构造
    print("模型构造:")
    print(config)

    print("模型信息:")
    print(model)

    

