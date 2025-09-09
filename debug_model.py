import torch
from light_kinfer.models.naive_llama_backup_2 import LlamaForCausalLM
from transformers import AutoTokenizer

# 加载模型
model_path = "download_models/Llama-3.2-1B-Instruct"
model = LlamaForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# 检查模型配置
print("Model config:")
print(f"vocab_size: {model.config.vocab_size}")
print(f"hidden_size: {model.config.hidden_size}")
print(f"num_layers: {model.config.num_layers}")
print(f"num_heads: {model.config.num_heads}")
print(f"num_kv_heads: {model.config.num_kv_heads}")

# 检查嵌入层权重
print(f"\nEmbedding weight shape: {model.model.embed_tokens.weight.shape}")
print(f"LM head weight shape: {model.lm_head.weight.shape}")

# 测试简单推理
model.eval()
model = model.half().cuda()

# 简单测试输入
input_text = "Hello"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs.input_ids.cuda()

print(f"\nInput IDs: {input_ids}")
print(f"Input text: {input_text}")

with torch.no_grad():
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    
print(f"Output logits shape: {logits.shape}")
print(f"Max logit value: {logits.max().item()}")
print(f"Min logit value: {logits.min().item()}")

# 生成下一个token
next_token_id = torch.argmax(logits[0, -1, :], dim=-1)
next_token = tokenizer.decode(next_token_id)
print(f"Next token ID: {next_token_id.item()}")
print(f"Next token: '{next_token}'")
