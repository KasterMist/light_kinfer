import torch
from typing import Optional, List
from pathlib import Path

from light_kinfer.utils.common import get_gpu_memory, detect_device, count_tokens, get_model_type
from light_kinfer.utils.prompt_templates import get_prompter
from light_kinfer.utils.logger import get_logger
from light_kinfer.models.naive_llama import LlamaForCausalLM

import sys, os, time
from transformers import AutoTokenizer

logger = get_logger(__name__)

class NaiveTextGenerator:
    def __init__(self, checkpoint_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=False)
        self.model = LlamaForCausalLM.from_pretrained(
            checkpoint_path,
        )
        print(self.model)
        
        # 设置模型数据类型和设备
        if self.device == "cuda":
            self.model = self.model.half().cuda()
        else:
            self.model = self.model.float()
            
        self.model.eval()
    
    @torch.inference_mode()
    def sample_top_p(self, probs: torch.Tensor, p: float) -> torch.Tensor:
        """
        执行 Top-p (Nucleus) 采样，从概率分布中采样下一个词。
        
        Args:
            probs: 形状为 [batch_size, vocab_size] 的概率分布张量
            p: 累积概率阈值（0到1之间）
        Returns:
            形状为 [batch_size, 1] 的采样得到的词索引
        """
        # 对概率分布进行降序排序
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # 计算累积概率
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # 创建掩码：标记累积概率超过阈值的位置
        mask = probs_sum - probs_sort > p
        # 将超过阈值的概率置为0
        probs_sort[mask] = 0.0
        # 重新归一化概率
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # 从处理后的概率分布中采样
        next_token_sorted_idx = torch.multinomial(probs_sort, num_samples=1)
        # 将采样的索引映射回原始词表空间
        next_token = torch.gather(probs_idx, -1, next_token_sorted_idx)
        return next_token

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        echo: bool = False,
    ) -> List[str]:
        """
        基于提示词生成文本序列
        
        Args:
            prompt_tokens: 输入提示的token序列列表
            max_gen_len: 最大生成长度
            temperature: 温度参数，控制随机性
            top_p: nucleus采样的概率阈值
            echo: 是否在输出中包含提示词
        Returns:
            生成的文本列表
        """
        bsz = len(prompt_tokens)  # 批次大小
        max_prompt_len = max(len(t) for t in prompt_tokens)
        # 计算总序列长度（提示词长度+生成长度）
        total_len = min(2048, max_gen_len + max_prompt_len)  # 假设最大长度为2048
        
        # 获取填充token的ID
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        
        # 创建tokens缓冲区并用pad_id填充
        tokens = torch.full(
            (bsz, total_len), pad_id, dtype=torch.long, device=self.device
        )
        
        # 填充输入提示词
        for i, token_ids in enumerate(prompt_tokens):
            tokens[i, :len(token_ids)] = torch.tensor(
                token_ids, dtype=torch.long, device=self.device
            )
        
        # 创建attention mask：标记非填充位置
        attention_mask = (tokens != pad_id).float()
        
        # 记录是否达到EOS
        eos_reached = torch.zeros(bsz, dtype=torch.bool, device=self.device)
        
        # 逐个位置生成token
        for cur_pos in range(max_prompt_len, total_len):
            # 准备当前输入
            current_tokens = tokens[:, :cur_pos]
            current_mask = attention_mask[:, :cur_pos]
            
            # 获取模型输出，直接调用模型的forward方法
            # outputs = self.model(
            #     input_ids=current_tokens,
            #     attention_mask=current_mask
            # ) 
            outputs = self.model.forward(
                input_ids=current_tokens,
                attention_mask=current_mask
            )
            
            # 获取最后一个位置的logits
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            
            # 应用温度缩放并计算概率分布
            probs = torch.softmax(logits / temperature, dim=-1)
            
            # 使用top-p采样获取下一个token
            next_token = self.sample_top_p(probs, top_p)
            
            # 更新tokens
            tokens[:, cur_pos] = next_token.squeeze(-1)
            
            # 更新attention mask
            attention_mask[:, cur_pos] = 1
            
            # 检查是否达到EOS
            eos_reached = eos_reached | (next_token.squeeze(-1) == self.tokenizer.eos_token_id)
            if eos_reached.all():
                break
        
        # 处理输出（是否包含原始提示）
        if not echo:
            output_tokens = [
                tokens[i, len(prompt_tokens[i]):cur_pos+1].tolist()
                for i in range(bsz)
            ]
        else:
            output_tokens = [
                tokens[i, :cur_pos+1].tolist()
                for i in range(bsz)
            ]
        
        # 解码生成的token序列
        generated_texts = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        return generated_texts

    def generate_text(
        self,
        prompt: str,
        max_gen_len: Optional[int] = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
        echo: bool = False,
    ) -> str:
        """
        便捷的文本生成接口
        """
        if max_gen_len is None:
            max_gen_len = 2048
            
        # 构建更好的提示模板
        prompt_template = f"[INST] {prompt} [/INST]"
            
        # 编码输入文本
        input_tokens = self.tokenizer.encode(prompt_template, return_tensors="pt")
        prompt_tokens = input_tokens.tolist()
        
        # 生成文本
        generated_texts = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            echo=echo
        )
        
        # 清理输出文本
        output_text = generated_texts[0]
        
        # 移除可能的指令标记
        output_text = output_text.replace("[INST]", "").replace("[/INST]", "").strip()
        
        return output_text

def main(
    prompt: str = "Hello, my name is",
    *,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: Optional[int] = 1024,
    echo: bool = False,
    checkpoint_path: str = "checkpoints/lit-llama/7B/"
):
    # 初始化生成器
    generator = NaiveTextGenerator(checkpoint_path)
    
    # 生成文本
    generated_text = generator.generate_text(
        prompt=prompt,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        echo=echo
    )
    
    print("\n生成的文本：")
    print(generated_text)



if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)