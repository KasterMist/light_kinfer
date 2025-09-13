#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apply_weight_convert.py
~~~~~~~~~~~~~~~~~~~~
将 Qwen-2/3、Llama、LLaVA 等 HuggingFace / PyTorch-bin 权重
整理为 lite_llama 框架的自定义格式模型权重的小工具。

Usage
-----
python lite_llama/apply_weight_convert.py /path/to/weights [--model-type qwen3] [--device cuda]

Author: harleyszhang (2025-06-08)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Mapping

import torch
from tqdm.auto import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM,
                          LlavaConfig, LlavaForConditionalGeneration)

from light_kinfer.utils.logger import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# 通用工具函数
# --------------------------------------------------------------------------- #
def ensure_dir(path: Path) -> Path:
    """若目录不存在则创建，最后返回自身。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_state_dict(out_dir: Path, model_id: str, state: dict[str, torch.Tensor]) -> None:
    """保存 state_dict 并打印信息。"""
    torch.save(state, out_dir / f"{model_id}.pth", _use_new_zipfile_serialization=True)
    logger.info("✅ 已保存权重到 %s", out_dir / f"{model_id}.pth")


def copy_metadata(src: Path, dst: Path) -> None:
    """复制 *.json 与 tokenizer.model 等辅助文件。"""
    for file in src.glob("*.json"):
        shutil.copy2(file, dst)
    tok = src / "tokenizer.model"
    if tok.exists():
        shutil.copy2(tok, dst)


# --------------------------------------------------------------------------- #
# 修订后的 merge_kv_weights —— 只生成 kv_proj_weight，下划线风格
# --------------------------------------------------------------------------- #
def merge_kv_weights(state: dict[str, torch.Tensor],
                     prefix: str,
                     with_bias: bool = False) -> None:
    """
    将 K/V 投影合并为 kv_proj_weight / kv_proj_bias（可选），
    完全使用 **下划线** 键名，保证与 light_kinfer 的 Qwen3 实现一致。
    同时若此前转换过留下旧的 kv_proj.weight，也会被删除。
    """
    # ---------- 1. 找到现有 K/V ----------
    # 两种候选命名：点号风格  layers.0.self_attn.k_proj.weight
    #            下划线风格 layers.0.self_attn.k_proj_weight
    candidates = [
        (f"{prefix}.k_proj.weight", f"{prefix}.v_proj.weight"),
        (f"{prefix}.k_proj_weight", f"{prefix}.v_proj_weight"),
    ]
    for k_key, v_key in candidates:
        if k_key in state and v_key in state:
            break
    else:  # 没有任何一对匹配
        return

    # ---------- 2. 合并权重 ----------
    fused_k = f"{prefix}.kv_proj_weight"            # 目标键（下划线）
    state[fused_k] = torch.cat([state[k_key], state[v_key]], dim=0)
    del state[k_key], state[v_key]

    # ---------- 3. 合并 bias（可选） ----------
    if with_bias:
        bias_cands = [
            (f"{prefix}.k_proj.bias",  f"{prefix}.v_proj.bias"),
            (f"{prefix}.k_proj_bias",  f"{prefix}.v_proj_bias"),
        ]
        for kb_key, vb_key in bias_cands:
            if kb_key in state and vb_key in state:
                fused_b = f"{prefix}.kv_proj_bias"
                state[fused_b] = torch.cat([state[kb_key], state[vb_key]], dim=0)
                del state[kb_key], state[vb_key]
                break

    # ---------- 4. 如有旧版 kv_proj.weight，顺带删掉 ----------
    old_key = f"{prefix}.kv_proj.weight"
    if old_key in state:
        del state[old_key]


def build_mapping(common: Mapping[str, str],
                  layer_tpl: Mapping[str, str],
                  num_layers: int) -> dict[str, str]:
    """根据层数展开模板映射表。"""
    mapping = dict(common)
    for i in range(num_layers):
        mapping.update({hf.format(i=i): custom.format(i=i) for hf, custom in layer_tpl.items()})
    return mapping

# --------------------------------------------------------------------------- #
# 具体各模型映射规则
# --------------------------------------------------------------------------- #
_SPEC = {
    # Qwen-2
    "qwen2": {
        "common": {
            "model.norm.weight":         "norm_weight",
            "model.embed_tokens.weight": "embed_tokens.weight",
            "lm_head.weight":            "lm_head_weight",
        },
        "layer": {
            # q_proj/k_proj/... 同下
            "model.layers.{i}.self_attn.q_proj.weight":  "layers.{i}.self_attn.q_proj_weight",
            "model.layers.{i}.self_attn.q_proj.bias":    "layers.{i}.self_attn.q_proj_bias",
            "model.layers.{i}.self_attn.k_proj.weight":  "layers.{i}.self_attn.k_proj_weight",
            "model.layers.{i}.self_attn.k_proj.bias":    "layers.{i}.self_attn.k_proj_bias",
            "model.layers.{i}.self_attn.v_proj.weight":  "layers.{i}.self_attn.v_proj_weight",
            "model.layers.{i}.self_attn.v_proj.bias":    "layers.{i}.self_attn.v_proj_bias",
            "model.layers.{i}.self_attn.o_proj.weight":  "layers.{i}.self_attn.o_proj_weight",
            "model.layers.{i}.mlp.gate_proj.weight":     "layers.{i}.mlp.gate_proj.weight",
            "model.layers.{i}.mlp.up_proj.weight":       "layers.{i}.mlp.up_proj.weight",
            "model.layers.{i}.mlp.down_proj.weight":     "layers.{i}.mlp.down_proj.weight",
            "model.layers.{i}.input_layernorm.weight":   "layers.{i}.input_layernorm_weight",
            "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.post_attention_layernorm_weight",
        },
        "merge_bias": True,
    },

    # Qwen-3
    "qwen3": {
        "common": {
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.norm.weight":         "norm_weight",
            "lm_head.weight":            "lm_head_weight",
        },
        "layer": {
            "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.self_attn.q_proj_weight",
            "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.self_attn.k_proj_weight",
            "model.layers.{i}.self_attn.v_proj.weight": "layers.{i}.self_attn.v_proj_weight",
            "model.layers.{i}.self_attn.q_norm.weight": "layers.{i}.self_attn.q_norm_weight",
            "model.layers.{i}.self_attn.k_norm.weight": "layers.{i}.self_attn.k_norm_weight",
            "model.layers.{i}.self_attn.o_proj.weight": "layers.{i}.self_attn.o_proj_weight",
            "model.layers.{i}.mlp.gate_proj.weight":    "layers.{i}.mlp.gate_proj.weight",
            "model.layers.{i}.mlp.up_proj.weight":      "layers.{i}.mlp.up_proj.weight",
            "model.layers.{i}.mlp.down_proj.weight":    "layers.{i}.mlp.down_proj.weight",
            "model.layers.{i}.input_layernorm.weight":  "layers.{i}.input_layernorm_weight",
            "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.post_attention_layernorm_weight",
        },
        "merge_bias": False,
    },

    # Llama-HF
    "llama": {
        "common": {
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.norm.weight":         "norm_weight",
            "lm_head.weight":            "lm_head.weight",
        },
        "layer": {
            "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.self_attn.q_proj.weight",
            "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.self_attn.k_proj.weight",
            "model.layers.{i}.self_attn.v_proj.weight": "layers.{i}.self_attn.v_proj.weight",
            "model.layers.{i}.self_attn.o_proj.weight": "layers.{i}.self_attn.o_proj.weight",
            "model.layers.{i}.mlp.gate_proj.weight":    "layers.{i}.mlp.gate_proj.weight",
            "model.layers.{i}.mlp.up_proj.weight":      "layers.{i}.mlp.up_proj.weight",
            "model.layers.{i}.mlp.down_proj.weight":    "layers.{i}.mlp.down_proj.weight",
            "model.layers.{i}.input_layernorm.weight":  "layers.{i}.attention_norm_weight",
            "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.ffn_norm_weight",
        },
        "merge_bias": False,
    },

    # Llama-bin（原 Fairseq/Llama.PTH 格式）
    "llama-bin": {
        "common": {
            "tok_embeddings.weight": "embed_tokens.weight",
            "norm.weight":           "norm_weight",
            "output.weight":         "lm_head.weight",
        },
        "layer": {
            "layers.{i}.attention.wq.weight": "layers.{i}.attention.q_proj.weight",
            "layers.{i}.attention.wk.weight": "layers.{i}.attention.k_proj.weight",
            "layers.{i}.attention.wv.weight": "layers.{i}.attention.v_proj.weight",
            "layers.{i}.attention.wo.weight": "layers.{i}.attention.o_proj.weight",
            "layers.{i}.feed_forward.w1.weight": "layers.{i}.feed_forward.gate_proj.weight",
            "layers.{i}.feed_forward.w3.weight": "layers.{i}.feed_forward.up_proj.weight",
            "layers.{i}.feed_forward.w2.weight": "layers.{i}.feed_forward.down_proj.weight",
            "layers.{i}.attention_norm.weight":  "layers.{i}.attention_norm_weight",
            "layers.{i}.ffn_norm.weight":        "layers.{i}.ffn_norm_weight",
        },
        "merge_bias": False,
    },

    # LLaVA-Llama
    "llava": {
        "common": {
            "language_model.model.embed_tokens.weight": "language_model.embed_tokens.weight",
            "language_model.model.norm.weight":         "language_model.norm_weight",
            "language_model.lm_head.weight":            "language_model.lm_head.weight",
        },
        "layer": {
            "language_model.model.layers.{i}.self_attn.q_proj.weight": "language_model.layers.{i}.self_attn.q_proj.weight",
            "language_model.model.layers.{i}.self_attn.k_proj.weight": "language_model.layers.{i}.self_attn.k_proj.weight",
            "language_model.model.layers.{i}.self_attn.v_proj.weight": "language_model.layers.{i}.self_attn.v_proj.weight",
            "language_model.model.layers.{i}.self_attn.o_proj.weight": "language_model.layers.{i}.self_attn.o_proj.weight",
            "language_model.model.layers.{i}.mlp.gate_proj.weight":    "language_model.layers.{i}.mlp.gate_proj.weight",
            "language_model.model.layers.{i}.mlp.up_proj.weight":      "language_model.layers.{i}.mlp.up_proj.weight",
            "language_model.model.layers.{i}.mlp.down_proj.weight":    "language_model.layers.{i}.mlp.down_proj.weight",
            "language_model.model.layers.{i}.input_layernorm.weight":  "language_model.layers.{i}.attention_norm_weight",
            "language_model.model.layers.{i}.post_attention_layernorm.weight": "language_model.layers.{i}.ffn_norm_weight",
        },
        "merge_bias": False,
    },
}

# --------------------------------------------------------------------------- #
# 核心转换逻辑
# --------------------------------------------------------------------------- #
def convert(checkpoints_dir: Path,
            hf_state: dict[str, torch.Tensor],
            model_type: str,
            num_layers: int) -> dict[str, torch.Tensor]:
    """执行主转换流程并把结果保存到 my_weight/<model_id>/ 目录。"""
    spec = _SPEC[model_type]
    mapping = build_mapping(spec["common"], spec["layer"], num_layers)
    new_sd: dict[str, torch.Tensor] = {}
    
    # 记录未转换的参数
    unmapped_params = []
    mapped_params = []

    logger.info("转换前参数数量: %d", len(hf_state))
    # ---------- 1. 重映射 ----------
    for k, v in tqdm(hf_state.items(), desc=f"[{model_type}] 权重重映射"):
        if (ck := mapping.get(k)) is not None:
            new_sd[ck] = v
            mapped_params.append(k)
        else:
            unmapped_params.append(k)
            logger.debug("忽略未映射参数 %s", k)
    
    # 打印详细的转换信息
    logger.info("成功转换的参数数量: %d", len(mapped_params))
    logger.info("未转换的参数数量: %d", len(unmapped_params))
    
    if unmapped_params:
        logger.warning("以下参数没有被转换:")
        for param in sorted(unmapped_params):
            logger.warning("  - %s", param)

    # ---------- 2. 仅对 *Qwen* 系列执行 KV权重 合并(拼接) ----------
    kv_merged_count = 0
    if model_type.startswith("qwen") or model_type.startswith("llama"):              # 只处理 Qwen-2 / Qwen-3 等
        for i in range(num_layers):
            prefix = f"layers.{i}.self_attn"       # Qwen 无额外前缀
            before_count = len(new_sd)
            merge_kv_weights(new_sd, prefix, with_bias=spec["merge_bias"])
            after_count = len(new_sd)
            if before_count != after_count:
                kv_merged_count += 1
                logger.info("层 %d: KV权重合并完成，参数数量从 %d 变为 %d", i, before_count, after_count)

    if kv_merged_count > 0:
        logger.info("总共有 %d 层执行了KV权重合并", kv_merged_count)

    # ---------- 3. 保存 ----------
    script_root = Path(__file__).resolve().parent
    out_dir = ensure_dir(script_root / "my_weight" / checkpoints_dir.name)
    save_state_dict(out_dir, checkpoints_dir.name, new_sd)
    copy_metadata(checkpoints_dir, out_dir)

    logger.info("🎉 转换完成，共 %d 个参数", len(new_sd))
    
    # 如果有未转换的参数，再次提醒
    if unmapped_params:
        logger.warning("⚠️  总计有 %d 个参数未被转换，请检查是否需要更新映射规则", len(unmapped_params))

    return new_sd



# --------------------------------------------------------------------------- #
# CLI 辅助：由 config.json 判别模型类型
# --------------------------------------------------------------------------- #
def detect_model_type(checkpoints_dir: Path) -> str:
    """
    读取 config.json 中的 model_type 字段。
    若该字段在 _SPEC 中无法找到，则抛出错误提示。
    """
    cfg = AutoConfig.from_pretrained(checkpoints_dir, trust_remote_code=True)
    mtype = cfg.model_type.lower()
    # 某些模型可能需要额外归一化 / 映射
    alias = {
        "qwen2":   "qwen2",
        "qwen3":   "qwen3",
        "llama":   "llama",
        "llava":   "llava",
    }.get(mtype, mtype)      # 默认原样返回
    if alias not in _SPEC:
        raise ValueError(f"暂不支持的 model_type '{mtype}'，请检查映射表")
    return alias


def load_hf_state(checkpoints_dir: Path,
                  model_type: str,
                  device: str = "cpu") -> dict[str, torch.Tensor]:
    """加载 HF / bin 权重到 state_dict。"""
    
    # 首先检查是否存在直接的 .pth 文件
    # pth_files = list(checkpoints_dir.glob("*.pth"))
    # if pth_files:
    #     logger.info("发现 .pth 文件，直接加载: %s", pth_files[0])
    #     state_dict = torch.load(pth_files[0], map_location=device)
    #     # 如果加载的是完整模型对象，提取 state_dict
    #     if hasattr(state_dict, 'state_dict'):
    #         state_dict = state_dict.state_dict()
    #     elif not isinstance(state_dict, dict):
    #         raise ValueError(f"无法从 {pth_files[0]} 中提取 state_dict")
    #     return state_dict
    
    # 如果没有 .pth 文件，使用 HuggingFace 加载方式
    if model_type == "llava":
        model = (LlavaForConditionalGeneration
                 .from_pretrained(checkpoints_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)
                 .to(device))
    else:
        model = (AutoModelForCausalLM
                 .from_pretrained(checkpoints_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)
                 .to(device))
    return model.state_dict()


def get_num_layers(checkpoints_dir: Path, model_type: str) -> int:
    """从 config 中提取 Transformer 层数。"""
    if model_type == "llava":
        cfg = LlavaConfig.from_pretrained(checkpoints_dir)
        return cfg.text_config.num_hidden_layers
    cfg = AutoConfig.from_pretrained(checkpoints_dir)
    return cfg.num_hidden_layers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HF / bin checkpoints into Lite-LLaMA format.")
    parser.add_argument("--checkpoints_dir", type=Path, help="模型权重目录")
    parser.add_argument("--model-type",
                        choices=_SPEC.keys(),
                        help="显式指定模型类型；默认根据目录名猜测")
    parser.add_argument("--device", default="cuda",
                        help="加载权重时使用的设备 (default: cuda)")
    args = parser.parse_args()

    ckpt_dir: Path = args.checkpoints_dir.resolve()
    
    # 1️⃣ **直接从 config.json 读取 model_type** 
    model_type = detect_model_type(ckpt_dir)
    logger.info("检测到 model_type = %s", model_type)

    # 2️⃣ 获取层数
    num_layers = get_num_layers(ckpt_dir, model_type)
    logger.info("Transformer 层数 %d", num_layers)

    # 3️⃣ 加载权重并执行转换
    hf_sd = load_hf_state(ckpt_dir, model_type, device=args.device)
    convert(ckpt_dir, hf_sd, model_type, num_layers)

# tests/test_convert.py
import torch
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

# ---------- 公共工具 ----------
def dummy_state_dict_qwen(num_layers: int, hidden: int = 4):
    """构造一个最小化的 Qwen state_dict，仅包含 KV/O/W1/W2 ... 层权重。"""
    sd = {
        "model.norm.weight": torch.ones(hidden),
        "model.embed_tokens.weight": torch.zeros(10, hidden),
        "lm_head.weight": torch.zeros(hidden, 10),
    }
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        sd.update({
            f"{prefix}.self_attn.q_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.self_attn.k_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.self_attn.v_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.self_attn.o_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.mlp.gate_proj.weight": torch.randn(2 * hidden, hidden),
            f"{prefix}.mlp.up_proj.weight":   torch.randn(2 * hidden, hidden),
            f"{prefix}.mlp.down_proj.weight": torch.randn(hidden, 2 * hidden),
            f"{prefix}.input_layernorm.weight": torch.ones(hidden),
            f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden),
        })
    return sd


def dummy_state_dict_llama(num_layers: int, hidden: int = 4):
    """构造最小化 Llama HF 权重，方便测试不进行 KV 合并。"""
    sd = {
        "model.norm.weight": torch.ones(hidden),
        "model.embed_tokens.weight": torch.zeros(10, hidden),
        "lm_head.weight": torch.zeros(hidden, 10),
    }
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        sd.update({
            f"{prefix}.self_attn.q_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.self_attn.k_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.self_attn.v_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.self_attn.o_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.mlp.gate_proj.weight": torch.randn(2 * hidden, hidden),
            f"{prefix}.mlp.up_proj.weight":   torch.randn(2 * hidden, hidden),
            f"{prefix}.mlp.down_proj.weight": torch.randn(hidden, 2 * hidden),
            f"{prefix}.input_layernorm.weight": torch.ones(hidden),
            f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden),
        })
    return sd


# ---------- fixtures ----------
@pytest.fixture(scope="function")
def tmp_ckpt_dir(tmp_path: Path):
    """创建临时 checkpoints 目录并写入最小 config.json。"""
    def _factory(model_type: str):
        ckpt = tmp_path / model_type
        ckpt.mkdir()
        (ckpt / "config.json").write_text(json.dumps({"model_type": model_type}))
        return ckpt
    return _factory


# ---------- 测试映射 ----------
@pytest.mark.parametrize("model_type", ["qwen3", "llama"])
def test_mapping_and_num_params(tmp_ckpt_dir, model_type):
    ckpt_dir = tmp_ckpt_dir(model_type)
    num_layers = 2

    # 构造假权重
    sd = (dummy_state_dict_qwen(num_layers) if model_type.startswith("qwen")
          else dummy_state_dict_llama(num_layers))

    # 跑转换
    new_sd = convert(ckpt_dir, sd, model_type, num_layers)

    # 基础 key 应当存在
    assert "embed_tokens.weight" in new_sd
    assert "norm_weight" in new_sd


# ---------- Qwen KV 合并 ----------
def test_qwen_kv_merge(tmp_ckpt_dir):
    model_type = "qwen3"
    num_layers = 1
    ckpt_dir = tmp_ckpt_dir(model_type)
    sd = dummy_state_dict_qwen(num_layers)

    new_sd = convert(ckpt_dir, sd, model_type, num_layers)

    # KV fused weight 应出现
    kv_key = "layers.0.self_attn.kv_proj.weight"
    assert kv_key in new_sd

    # 原 K、V 不应保留
    assert "layers.0.self_attn.k_proj_weight" not in new_sd
    assert "layers.0.self_attn.v_proj_weight" not in new_sd

    # 维度检查：concat 后第一维应为 2*hidden
    hidden = sd["model.norm.weight"].numel()     # 4
    assert new_sd[kv_key].shape[0] == 2 * hidden


# ---------- Llama 不合并 ----------
def test_llama_no_kv_merge(tmp_ckpt_dir):
    model_type = "llama"
    num_layers = 1
    ckpt_dir = tmp_ckpt_dir(model_type)
    sd = dummy_state_dict_llama(num_layers)

    new_sd = convert(ckpt_dir, sd, model_type, num_layers)

    # KV fused 不存在
    assert "layers.0.self_attn.kv_proj.weight" not in new_sd
    # 原 K/V 仍然存在
    assert "layers.0.self_attn.k_proj.weight" in new_sd
    assert "layers.0.self_attn.v_proj.weight" in new_sd

if __name__ == "__main__":
    main()
