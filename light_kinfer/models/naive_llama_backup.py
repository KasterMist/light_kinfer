import torch
import torch.nn as nn
import torch.nn.functional as F

class LlamaRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return x / (norm + self.eps) * self.weight

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        # 占位实现
    def forward(self, x):
        return x

class LlamaSdpaAttention(nn.Module):
    def __init__(self, hidden_size, kv_size):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding()

    def forward(self, x):
        # 简化实现
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # rotary_emb 占位
        q = self.rotary_emb(q)
        k = self.rotary_emb(k)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn_output = torch.matmul(F.softmax(attn_weights, dim=-1), v)
        out = self.o_proj(attn_output)
        return out

class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = self.act_fn(gate) * up
        out = self.down_proj(hidden)
        return out

class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size, kv_size, intermediate_size):
        super().__init__()
        self.self_attn = LlamaSdpaAttention(hidden_size, kv_size)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)
        self.input_layernorm = LlamaRMSNorm(hidden_size)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size)

    def forward(self, x):
        x_norm = self.input_layernorm(x)
        attn_out = self.self_attn(x_norm)
        x = x + attn_out
        x_norm = self.post_attention_layernorm(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        return x

class LlamaModel(nn.Module):
    def __init__(self, vocab_size=128256, hidden_size=2048, kv_size=512, intermediate_size=8192, num_layers=16):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(hidden_size, kv_size, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = LlamaRMSNorm(hidden_size)
        self.rotary_emb = LlamaRotaryEmbedding()

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class LlamaForCausalLM(nn.Module):
    def __init__(self, vocab_size=128256, hidden_size=2048, kv_size=512, intermediate_size=8192, num_layers=16):
        super().__init__()
        self.model = LlamaModel(vocab_size, hidden_size, kv_size, intermediate_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.model(input_ids)
        logits = self.lm_head(x)
        return logits

    def generate(self, input_ids, max_length=20):
        # 简单贪心生成
        for _ in range(max_length):
            logits = self.forward(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
