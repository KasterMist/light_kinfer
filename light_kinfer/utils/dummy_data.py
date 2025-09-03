import torch


class DummyInputGenerator:
    """一个用于生成虚拟输入以进行内存分析的类。"""

    def __init__(self, device="cuda:5"):
        """
        初始化 DummyInputGenerator 类。

        Args:
            device (str): 设备名称，例如 "cuda:0" 表示使用第 1 个 GPU。
        """
        self.device = device

    def generate_dummy_input(self, model_config, batch_size=1, seq_len=32):
        """
        生成用于内存分析的虚拟输入张量。

        Args:
            model_config: 模型配置对象，包含模型的相关参数（如词汇表大小）。
            batch_size (int): 虚拟输入的批量大小。
            seq_len (int): 虚拟输入的序列长度。

        Returns:
            tuple: 包含以下两个元素的元组：
                - dummy_input: 虚拟输入张量，形状为 (batch_size, seq_len)。
                - dummy_position_ids: 虚拟位置 ID 张量，形状为 (batch_size, seq_len)。
        """
        # 生成随机整数张量，表示虚拟输入数据
        dummy_input = torch.randint(
            0, model_config.vocab_size,  # 随机整数范围为 [0, vocab_size)
            (batch_size, seq_len),  # 张量形状为 (batch_size, seq_len)
            device=self.device  # 张量所在设备
        )
        # 生成位置 ID 张量，表示序列中每个位置的索引
        dummy_position_ids = torch.arange(
            0, seq_len,  # 范围为 [0, seq_len)
            dtype=torch.long,  # 数据类型为长整型
            device=self.device  # 张量所在设备
        ).unsqueeze(0).expand(batch_size, -1)  # 扩展为 (batch_size, seq_len)

        return dummy_input, dummy_position_ids