import torch
import logging

logger = logging.getLogger(__name__)

class ReqTokensManager:
    """请求令牌管理器类
    
    该类用于管理多个请求序列的 KV 缓存内存分配和释放。
    主要负责跟踪每个请求的状态以及对应的 token 表映射关系。
    
    TokenTable 将一系列 kv tokens 映射到一组token 表中, 每个 token 表代表请求序列分配的 kv cache 内存空间。
    
    Attributes:
        max_can_use_req_size (int): 最大可使用的请求数量
        can_use_req_size (int): 当前可用的请求数量
        max_seq_len (int): 每个请求的最大序列长度
        req_state (torch.Tensor): 请求状态张量，0表示空闲，1表示已分配
        b_req_tokens_table (torch.Tensor): 批量请求token表，存储每个请求的token索引
    """

    def __init__(self, max_request_num, max_seq_len, mem_manager=None, device="cuda"):
        """初始化请求令牌管理器
        
        Args:
            max_request_num (int): 最大支持的并发请求数量
            max_seq_len (int): 每个请求支持的最大序列长度
            mem_manager: 内存管理器实例（暂未使用）
            device (str): 计算设备，默认为"cuda"
        """
        # 设置最大和当前可用的请求数量
        self.max_can_use_req_size = max_request_num
        self.can_use_req_size = max_request_num
        # 设置最大序列长度
        self.max_seq_len = max_seq_len
        
        # 请求状态张量：0表示空闲可分配，1表示已被分配使用
        # 形状为 [max_request_num]，每个元素对应一个请求槽位的状态
        self.req_state = torch.zeros(
            (max_request_num), dtype=torch.int32, device=device
        )
        
        # 批量请求tokens表：二维张量，形状为 [max_request_num, max_seq_len]
        # 用于存储每个请求的 Token 索引映射关系
        # 每行表示一个请求，每列表示该请求在特定序列位置上的 Token 索引
        self.b_req_tokens_table = torch.zeros(
            (max_request_num, max_seq_len), dtype=torch.int32, device=device
        )
        # 内存管理器（当前版本暂未使用）
        # self.mem_manager = mem_manager

    def alloc_req(self, request_num):
        """分配指定数量的请求槽位
        
        从可用的请求槽位中分配指定数量的槽位给新的请求。
        该方法会查找状态为0（空闲）的槽位，并将其标记为1（已分配）。
        
        Args:
            request_num (int): 需要分配的请求数量
            
        Returns:
            torch.Tensor or None: 
                - 成功时返回分配的请求索引张量
                - 失败时返回None（当可用槽位不足时）
        """
        # 检查是否有足够的可用请求槽位
        if request_num > self.can_use_req_size:
            logger.error(
                f"Insufficient requested capacity, remaining {self.can_use_req_size}"
            )
            return None

        # 查找状态为0（空闲）的请求槽位，并选择前request_num个
        logical_select_index = torch.nonzero(self.req_state == 0).reshape(-1)[
            :request_num
        ]
        # 将选中的槽位状态设置为1（已分配）
        self.req_state[logical_select_index] = 1
        # 更新可用请求数量
        self.can_use_req_size -= len(logical_select_index)
        return logical_select_index

    def free_reqs(self, free_req_index, free_token_index):
        """释放批量请求的槽位
        
        释放指定的多个请求槽位，将其状态重置为0（空闲），
        并更新可用请求数量计数器。
        
        Args:
            free_req_index (torch.Tensor): 要释放的请求索引数组（当前未使用）
            free_token_index (torch.Tensor): 要释放的token索引数组，用于重置请求状态
        """
        # 增加可用请求数量
        self.can_use_req_size += len(free_req_index)
        # 将对应的请求状态重置为0（空闲）
        self.req_state[free_token_index] = 0  # 对应批次请求的索引重新置为 0
        
        # 如果所有请求都被释放，记录调试信息
        if self.can_use_req_size == len(self.req_state):
            logger.debug(f"freed all request size {self.can_use_req_size}")
        # 暂未使用内存管理器进行实际内存释放
        # self.mem_manager.free(free_token_index)

    def free_req(self, free_req_index):
        """释放单个请求的槽位
        
        释放指定索引的单个请求槽位，将其状态重置为0（空闲），
        并增加可用请求数量计数器。包含边界检查以确保索引有效。
        
        Args:
            free_req_index (int): 要释放的请求索引
        """
        # 检查索引是否有效
        if free_req_index < 0 or free_req_index >= self.req_state.size(0):
            logger.error(f"Invalid free_req_index: {free_req_index}")
            return
        
        # 增加可用请求数量
        self.can_use_req_size += 1
        # 将指定请求的状态重置为0（空闲）
        self.req_state[free_req_index] = 0
        return

    def free_all(self):
        """释放所有请求的槽位
        
        将所有请求状态重置为0（空闲），并恢复所有可用请求数量。
        这是一个批量重置操作，用于清空所有分配状态。
        """
        # 恢复所有可用请求数量
        self.can_use_req_size = self.max_can_use_req_size
        # 将所有请求状态重置为0（空闲）
        self.req_state[:] = 0
