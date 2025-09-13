"""
ReqTokensManager 类的 pytest 测试模块

该模块包含了对请求令牌管理器的所有功能测试，包括：
- 请求槽位的分配
- 单个和批量请求的释放
- 边界条件和错误处理
"""

import torch
import pytest
from unittest.mock import MagicMock

from light_kinfer.executor.req_tokens_manager import ReqTokensManager


class TestReqTokensManager:
    """ReqTokensManager 的测试类"""

    @pytest.fixture
    def device(self):
        """返回可用的计算设备"""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture
    def mem_manager_mock(self):
        """返回模拟的内存管理器"""
        return MagicMock()

    @pytest.fixture
    def req_tokens_manager(self, mem_manager_mock, device):
        """创建 ReqTokensManager 实例的 fixture"""
        return ReqTokensManager(
            max_request_num=10,
            max_seq_len=5,
            mem_manager=mem_manager_mock,
            device=device,
        )

    def test_initialization(self, req_tokens_manager, device):
        """测试 ReqTokensManager 的初始化"""
        assert req_tokens_manager.max_can_use_req_size == 10
        assert req_tokens_manager.can_use_req_size == 10
        assert req_tokens_manager.max_seq_len == 5
        assert req_tokens_manager.req_state.shape == (10,)
        assert req_tokens_manager.b_req_tokens_table.shape == (10, 5)
        assert req_tokens_manager.req_state.device.type == device
        assert torch.all(req_tokens_manager.req_state == 0)

    def test_alloc_req_success(self, req_tokens_manager):
        """测试成功分配请求槽位"""
        indices = req_tokens_manager.alloc_req(3)
        
        # 验证返回的索引数量正确
        assert len(indices) == 3
        # 验证分配的槽位状态为1
        assert torch.all(req_tokens_manager.req_state[indices] == 1)
        # 验证可用请求数量减少
        assert req_tokens_manager.can_use_req_size == 7

    def test_alloc_req_exceed_capacity(self, req_tokens_manager):
        """测试分配请求数量超过容量的情况"""
        indices = req_tokens_manager.alloc_req(11)
        
        # 验证返回 None
        assert indices is None
        # 验证可用请求数量未改变
        assert req_tokens_manager.can_use_req_size == 10
        # 验证所有槽位状态仍为0
        assert torch.all(req_tokens_manager.req_state == 0)

    def test_alloc_req_exact_capacity(self, req_tokens_manager):
        """测试分配恰好等于容量的请求数量"""
        indices = req_tokens_manager.alloc_req(10)
        
        assert len(indices) == 10
        assert torch.all(req_tokens_manager.req_state == 1)
        assert req_tokens_manager.can_use_req_size == 0

    def test_alloc_req_zero_requests(self, req_tokens_manager):
        """测试分配0个请求"""
        indices = req_tokens_manager.alloc_req(0)
        
        assert len(indices) == 0
        assert torch.all(req_tokens_manager.req_state == 0)
        assert req_tokens_manager.can_use_req_size == 10

    def test_free_reqs_success(self, req_tokens_manager):
        """测试成功释放批量请求"""
        # 先分配一些请求
        indices = req_tokens_manager.alloc_req(3)
        assert req_tokens_manager.can_use_req_size == 7
        
        # 释放这些请求
        req_tokens_manager.free_reqs(indices, indices)
        
        # 验证槽位状态重置为0
        assert torch.all(req_tokens_manager.req_state[indices] == 0)
        # 验证可用请求数量恢复
        assert req_tokens_manager.can_use_req_size == 10

    def test_free_reqs_partial(self, req_tokens_manager):
        """测试部分释放请求"""
        # 分配5个请求
        indices = req_tokens_manager.alloc_req(5)
        assert req_tokens_manager.can_use_req_size == 5
        
        # 释放其中3个
        free_indices = indices[:3]
        req_tokens_manager.free_reqs(free_indices, free_indices)
        
        # 验证部分槽位被释放
        assert torch.all(req_tokens_manager.req_state[free_indices] == 0)
        assert torch.all(req_tokens_manager.req_state[indices[3:]] == 1)
        assert req_tokens_manager.can_use_req_size == 8

    def test_free_req_success(self, req_tokens_manager):
        """测试成功释放单个请求"""
        # 分配一个请求
        indices = req_tokens_manager.alloc_req(1)
        req_index = indices[0].item()
        
        # 释放该请求
        req_tokens_manager.free_req(req_index)
        
        # 验证槽位状态重置为0
        assert req_tokens_manager.req_state[req_index] == 0
        # 验证可用请求数量恢复
        assert req_tokens_manager.can_use_req_size == 10

    def test_free_req_invalid_negative_index(self, req_tokens_manager):
        """测试释放无效的负索引"""
        initial_count = req_tokens_manager.can_use_req_size
        
        # 尝试释放负索引（应该不会抛出异常）
        req_tokens_manager.free_req(-1)
        
        # 验证状态未改变
        assert req_tokens_manager.can_use_req_size == initial_count
        assert torch.all(req_tokens_manager.req_state == 0)

    def test_free_req_invalid_large_index(self, req_tokens_manager):
        """测试释放无效的过大索引"""
        initial_count = req_tokens_manager.can_use_req_size
        
        # 尝试释放过大索引（应该不会抛出异常）
        req_tokens_manager.free_req(100)
        
        # 验证状态未改变
        assert req_tokens_manager.can_use_req_size == initial_count
        assert torch.all(req_tokens_manager.req_state == 0)

    def test_free_req_already_free(self, req_tokens_manager):
        """测试释放已经空闲的请求槽位"""
        # 释放一个未分配的槽位
        req_tokens_manager.free_req(0)
        
        # 验证可用请求数量增加
        assert req_tokens_manager.can_use_req_size == 11
        assert req_tokens_manager.req_state[0] == 0

    def test_free_all_success(self, req_tokens_manager):
        """测试成功释放所有请求"""
        # 分配一些请求
        req_tokens_manager.alloc_req(5)
        assert req_tokens_manager.can_use_req_size == 5
        
        # 释放所有请求
        req_tokens_manager.free_all()
        
        # 验证所有槽位状态重置为0
        assert torch.all(req_tokens_manager.req_state == 0)
        # 验证可用请求数量完全恢复
        assert req_tokens_manager.can_use_req_size == req_tokens_manager.max_can_use_req_size

    def test_free_all_empty(self, req_tokens_manager):
        """测试在没有分配任何请求的情况下释放所有请求"""
        # 在空状态下释放所有请求
        req_tokens_manager.free_all()
        
        # 验证状态保持不变
        assert torch.all(req_tokens_manager.req_state == 0)
        assert req_tokens_manager.can_use_req_size == req_tokens_manager.max_can_use_req_size

    def test_sequential_alloc_and_free(self, req_tokens_manager):
        """测试连续的分配和释放操作"""
        # 第一轮：分配3个请求
        indices1 = req_tokens_manager.alloc_req(3)
        assert len(indices1) == 3
        assert req_tokens_manager.can_use_req_size == 7
        
        # 释放第一轮的请求
        req_tokens_manager.free_reqs(indices1, indices1)
        assert req_tokens_manager.can_use_req_size == 10
        
        # 第二轮：分配5个请求
        indices2 = req_tokens_manager.alloc_req(5)
        assert len(indices2) == 5
        assert req_tokens_manager.can_use_req_size == 5
        
        # 部分释放
        req_tokens_manager.free_reqs(indices2[:2], indices2[:2])
        assert req_tokens_manager.can_use_req_size == 7

    def test_edge_case_allocate_after_partial_free(self, req_tokens_manager):
        """测试部分释放后重新分配的边界情况"""
        # 分配所有槽位
        indices = req_tokens_manager.alloc_req(10)
        assert req_tokens_manager.can_use_req_size == 0
        
        # 释放其中3个
        req_tokens_manager.free_reqs(indices[:3], indices[:3])
        assert req_tokens_manager.can_use_req_size == 3
        
        # 重新分配2个（应该复用之前释放的槽位）
        new_indices = req_tokens_manager.alloc_req(2)
        assert len(new_indices) == 2
        assert req_tokens_manager.can_use_req_size == 1
        
        # 验证新分配的索引是之前释放的槽位
        assert torch.all(torch.isin(new_indices, indices[:3]))

    @pytest.mark.parametrize("max_request_num,max_seq_len", [
        (1, 1),      # 最小配置
        (5, 10),     # 中等配置
        (100, 512),  # 大配置
    ])
    def test_different_configurations(self, mem_manager_mock, device, max_request_num, max_seq_len):
        """测试不同配置下的ReqTokensManager"""
        manager = ReqTokensManager(
            max_request_num=max_request_num,
            max_seq_len=max_seq_len,
            mem_manager=mem_manager_mock,
            device=device,
        )
        
        assert manager.max_can_use_req_size == max_request_num
        assert manager.can_use_req_size == max_request_num
        assert manager.max_seq_len == max_seq_len
        assert manager.req_state.shape == (max_request_num,)
        assert manager.b_req_tokens_table.shape == (max_request_num, max_seq_len)

    def test_device_consistency(self, mem_manager_mock):
        """测试设备一致性"""
        # 测试 CPU 设备
        manager_cpu = ReqTokensManager(
            max_request_num=5,
            max_seq_len=3,
            mem_manager=mem_manager_mock,
            device="cpu",
        )
        assert manager_cpu.req_state.device.type == "cpu"
        assert manager_cpu.b_req_tokens_table.device.type == "cpu"
        
        # 如果有GPU，测试 CUDA 设备
        if torch.cuda.is_available():
            manager_cuda = ReqTokensManager(
                max_request_num=5,
                max_seq_len=3,
                mem_manager=mem_manager_mock,
                device="cuda",
            )
            assert manager_cuda.req_state.device.type == "cuda"
            assert manager_cuda.b_req_tokens_table.device.type == "cuda"
