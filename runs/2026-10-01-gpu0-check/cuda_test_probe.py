"""pytest 插件：找出哪些测试在有 GPU 时真的用到了 CUDA。

    PYTHONPATH=runs/2026-10-01-gpu0-check CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 \
        uv run pytest -q -p cuda_test_probe

每个测试前清零 CUDA 的峰值显存统计，测试后看峰值是否超过测试开始时已占用的显存；CUDA 上下文第一次被初始化的测试单独记一笔
（`rng_state()` 里的 `torch.cuda.get_rng_state_all()` 只初始化上下文、不分配显存）。
"""

from __future__ import annotations

import pytest
import torch

_allocated: list[str] = []
_first_init: list[str] = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):  # noqa: ANN001, ANN201
    was_init = torch.cuda.is_initialized()
    base = 0
    if was_init:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()  # 之前的测试可能还留着显存（缓存的模型等）
    yield
    if not torch.cuda.is_initialized():
        return
    if not was_init:
        _first_init.append(item.nodeid)
    if torch.cuda.max_memory_allocated() > base:
        _allocated.append(item.nodeid)


def pytest_terminal_summary(terminalreporter):  # noqa: ANN001, ANN201
    tr = terminalreporter
    tr.write_line(f"[cuda-probe] 首次初始化 CUDA 的测试：{_first_init}")
    tr.write_line(f"[cuda-probe] 分配过 CUDA 显存的测试（{len(_allocated)} 个）：")
    for n in _allocated:
        tr.write_line(f"[cuda-probe]   {n}")
