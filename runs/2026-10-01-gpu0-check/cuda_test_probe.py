"""pytest plugin: find the tests that really use CUDA when a GPU is available.

    PYTHONPATH=runs/2026-10-01-gpu0-check CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 \
        uv run pytest -q -p cuda_test_probe

Before each test, the plugin resets the CUDA peak-memory statistics. After the test, it checks if
the peak is higher than the memory that was already allocated when the test started. The test that
initializes the CUDA context for the first time gets a separate record
(`torch.cuda.get_rng_state_all()` in `rng_state()` only initializes the context; it does not allocate memory).
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
        base = torch.cuda.memory_allocated()  # Earlier tests can still hold GPU memory (cached models, for example)
    yield
    if not torch.cuda.is_initialized():
        return
    if not was_init:
        _first_init.append(item.nodeid)
    if torch.cuda.max_memory_allocated() > base:
        _allocated.append(item.nodeid)


def pytest_terminal_summary(terminalreporter):  # noqa: ANN001, ANN201
    tr = terminalreporter
    tr.write_line(f"[cuda-probe] Tests that initialized CUDA for the first time: {_first_init}")
    tr.write_line(f"[cuda-probe] Tests that allocated CUDA memory ({len(_allocated)}):")
    for n in _allocated:
        tr.write_line(f"[cuda-probe]   {n}")
