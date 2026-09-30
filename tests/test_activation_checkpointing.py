"""激活检查点（第 14 章）：打开后 loss 和梯度必须与不打开时一致。"""

import torch

from zero.config import ModelConfig
from zero.model import Transformer


def test_checkpointing_matches_plain_backward():
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=97, dim=32, n_layers=3, n_heads=4, n_kv_heads=2, ffn_dim=64, max_seq_len=32
    )
    model = Transformer(cfg)
    x = torch.randint(0, 97, (2, 16))
    y = torch.randint(0, 97, (2, 16))

    def grads(flag: bool):
        model.zero_grad(set_to_none=True)
        model.activation_checkpointing = flag
        model.train()
        loss = model.loss(x, y)
        loss.backward()
        return loss.detach(), [p.grad.clone() for p in model.parameters()]

    l0, g0 = grads(False)
    l1, g1 = grads(True)
    assert torch.allclose(l0, l1)
    for a, b in zip(g0, g1):
        torch.testing.assert_close(a, b)
