# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.models.layers.transformer.rtdetr_layers import (
    RTDETRHybridEncoder, RotaryMultiheadAttention)


def test_rotary_multihead_attention_forward_and_backward():
    attention = RotaryMultiheadAttention(embed_dims=32, num_heads=4)
    query = torch.randn(2, 6, 32, requires_grad=True)
    rope = RTDETRHybridEncoder.build_2d_rope_position_embedding(
        h=2, w=3, head_dim=8, base=10000., device=query.device)

    output = attention(query, rope=rope)

    assert output.shape == query.shape
    output.sum().backward()
    assert query.grad is not None


def test_hybrid_encoder_uses_rope_without_absolute_position_embedding():
    encoder = RTDETRHybridEncoder(
        in_channels=[32],
        use_encoder_idx=[0],
        num_encoder_layers=1,
        use_rope=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=32, num_heads=4),
            ffn_cfg=dict(embed_dims=32, feedforward_channels=64)),
        fpn_cfg=None)

    outputs = encoder((torch.randn(2, 32, 2, 3), ))

    assert outputs[0].shape == (2, 32, 2, 3)
    assert isinstance(encoder.transformer_blocks[0].layers[0].self_attn,
                      RotaryMultiheadAttention)
