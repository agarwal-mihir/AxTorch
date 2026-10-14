import importlib.util
import unittest

import torch
import torch.nn.functional as F

from approxtorch.backend import ops
from approxtorch.transformer import approx_attention, approximate_model, lut_matmul

VALUES = torch.arange(-128, 128)
PRODUCT = VALUES[:, None] * VALUES[None, :]
EXACT = PRODUCT.int().reshape(-1)
# Sign-symmetric truncated multiplier: |a*b| with the low 6 bits dropped.
TRUNCATED = (PRODUCT.sign() * (PRODUCT.abs() >> 6 << 6)).int().reshape(-1)


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def causal_mask(n):
    return torch.full((n, n), float("-inf")).triu(1)


class LutMatmulTests(unittest.TestCase):
    def test_gemm_int8_indexes_lut_by_activation_then_weight(self):
        torch.manual_seed(0)
        lut = torch.randint(-1000, 1000, (65536,), dtype=torch.int32)
        a = torch.randint(-128, 128, (5, 7), dtype=torch.int8)
        b = torch.randint(-128, 128, (7, 3), dtype=torch.int8)
        brute = [[sum(int(lut[(int(a[m, k]) + 128) * 256 + int(b[k, n]) + 128]) for k in range(7))
                  for n in range(3)] for m in range(5)]
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            got = ops.gemm_int8(a.to(device), b.to(device), lut.to(device))
            self.assertEqual(got.tolist(), brute, device)

    def test_exact_lut_is_integer_matmul(self):
        a = torch.randint(-127, 128, (3, 64, 300), dtype=torch.int8)
        b = torch.randint(-127, 128, (300, 40), dtype=torch.int8)
        self.assertTrue(torch.equal(lut_matmul(a, b, EXACT, symmetric=True).long(), a.long() @ b.long()))


class AttentionTests(unittest.TestCase):
    def test_attention_tracks_sdpa_and_approximate_lut_perturbs_it(self):
        torch.manual_seed(0)
        q, k, v = torch.randn(3, 2, 4, 33, 64).unbind(0)
        mask = causal_mask(33)
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        exact = approx_attention(q, k, v, EXACT, attn_mask=mask)
        approx = approx_attention(q, k, v, TRUNCATED, attn_mask=mask)
        self.assertLess(rel_err(exact, expected), 0.02)
        self.assertGreater(rel_err(approx, expected), 2 * rel_err(exact, expected))


@unittest.skipUnless(importlib.util.find_spec("transformers"), "needs transformers")
class HuggingFaceTests(unittest.TestCase):
    def check(self, model, inputs, expected_names, **kwargs):
        with torch.no_grad():
            expected = model(inputs).logits
            lut = EXACT.clone()
            names = approximate_model(model, lut, **kwargs)
            exact = model(inputs).logits
            lut.copy_(TRUNCATED)  # in-place update switches every patched layer
            approx = model(inputs).logits
        self.assertEqual(len(names), expected_names)
        self.assertEqual(model.config._attn_implementation, "approx_lut")
        self.assertLess(rel_err(exact, expected), 0.03)
        self.assertGreater(rel_err(approx, expected), 2 * rel_err(exact, expected))

    def test_tiny_gemma3_matches_eager_with_exact_lut(self):
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig

        config = Gemma3TextConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=1, head_dim=16,
            sliding_window=8, layer_types=["sliding_attention", "full_attention"],
            attn_implementation="eager",
        )
        torch.manual_seed(0)
        self.check(Gemma3ForCausalLM(config).eval(), torch.randint(0, 128, (2, 24)), 2 * 7)

    def test_tiny_vit_matches_eager_with_exact_lut(self):
        from transformers import ViTConfig, ViTForImageClassification

        config = ViTConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128,
            image_size=28, patch_size=4, num_channels=1, num_labels=10, attn_implementation="eager",
        )
        torch.manual_seed(0)
        model = ViTForImageClassification(config).eval()
        self.check(model, torch.randn(3, 1, 28, 28), 2 * 6 + 1, skip=())


if __name__ == "__main__":
    unittest.main()
