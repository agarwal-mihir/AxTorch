import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

from approxtorch.backend import lut_kernel, ops
from approxtorch.transformer import lut_matmul, quantize

VALUES = torch.arange(-128, 128)
PRODUCT = VALUES[:, None] * VALUES[None, :]
# Sign-symmetric truncated multiplier: |a*b| with the low 6 bits dropped.
TRUNCATED = (PRODUCT.sign() * (PRODUCT.abs() >> 6 << 6)).int().reshape(-1)


def ints(low, high, *shape):
    return torch.randint(low, high, shape, dtype=torch.int8, device="cuda")


def floats(*shape):
    return torch.randn(*shape, device="cuda")


@unittest.skipUnless(
    torch.cuda.is_available() and lut_kernel._extension() is not None, "needs CUDA"
)
class LutKernelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # Asymmetric random LUT: catches swapped operand roles in modes 0/1.
        self.lut = torch.randint(-32767, 32768, (65536,), dtype=torch.int32, device="cuda")

    def test_conv_matches_bgemm_and_pytorch_epilogue_bit_for_bit(self):
        t = lut_kernel.table(self.lut, lut_kernel.CONV)
        for o, k, batch, l in [(6, 25, 3, 37), (16, 144, 5, 64), (33, 17, 2, 9), (70, 288, 2, 50)]:
            kernels, features = ints(-128, 128, o, k), ints(0, 128, batch, k, l)
            c, s1, bias = floats(o), floats(o).abs(), floats(o)
            acc = ops.bgemm_int8(features, kernels, self.lut).float()
            expected = (acc + c[None, :, None]) * s1[None, :, None] + bias[None, :, None]
            self.assertTrue(torch.equal(lut_kernel.conv(kernels, features, t, c, s1, bias), expected))

    def test_linear_matches_gemm_and_pytorch_epilogue_bit_for_bit(self):
        t = lut_kernel.table(self.lut, lut_kernel.LINEAR)
        for m, k, n in [(256, 784, 300), (7, 25, 10), (65, 17, 129)]:
            x, w = ints(0, 128, m, k), ints(-128, 128, k, n)
            c, x_scale, s2, bias = floats(n), floats(1).abs(), floats(n).abs(), floats(n)
            acc = ops.gemm_int8(x, w, self.lut).float()
            expected = (acc + c) * x_scale * s2 + bias
            got = lut_kernel.linear(x, w, t, c, x_scale.expand(n).contiguous(), s2, bias)
            self.assertTrue(torch.equal(got, expected))

    def test_signmag_batched_matches_gemm_with_strided_operands(self):
        lut = TRUNCATED.cuda()
        self.assertIsNone(lut_kernel.table(self.lut, lut_kernel.SIGNMAG))  # not sign-symmetric
        t = lut_kernel.table(lut, lut_kernel.SIGNMAG)
        a, keys = ints(-127, 128, 12, 50, 64), ints(-127, 128, 12, 50, 64)
        got = lut_kernel.batched(a, keys.transpose(1, 2), t)
        for z in range(12):
            self.assertTrue(torch.equal(got[z], ops.gemm_int8(a[z], keys[z].t().contiguous(), lut)))

    def test_implicit_conv2d_matches_im2col_plus_bgemm(self):
        t = lut_kernel.table(self.lut, lut_kernel.CONV)
        # (batch, cin, h, w, cout, kernel, stride, padding, dilation)
        for b, ci, h, w, o, ks, st, pd, dl in [
            (3, 16, 32, 32, 16, (3, 3), (1, 1), (1, 1), (1, 1)),
            (2, 32, 16, 16, 64, (3, 3), (2, 2), (1, 1), (1, 1)),
            (2, 16, 32, 32, 32, (1, 1), (2, 2), (0, 0), (1, 1)),
            (4, 1, 28, 28, 6, (5, 5), (1, 1), (2, 2), (1, 1)),
            (2, 3, 13, 11, 5, (3, 2), (2, 1), (2, 0), (2, 3)),
            (1, lut_kernel.MAX_IM2COL_K, 3, 3, 8, (1, 1), (1, 1), (0, 0), (1, 1)),
        ]:
            x_q = ints(0, 128, b, ci, h, w)
            kernels = ints(-128, 128, o, ci * ks[0] * ks[1])
            c, s1, bias = floats(o), floats(o).abs(), floats(o)
            cols = ops.im2col_int8(x_q, ks, st, pd, dl)
            oh = (h + 2 * pd[0] - dl[0] * (ks[0] - 1) - 1) // st[0] + 1
            ow = (w + 2 * pd[1] - dl[1] * (ks[1] - 1) - 1) // st[1] + 1
            expected = lut_kernel.conv(kernels, cols, t, c, s1, bias).reshape(b, o, oh, ow)
            x_pad = F.pad(x_q, (pd[1], pd[1], pd[0], pd[0]))
            ktab = lut_kernel.im2col_table(ci, ks, dl, *x_pad.shape[2:], "cuda")
            got = lut_kernel.conv2d(x_pad, kernels, t, ktab, (oh, ow), st, c, s1, bias)
            self.assertTrue(torch.equal(got, expected), (b, ci, h, w, o, ks, st, pd, dl))

    def test_quantize_matches_torch_expression(self):
        x = torch.cat([floats(100000) * 40, torch.arange(-300, 300, 0.25, device="cuda")])
        x = x[: 2 * 3 * 101 * 99].reshape(2, 3, 101, 99)
        for scale, zero in [(0.0173, 0.0), (0.5, 3.0), (1.0, 0.0), (0.0371, 17.0)]:
            s, zp = torch.tensor([scale], device="cuda"), torch.tensor([zero], device="cuda")
            expected = torch.round(x / s + zp).clamp(0, 127).to(torch.int8)
            self.assertTrue(torch.equal(lut_kernel.quantize_u7(x, s, zp), expected))
            padded = F.pad(expected, (1, 1, 2, 2))
            self.assertTrue(torch.equal(lut_kernel.quantize_u7(x, s, zp, (2, 1)), padded))

    def test_transformer_fast_path_matches_gemm_int8(self):
        lut = TRUNCATED.cuda()
        q, _ = quantize(floats(2, 4, 50, 64), -1)
        keys, _ = quantize(floats(2, 1, 50, 64).transpose(-2, -1), -2)  # broadcast heads, strided
        x, _ = quantize(floats(77, 640), -1)
        w, _ = quantize(floats(640, 300), 0)
        for a, b in [(q, keys), (x, w)]:
            self.assertTrue(torch.equal(lut_matmul(a, b, lut, symmetric=True), lut_matmul(a, b, lut)))

    def test_table_cache_follows_in_place_lut_updates(self):
        t = lut_kernel.table(self.lut, lut_kernel.CONV)
        self.assertIs(lut_kernel.table(self.lut, lut_kernel.CONV), t)
        self.lut.copy_(TRUNCATED)
        fresh = lut_kernel.table(self.lut, lut_kernel.CONV)
        self.assertTrue(torch.equal(fresh, self.lut.view(256, 256)[:, 128:].short().reshape(-1)))

    def test_small_shared_memory_gpus_fall_back(self):
        t4 = SimpleNamespace(shared_memory_per_block_optin=64 * 1024)
        lut_kernel._fits.cache_clear()
        try:
            with mock.patch.object(torch.cuda, "get_device_properties", return_value=t4):
                self.assertIsNone(lut_kernel.table(self.lut.clone(), lut_kernel.LINEAR))
        finally:
            lut_kernel._fits.cache_clear()
        self.assertIsNotNone(lut_kernel.table(self.lut, lut_kernel.LINEAR))


if __name__ == "__main__":
    unittest.main()
