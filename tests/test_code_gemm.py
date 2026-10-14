import unittest

import torch

from approxtorch.backend import lut_kernel, ops
from approxtorch.transformer import LutLinear, approx_attention, approx_matmul, lut_matmul

CUDA = torch.cuda.is_available() and lut_kernel._extension() is not None
# (M, K, N): K tails and K = 1. Batch 1 runs the 32x32 config (too few tiles); batch 64 the
# 16x256 (M <= 16, or a table in global memory), 64x64 (N <= 64) and 32x128 configs.
SHAPES = [(5, 37, 300), (70, 64, 33), (130, 100, 200), (17, 1, 65), (33, 31, 129)]


def mantissa_lut(f):
    """Exact (truncating) mantissa multiplier: entry = norm << f | fraction."""
    one = 1 << f
    p = (one + torch.arange(one))[:, None] * (one + torch.arange(one))[None, :]
    norm = (p >= 2 * one * one).long()
    return (norm << f | (p >> f + norm) - one).to(torch.uint32)


def exact_lut(kind):
    if kind in ("bf16", "fp16"):
        return mantissa_lut(7 if kind == "bf16" else 10)
    if kind in lut_kernel.FP_FORMATS:
        return lut_kernel.fp_table(mantissa_lut(lut_kernel.FP_FORMATS[kind][1]), kind)
    v = {"int8": torch.arange(-128, 128), "int4": torch.arange(-8, 8)}.get(kind, torch.arange(256))
    return (v[:, None] * v[None, :]).int().reshape(-1)


def random_lut(kind, device):
    if kind in ("bf16", "fp16"):
        side, top = (128, 256) if kind == "bf16" else (1024, 2048)
        return torch.randint(0, top, (side, side), device=device).to(torch.uint32)
    if kind in ("fp8_e4m3", "fp8_e5m2"):
        return torch.randn(256, 256, device=device)
    if kind == "fp4":
        return torch.randn(16, 16, device=device)
    n = 256 if kind == "int4" else 65536
    low, high = (-32768, 32768) if kind in ("int8", "int4") else (0, 65536)
    return torch.randint(low, high, (n,), dtype=torch.int32, device=device)


def operands(kind, *shape, device="cuda"):
    if kind in ("bf16", "fp16"):
        return torch.randn(*shape, device=device).to(torch.bfloat16 if kind == "bf16" else torch.float16)
    if kind.startswith("fp8"):  # every code, NaN included: the RTL ignores special values
        dtype = torch.float8_e4m3fn if kind == "fp8_e4m3" else torch.float8_e5m2
        return torch.randint(0, 256, shape, dtype=torch.uint8, device=device).view(dtype)
    low, high, dtype = {"int8": (-128, 128, torch.int8), "int4": (-8, 8, torch.int8),
                        "uint8": (0, 256, torch.uint8), "fp4": (0, 16, torch.uint8),
                        "int16": (-32768, 32768, torch.int16)}[kind]
    return torch.randint(low, high, shape, device=device).to(dtype)


def bits(x):
    if not x.is_floating_point():
        return x
    x = x.masked_fill(x.isnan(), 0)  # any NaN payload is the same NaN
    return x.view({2: torch.int16, 4: torch.int32}[x.element_size()])


@unittest.skipUnless(CUDA, "needs CUDA")
class CodeGemmTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def same(self, got, want, msg=None):
        self.assertEqual((got.dtype, got.shape), (want.dtype, want.shape), msg)
        self.assertTrue(torch.equal(got.isnan(), want.isnan()) if got.is_floating_point() else True, msg)
        self.assertTrue(torch.equal(bits(got), bits(want)), msg)

    def test_every_kind_matches_reference(self):
        for kind in lut_kernel.KINDS:
            luts = [random_lut(kind, "cuda"), exact_lut(kind).cuda()]
            for lut in luts:
                for (m, k, n), z in ((s, z) for s in SHAPES for z in (1, 64)):
                    a, b = operands(kind, z, m, k), operands(kind, k, n)
                    want = lut_kernel.reference(a, b, lut, kind)
                    for smem in (True, False):
                        self.same(lut_kernel.matmul(a, b, lut, kind, smem), want, (kind, z, m, k, n, smem))

    def test_strided_and_broadcast_operands(self):
        for kind in lut_kernel.KINDS:
            lut = random_lut(kind, "cuda")
            a = operands(kind, 3, 80, 35)[:, ::2]  # row stride 70
            b = operands(kind, 3, 70, 50)[:, ::2]  # k stride 100
            c = operands(kind, 50, 35).t()  # n stride 35, broadcast over the batch
            d = operands(kind, 4, 1, 35, 20)  # broadcast batch dims -> [4, 3, 40, 20]
            for x, y in ((a, b), (a, c), (a, d)):
                want = lut_kernel.reference(x.contiguous(), y.contiguous(), lut, kind)
                self.same(lut_kernel.matmul(x, y, lut, kind), want, (kind, y.shape))

    def test_bgemm_lut_row_is_w_or_x(self):
        for kind in lut_kernel.KINDS:
            lut = random_lut(kind, "cuda")
            x = operands(kind, 3, 45, 70)
            for w in (operands(kind, 20, 45), operands(kind, 45, 20).t()):
                want_w = lut_kernel.reference(w.contiguous(), x, lut, kind)
                want_x = lut_kernel.reference(x.transpose(1, 2), w.t(), lut, kind).transpose(1, 2).contiguous()
                for smem in (True, False):
                    self.same(lut_kernel.bgemm(x, w, lut, kind, smem), want_w, (kind, smem))
                    self.same(lut_kernel.bgemm(x, w, lut, kind, smem, x_row=True), want_x, (kind, smem))

    def test_int8_uint8_match_upstream_integer_gemms(self):
        for kind in ("int8", "uint8"):
            lut = random_lut(kind, "cuda")
            a, b, x = operands(kind, 70, 100), operands(kind, 100, 33), operands(kind, 3, 100, 50)
            self.same(lut_kernel.matmul(a, b, lut, kind), getattr(ops, f"gemm_{kind}")(a, b, lut), kind)
            self.same(lut_kernel.bgemm(x, a, lut, kind), getattr(ops, f"bgemm_{kind}")(x, a, lut), kind)

    def test_routed_bf16_fp16_ops_match_upstream_naive(self):
        for kind, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
            gemm, naive = getattr(ops, f"gemm_{kind}"), getattr(ops, f"gemm_{kind}_naive")
            bgemm, bnaive = getattr(ops, f"bgemm_{kind}"), getattr(ops, f"bgemm_{kind}_naive")
            lut = random_lut(kind, "cuda")
            for m, k, n in SHAPES:
                a, b = operands(kind, m, k), operands(kind, k, n)
                self.same(gemm(a, b, lut), naive(a, b, lut), (kind, m, k, n))
            # Raw bit patterns: exponent wrap, inf/NaN codes and +-0 operands.
            a = torch.randint(-32768, 32768, (40, 64), dtype=torch.int16, device="cuda").view(dtype)
            b = torch.randint(-32768, 32768, (64, 50), dtype=torch.int16, device="cuda").view(dtype)
            a[0, :5], b[:5, 1] = 0.0, -0.0
            self.same(gemm(a, b, lut), naive(a, b, lut), kind)
            x, w = operands(kind, 3, 37, 50), operands(kind, 20, 37)
            self.same(bgemm(x, w, lut), bnaive(x, w, lut), kind)

    def test_kernel_kind_and_table_cache(self):
        lut = random_lut("int8", "cuda")
        self.assertEqual(lut_kernel._code_table(lut, "int8", False)[0], lut_kernel._I16)
        self.assertEqual(lut_kernel._code_table(lut.abs() % 32768 + 32768, "int8", False)[0], lut_kernel._U16)
        self.assertEqual(lut_kernel._code_table(exact_lut("fp8_e4m3").cuda(), "fp8_e4m3", False)[0],
                         lut_kernel._F32SYM)
        self.assertEqual(lut_kernel._code_table(random_lut("fp8_e4m3", "cuda"), "fp8_e4m3", False)[0],
                         lut_kernel._F32)
        prep = lut_kernel._code_table(lut, "int8", False)
        self.assertIs(lut_kernel._code_table(lut, "int8", False), prep)
        lut.mul_(-1)  # in-place update invalidates
        self.assertIsNot(lut_kernel._code_table(lut, "int8", False), prep)
        a, b = operands("int8", 20, 30), operands("int8", 30, 40)
        self.same(lut_kernel.matmul(a, b, lut, "int8"), lut_kernel.reference(a, b, lut, "int8"))
        wide = lut.clone()
        wide[0] = 70000  # fits neither int16 nor uint16: torch reference
        self.assertIsNone(lut_kernel._code_table(wide, "int8", False))
        self.same(lut_kernel.matmul(a, b, wide, "int8"), lut_kernel.reference(a, b, wide, "int8"))

    def test_empty_dims_unaligned_table_and_inference_mode_lut(self):
        for kind in lut_kernel.KINDS:
            lut = random_lut(kind, "cuda")
            for m, k, n in ((0, 5, 3), (4, 0, 3), (4, 5, 0)):
                a, b = operands(kind, 2, m, k), operands(kind, k, n)
                self.same(lut_kernel.matmul(a, b, lut, kind), lut_kernel.reference(a, b, lut, kind), (kind, m, k, n))
        a, b = operands("fp4", 40, 30), operands("fp4", 30, 50)
        lut = torch.randn(257, device="cuda")[1:].reshape(16, 16)  # data_ptr % 16 == 4
        self.same(lut_kernel.matmul(a, b, lut, "fp4"), lut_kernel.reference(a, b, lut, "fp4"))
        with torch.inference_mode():  # no version counter: tables are rebuilt, never stale
            lut = random_lut("int8", "cuda")
            a, b = operands("int8", 20, 30), operands("int8", 30, 40)
            first = lut_kernel.matmul(a, b, lut, "int8")
            lut.mul_(-1)
            self.same(lut_kernel.matmul(a, b, lut, "int8"), -first)
            x = torch.randn(5, 30, device="cuda")
            approx_matmul(x, x.t(), exact_lut("int8").cuda())  # SIGNMAG table() path
        for m, k, n in ((0, 5, 3), (4, 0, 3), (4, 5, 0)):  # SIGNMAG kernel (lut_matmul's int8 path)
            got = lut_matmul(operands("int8", 2, m, k) // 2, operands("int8", k, n) // 2, exact_lut("int8").cuda(), True)
            self.same(got, torch.zeros(2, m, n, device="cuda"), (m, k, n))

    def test_bad_inputs_raise(self):
        lut = random_lut("int16", "cuda")
        with self.assertRaises(RuntimeError):  # int16 sums are exact only up to K = 16384
            lut_kernel.matmul(operands("int16", 2, 16385), operands("int16", 16385, 2), lut, "int16")
        with self.assertRaises(ValueError):
            lut_kernel.matmul(operands("int8", 2, 3), operands("int8", 3, 2), lut, "int9")


class FormatTests(unittest.TestCase):
    def test_exact_luts_multiply_exactly(self):
        a = torch.randint(-127, 128, (3, 20, 30))
        b = torch.randint(-127, 128, (30, 10))
        self.assertTrue(torch.equal(lut_kernel.reference(a.to(torch.int8), b.to(torch.int8), exact_lut("int8"),
                                                         "int8").long(), a @ b))
        a = torch.randint(-32768, 32768, (20, 30))
        b = torch.randint(-32768, 32768, (30, 10))
        got = lut_kernel.reference(a.to(torch.int16), b.to(torch.int16), exact_lut("int16"), "int16")
        self.assertTrue(torch.equal(got, a @ b))

    def test_fp_table_is_the_truncated_exact_product(self):
        grid = torch.tensor(lut_kernel._FP4_GRID, dtype=torch.float64)
        for fmt, (e, f, _) in lut_kernel.FP_FORMATS.items():
            codes = torch.arange(1 << 1 + e + f)
            if fmt == "fp4":
                value = torch.where(codes >= 8, -grid[codes & 7], grid[codes & 7])
            else:
                dtype = torch.float8_e4m3fn if fmt == "fp8_e4m3" else torch.float8_e5m2
                value = codes.to(torch.uint8).view(dtype).double()  # NaN/inf codes: skipped below
            want = value[:, None] * value[None, :]
            mantissa, exp = torch.frexp(want)  # no exponent limits: subnormals, under/overflow
            want = torch.trunc(mantissa * 2.0 ** (f + 1)) * 2.0 ** (exp - f - 1)
            finite = want.isfinite()
            self.assertTrue(torch.equal(exact_lut(fmt).double()[finite], want[finite]), fmt)

    def test_fp4_table_examples(self):
        table = exact_lut("fp4")
        for x, y in [(1.0, 6.0), (1.5, 2.0), (1.5, 4.0), (2.0, 3.0), (-2.0, 2.0), (0.0, -3.0)]:
            got = table[lut_kernel.fp4_encode(torch.tensor(x)).long(), lut_kernel.fp4_encode(torch.tensor(y)).long()]
            self.assertEqual(got.item(), x * y, (x, y))
        self.assertEqual(table[3, 3].item(), 2.0)  # 1.5 * 1.5 = 2.25 truncates to 2
        self.assertEqual(table[11, 3].item(), -2.0)

    def test_fp4_encode_rounds_to_nearest_even_and_saturates(self):
        x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 0.3, 7.0, 1e9, -0.0, -1.6, -5.1])
        want = [0, 2, 2, 4, 4, 6, 6, 1, 7, 7, 8, 8 + 3, 8 + 7]
        self.assertEqual(lut_kernel.fp4_encode(x).tolist(), want)


class TransformerKindTests(unittest.TestCase):
    TOLERANCE = {"int4": 0.4, "int8": 0.025, "int16": 1e-4, "fp8_e4m3": 0.16, "fp8_e5m2": 0.25,
                 "fp4": 0.4, "bf16": 0.015, "fp16": 0.0015}  # ~2x the measured error

    def test_linear_and_attention_track_float_for_every_kind(self):
        device = "cuda" if CUDA else "cpu"
        torch.manual_seed(0)
        linear = torch.nn.Linear(64, 48).to(device)
        x = torch.randn(2, 9, 64, device=device)
        q, k, v = torch.randn(3, 2, 4, 17, 32, device=device).unbind(0)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        for kind, tol in self.TOLERANCE.items():
            lut = exact_lut(kind).to(device)
            layer = LutLinear(linear, lut, kind)
            self.assertLess(rel_err(layer(x), linear(x)), tol, kind)
            self.assertLess(rel_err(layer.half()(x.half()).float(), linear(x)), max(tol, 2e-3), kind)
            self.assertLess(rel_err(approx_attention(q, k, v, lut, kind=kind), expected), tol, kind)

    def test_long_same_sign_sums_stay_finite(self):
        device = "cuda" if CUDA else "cpu"
        a, b = torch.ones(3, 512, device=device), torch.ones(512, 2, device=device)
        for kind, tol in self.TOLERANCE.items():  # fp16: |sum| <= K * 4 * scaled max^2 must fit
            self.assertLess(rel_err(approx_matmul(a, b, exact_lut(kind).to(device), kind), a @ b), tol, kind)

    def test_lutlinear_loads_float_scale_checkpoints(self):
        linear = torch.nn.Linear(16, 8)
        layer = LutLinear(linear, exact_lut("int8"))
        state = layer.state_dict()
        state["weight_scale"] = state["weight_scale"].view(torch.float32).clone()  # AxTorch 2.0 format
        fresh = LutLinear(torch.nn.Linear(16, 8), exact_lut("int8"))
        fresh.load_state_dict(state)
        x = torch.randn(4, 16)
        self.assertTrue(torch.equal(fresh(x), layer(x)))


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


if __name__ == "__main__":
    unittest.main()
