import unittest

import torch

from scripts.convert_viggle_lora import convert_tensors, source_targets


def fixture(rank=2):
    generator = torch.Generator().manual_seed(42)
    return {f"{name}.lora_{part}.weight": torch.randn(
                (rank, 4) if part == "A" else (6, rank), generator=generator).bfloat16()
            for name in sorted(source_targets()) for part in ("A", "B")}


class ViggleConversionTest(unittest.TestCase):
    def test_every_source_update_is_preserved_and_fused_scale_is_one(self):
        original = fixture()
        converted = convert_tensors(original, 2)
        self.assertEqual(len(converted), 195 * 3)
        for name in source_targets():
            native = name.replace("transformer.", "diffusion_model.", 1)
            if name.endswith((".img_mlp.gate_layer", ".img_mlp.proj")):
                native = native.rsplit(".", 1)[0] + ".gate_up"
            a, b = converted[native + ".lora_down.weight"], converted[native + ".lora_up.weight"]
            update = b.double() @ a.double()
            if name.endswith(".img_mlp.gate_layer"):
                update = update[:6]
            elif name.endswith(".img_mlp.proj"):
                update = update[6:]
            expected = original[name + ".lora_B.weight"].double() @ original[name + ".lora_A.weight"].double()
            torch.testing.assert_close(update, expected, rtol=0, atol=0)
            self.assertEqual(converted[native + ".alpha"].item(), a.shape[0])

    def test_rejects_missing_extra_and_wrong_rank(self):
        weights = fixture()
        missing = weights.copy()
        missing.pop(next(iter(missing)))
        extra = {**weights, "unknown": torch.zeros(1)}
        for broken in (missing, extra):
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                convert_tensors(broken, 2)
        with self.assertRaisesRegex(ValueError, "rank"):
            convert_tensors(weights, 4)


if __name__ == "__main__":
    unittest.main()
