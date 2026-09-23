import base64
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from scripts.convert import convert, read_header


def safetensors(path: Path, name: str, payload: bytes, metadata=None):
    header = {"__metadata__": metadata or {},
              name: {"dtype": "U8", "shape": [len(payload)], "data_offsets": [0, len(payload)]}}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class ConvertTest(unittest.TestCase):
    def test_single_file_keeps_native_tensors_and_embeds_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = {}
            names = {"diffusion": "transformer_blocks.0.weight",
                     "vae": "decoder.head.2.weight",
                     "student": "model.language_model.embed_tokens.weight",
                     "fusion": "attn.pos"}
            for part, name in names.items():
                path = root / f"{part}.safetensors"
                safetensors(path, name, part.encode(), {"student_layers": "4,8,12,16,20,24"})
                inputs[part] = path
            # The VAE requires an encoder marker too.
            vae = inputs["vae"]
            header, start, _ = read_header(vae)
            payload = vae.read_bytes()[start:]
            header["encoder.conv1.weight"] = {"dtype": "U8", "shape": [1],
                                               "data_offsets": [len(payload), len(payload) + 1]}
            encoded = json.dumps(header, separators=(",", ":")).encode()
            encoded += b" " * (-len(encoded) % 8)
            vae.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload + b"v")

            assets = root / "assets"
            for item in ("text_encoder/config.json", "processor/preprocessor_config.json",
                         "processor/tokenizer.json", "tokenizer/tokenizer.json"):
                destination = assets / item
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text("{}", encoding="utf-8")
            output = root / "zen.safetensors"
            manifest = convert(inputs, assets, output)
            packed, data_start, _ = read_header(output)
            self.assertEqual(manifest["tensors"], 5)
            with output.open("rb") as stream:
                for part, name in names.items():
                    first, last = packed[f"zen.{part}.{name}"]["data_offsets"]
                    stream.seek(data_start + first)
                    self.assertEqual(stream.read(last - first), part.encode())
            raw_assets = zlib.decompress(base64.b64decode(
                packed["__metadata__"]["zen_assets_zlib_b64"]))
            self.assertEqual(len(json.loads(raw_assets)), 4)

    def test_rejects_truncated_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.safetensors"
            safetensors(path, "weight", b"12345678")
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                read_header(path)


if __name__ == "__main__":
    unittest.main()
