"""Pack native ComfyUI Qwen-Image-2.1 weights and Zen TE into one safetensors file.

The input diffusion and VAE weights must already be ComfyUI-format safetensors.
This script copies tensor payloads without materializing model weights in RAM.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARTS = ("diffusion", "vae", "student", "fusion")
ASSET_DIRS = ("text_encoder", "processor", "tokenizer")
FORMAT = "zen-image-edit-comfy-single-v1"


def read_header(path: Path) -> tuple[dict, int, int]:
    with path.open("rb") as stream:
        header_size_bytes = stream.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        if header_size > 100_000_000:
            raise ValueError(f"Unreasonably large safetensors header: {path}")
        header = json.loads(stream.read(header_size))
    tensors = [entry for name, entry in header.items() if name != "__metadata__"]
    if not tensors:
        raise ValueError(f"No tensors in {path}")
    data_size = max(entry["data_offsets"][1] for entry in tensors)
    expected_size = 8 + header_size + data_size
    actual_size = path.stat().st_size
    if expected_size != actual_size:
        raise ValueError(f"Incomplete or invalid safetensors file {path}: {actual_size} != {expected_size}")
    return header, 8 + header_size, data_size


def pack_assets(asset_dir: Path) -> str:
    files = {}
    for folder in ASSET_DIRS:
        directory = asset_dir / folder
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing model asset directory: {directory}")
        for file in sorted(directory.rglob("*")):
            if file.is_file():
                relative = file.relative_to(asset_dir).as_posix()
                files[relative] = base64.b64encode(file.read_bytes()).decode("ascii")
    for required in ("text_encoder/config.json", "processor/preprocessor_config.json",
                     "processor/tokenizer.json", "tokenizer/tokenizer.json"):
        if required not in files:
            raise FileNotFoundError(f"Missing model asset: {asset_dir / required}")
    blob = json.dumps(files, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.b64encode(zlib.compress(blob, level=9)).decode("ascii")


def convert(inputs: dict[str, Path], asset_dir: Path, output: Path) -> dict:
    headers = {}
    for part in PARTS:
        headers[part] = read_header(inputs[part])

    diffusion = headers["diffusion"][0]
    vae = headers["vae"][0]
    student = headers["student"][0]
    fusion = headers["fusion"][0]
    if not any(key.startswith("transformer_blocks.") for key in diffusion):
        raise ValueError("The diffusion input is not a native ComfyUI Qwen-Image-2.1 model")
    if "decoder.head.2.weight" not in vae or "encoder.conv1.weight" not in vae:
        raise ValueError("The VAE input is not a native ComfyUI Qwen-Image-2.1 VAE")
    if "model.language_model.embed_tokens.weight" not in student:
        raise ValueError("The student input is not the Qwen3.5 multimodal text encoder")
    if "attn.pos" not in fusion:
        raise ValueError("The fusion input does not contain the Zen position table")

    fusion_config = {key: value for key, value in fusion.get("__metadata__", {}).items()
                     if key in {"in_dim", "out_dim", "hidden", "proj_layers", "norm", "n_slices",
                                "student_layers", "attention", "attn_dim", "attn_heads",
                                "attn_max_len", "max_len", "mixer", "mixer_heads", "mixer_ffn", "drop_idx"}}
    metadata = {
        "format": "pt",
        "zen_format": FORMAT,
        "zen_assets_zlib_b64": pack_assets(asset_dir),
        "zen_fusion_config": json.dumps(fusion_config, sort_keys=True),
        "zen_source": json.dumps({
            "diffusion": "Comfy-Org/Qwen-Image-2.1@9a44dbdb47cefd046be9c0a13476192f34c8db8e",
            "vae": "Comfy-Org/Qwen-Image-2.1@9a44dbdb47cefd046be9c0a13476192f34c8db8e",
            "student": "AiArtLab/zen-image-edit@be4afb16f1f8d5032e7d77af4d5f63caa81ff25a",
            "fusion": "AiArtLab/zen-image-edit transformer shard",
        }, sort_keys=True),
    }

    output_header = {"__metadata__": metadata}
    position = 0
    for part in PARTS:
        header, _, data_size = headers[part]
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            key = f"zen.{part}.{name}"
            output_header[key] = {
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "data_offsets": [position + entry["data_offsets"][0],
                                 position + entry["data_offsets"][1]],
            }
        position += data_size

    encoded = json.dumps(output_header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    digests = {}
    try:
        with temporary.open("wb") as dest:
            dest.write(struct.pack("<Q", len(encoded)))
            dest.write(encoded)
            for part in PARTS:
                _, data_start, data_size = headers[part]
                sha = hashlib.sha256()
                with inputs[part].open("rb") as source:
                    source.seek(data_start)
                    remaining = data_size
                    while remaining:
                        chunk = source.read(min(16 * 1024 * 1024, remaining))
                        if not chunk:
                            raise EOFError(f"Source changed while copying: {inputs[part]}")
                        dest.write(chunk)
                        sha.update(chunk)
                        remaining -= len(chunk)
                digests[part] = sha.hexdigest()
            dest.flush()
            os.fsync(dest.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    read_header(output)
    return {"format": FORMAT, "file": str(output), "bytes": output.stat().st_size,
            "tensors": len(output_header) - 1, "payload_sha256": digests,
            "source_files": {part: str(inputs[part]) for part in PARTS}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for part in PARTS:
        parser.add_argument(f"--{part}", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    inputs = {part: getattr(arguments, part) for part in PARTS}
    manifest = convert(inputs, arguments.assets, arguments.output)
    print(json.dumps(manifest, indent=2))
    manifest_path = arguments.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
