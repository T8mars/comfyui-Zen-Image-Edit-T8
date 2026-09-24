"""Convert Viggle v0.2.1 for ComfyUI's built-in LoRA loaders.

Fuses the independent gate/up adapters into one block-diagonal adapter for
ComfyUI's gate_up layer. No base weights are included or modified. Use the
built-in LoraLoaderBypassModelOnly with the BF16 Qwen-Image-2.1 base.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


REVISION = "b667ee50696e8f4ccd161e02f8af0e43bb5665ed"
SOURCE = "Viggle/Qwen-Image-2.1-viggle-turbo"
SOURCE_HASHES = {
    256: "2a0148f5c73abbed5f97da5ea356e439318aadb281d01fce4af39cdf43728803",
    128: "bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_targets() -> set[str]:
    targets = {f"transformer.transformer_blocks.{i}.{part}" for i in range(32)
               for part in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                            "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out")}
    targets.update({"transformer.time_text_embed.timestep_embedder.linear_1",
                    "transformer.time_text_embed.timestep_embedder.linear_2",
                    "transformer.modulation.1"})
    return targets


def fuse_gate_up(gate_a, gate_b, up_a, up_b):
    """[Bg Ag; Bu Au] = block_diag(Bg, Bu) @ [Ag; Au], without rounding."""
    if gate_a.shape != up_a.shape or gate_b.shape != up_b.shape:
        raise ValueError("Gate and up adapter shapes differ")
    return torch.cat((gate_a, up_a), dim=0), torch.block_diag(gate_b, up_b)


def convert_tensors(weights: dict[str, torch.Tensor], rank: int) -> dict[str, torch.Tensor]:
    targets = source_targets()
    expected = {f"{name}.lora_{part}.weight" for name in targets for part in ("A", "B")}
    if weights.keys() != expected:
        missing, extra = sorted(expected - weights.keys()), sorted(weights.keys() - expected)
        raise ValueError(f"Incomplete Viggle adapter: missing={missing[:4]}, extra={extra[:4]}")
    for name in targets:
        a, b = (weights[f"{name}.lora_{part}.weight"] for part in ("A", "B"))
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError(f"Invalid rank or dimensions: {name}")
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            raise ValueError(f"Expected BF16 adapter tensors: {name}")

    output = {}
    for name in sorted(targets):
        if name.endswith(".img_mlp.proj"):
            continue
        a, b = (weights[f"{name}.lora_{part}.weight"] for part in ("A", "B"))
        target = name.replace("transformer.", "diffusion_model.", 1)
        if name.endswith(".img_mlp.gate_layer"):
            up = name.removesuffix("gate_layer") + "proj"
            a, b = fuse_gate_up(a, b, weights[up + ".lora_A.weight"], weights[up + ".lora_B.weight"])
            target = target.removesuffix("gate_layer") + "gate_up"
        output[target + ".lora_down.weight"] = a.contiguous()
        output[target + ".lora_up.weight"] = b.contiguous()
        # Fused gate/up has twice the rank. Equal alpha keeps the original scale of 1.
        output[target + ".alpha"] = torch.tensor(float(a.shape[0]), dtype=torch.float32)
    return output


def convert(source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    source_hash = file_sha256(source)
    ranks = [rank for rank, expected in SOURCE_HASHES.items() if source_hash == expected]
    if not ranks:
        raise ValueError("Source is not an official Viggle v0.2.1 r128/r256 file (SHA256 mismatch)")
    rank = ranks[0]
    with safe_open(source, framework="pt", device="cpu") as reader:
        config = json.loads((reader.metadata() or {}).get("lora_adapter_metadata", "{}"))
        if config.get("transformer.r") != rank or config.get("transformer.lora_alpha") != rank:
            raise ValueError("Unexpected Viggle rank/alpha metadata")
        weights = {key: reader.get_tensor(key) for key in reader.keys()}
    converted = convert_tensors(weights, rank)
    metadata = {
        "format": "pt", "t8_format": "viggle-comfy-lora-v1", "viggle_version": "0.2.1",
        "source_repo": SOURCE, "source_revision": REVISION, "source_sha256": source_hash,
        "source_rank": str(rank), "base_model": "Qwen/Qwen-Image-2.1",
        "recommended_loader": "LoraLoaderBypassModelOnly", "base_precision": "BF16",
        "modification": "T8star: native keys and block-diagonal gate/up fusion; no base weights; no rank truncation.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    try:
        save_file(converted, temporary, metadata=metadata)
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    manifest = {**metadata, "file": output.name, "bytes": output.stat().st_size,
                "sha256": file_sha256(output), "source_modules": len(source_targets()),
                "native_modules": len(converted) // 3, "tensors": len(converted)}
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
