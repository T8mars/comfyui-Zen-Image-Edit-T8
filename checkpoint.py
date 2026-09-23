"""Read the single-file native ComfyUI Zen Image Edit checkpoint."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import struct
import tempfile
import zlib
from pathlib import Path

import torch


FORMAT = "zen-image-edit-comfy-single-v1"
PREFIXES = {part: f"zen.{part}." for part in ("diffusion", "vae", "student", "fusion")}


def metadata(path: str | Path) -> dict[str, str]:
    with Path(path).open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        if header_size > 100_000_000:
            raise ValueError("Safetensors header is too large")
        result = json.loads(source.read(header_size)).get("__metadata__", {})
    if result.get("zen_format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} checkpoint")
    return result


def component(path: str | Path, part: str) -> dict:
    prefix = PREFIXES[part]
    dtypes = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32,
              "I64": torch.int64, "I32": torch.int32, "I16": torch.int16,
              "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}
    result = {}
    file_size = Path(path).stat().st_size
    with Path(path).open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        if header_size > 100_000_000:
            raise ValueError("Safetensors header is too large")
        header = json.loads(source.read(header_size))
        data_start = 8 + header_size
        keys = [key for key in header if key.startswith(prefix)]
        if not keys:
            raise ValueError(f"Checkpoint has no {part} weights: {path}")
        for key in keys:
            entry = header[key]
            start, end = entry["data_offsets"]
            if start < 0 or end < start or data_start + end > file_size:
                raise ValueError(f"Invalid {part} weight offsets: {key}")
            dtype = dtypes[entry["dtype"]]
            expected_bytes = torch.empty((), dtype=dtype).element_size()
            for dimension in entry["shape"]:
                expected_bytes *= dimension
            if end - start != expected_bytes:
                raise ValueError(f"Invalid {part} weight length: {key}")
            source.seek(data_start + start)
            data = bytearray(end - start)
            if source.readinto(data) != end - start:
                raise EOFError(f"Truncated {part} weight: {key}")
            tensor = torch.frombuffer(data, dtype=dtype).reshape(entry["shape"])
            result[key[len(prefix):]] = tensor
    return result


def copy_component_to_module(path: str | Path, part: str, module: torch.nn.Module) -> None:
    """Stream tensors into an allocated module without mapping the 17 GB file."""
    prefix = PREFIXES[part]
    parameters = module.state_dict(keep_vars=True)
    found = set()
    dtypes = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32,
              "I64": torch.int64, "I32": torch.int32, "I16": torch.int16,
              "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}
    with Path(path).open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        if header_size > 100_000_000:
            raise ValueError("Safetensors header is too large")
        header = json.loads(source.read(header_size))
        data_start = 8 + header_size
        keys = [key for key in header if key.startswith(prefix)]
        if not keys:
            raise ValueError(f"Checkpoint has no {part} weights: {path}")
        with torch.no_grad():
            for key in keys:
                name = key[len(prefix):]
                if name not in parameters:
                    if part == "student" and name.startswith("mtp."):
                        continue  # Training-only multi-token prediction head.
                    raise ValueError(f"Unexpected {part} weight: {name}")
                entry = header[key]
                target = parameters[name]
                if tuple(entry["shape"]) != tuple(target.shape):
                    raise ValueError(f"{part} weight shape mismatch for {name}: {entry['shape']} != {list(target.shape)}")
                dtype = dtypes[entry["dtype"]]
                element_size = torch.empty((), dtype=dtype).element_size()
                start, end = entry["data_offsets"]
                if end - start != target.numel() * element_size:
                    raise ValueError(f"Invalid byte length for {part} weight: {name}")
                source.seek(data_start + start)
                flat = target.view(-1)
                offset = 0
                while offset < target.numel():
                    count = min(target.numel() - offset, 8 * 1024 * 1024 // element_size)
                    chunk = source.read(count * element_size)
                    if len(chunk) != count * element_size:
                        raise EOFError(f"Truncated {part} weight: {name}")
                    values = torch.frombuffer(bytearray(chunk), dtype=dtype)
                    flat[offset:offset + count].copy_(values)
                    offset += count
                found.add(name)
    missing = parameters.keys() - found
    if part == "student" and "lm_head.weight" in missing:
        if module.lm_head.weight is module.model.language_model.embed_tokens.weight:
            missing.remove("lm_head.weight")
    if missing:
        raise ValueError(f"Missing {part} weights: {sorted(missing)[:8]}")


def assets_path(path: str | Path) -> Path:
    meta = metadata(path)
    encoded = meta.get("zen_assets_zlib_b64")
    if not encoded or len(encoded) > 100_000_000:
        raise ValueError("Checkpoint lacks valid embedded tokenizer and processor assets")
    compressed = base64.b64decode(encoded, validate=True)
    fingerprint = hashlib.sha256(compressed).hexdigest()
    cache_root = Path(tempfile.gettempdir()) / "zen-image-edit-comfy-assets"
    target = cache_root / fingerprint
    if (target / "complete.json").is_file():
        return target

    decompressor = zlib.decompressobj()
    blob = decompressor.decompress(compressed, 100_000_001)
    if len(blob) > 100_000_000 or decompressor.unconsumed_tail or not decompressor.eof:
        raise ValueError("Embedded model assets exceed the 100 MB limit")
    assets = json.loads(blob)
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="unpack-", dir=cache_root))
    try:
        for name, encoded_file in assets.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in (
                "text_encoder", "processor", "tokenizer"
            ):
                raise ValueError(f"Unsafe embedded asset path: {name}")
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(base64.b64decode(encoded_file, validate=True))
        (temporary / "complete.json").write_text(
            json.dumps({"format": FORMAT, "fingerprint": fingerprint}), encoding="utf-8"
        )
        try:
            os.replace(temporary, target)
        except FileExistsError:
            if not (target / "complete.json").is_file():
                raise
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
