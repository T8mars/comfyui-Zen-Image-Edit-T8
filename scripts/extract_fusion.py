"""Extract Zen's text-fusion tensors from the author's published transformer shard.

Only the adapter is retained; the Diffusers transformer weights are not used in
the converted ComfyUI checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

if __package__:
    from .convert import read_header
else:
    from convert import read_header


def extract(shard: Path, config_path: Path, output: Path) -> None:
    header, start, _ = read_header(shard)
    tensors = {name[len("text_fusion."):]: value for name, value in header.items()
               if name.startswith("text_fusion.")}
    if not tensors or "attn.pos" not in tensors:
        raise ValueError("No complete Zen text-fusion adapter in this shard")
    first = min(item["data_offsets"][0] for item in tensors.values())
    last = max(item["data_offsets"][1] for item in tensors.values())
    spans = sorted(item["data_offsets"] for item in tensors.values())
    if spans[0][0] != first or spans[-1][1] != last or any(
        a[1] != b[0] for a, b in zip(spans, spans[1:])
    ):
        raise ValueError("Zen fusion tensors are not contiguous in this shard")
    config = json.loads(config_path.read_text(encoding="utf-8"))["text_fusion_config"]
    meta = {key: ",".join(map(str, value)) if isinstance(value, list) else str(value)
            for key, value in config.items()}
    out_header = {"__metadata__": meta}
    for name, value in tensors.items():
        out_header[name] = {"dtype": value["dtype"], "shape": value["shape"],
                            "data_offsets": [offset - first for offset in value["data_offsets"]]}
    encoded = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    try:
        with shard.open("rb") as source, temporary.open("wb") as dest:
            dest.write(struct.pack("<Q", len(encoded)))
            dest.write(encoded)
            source.seek(start + first)
            remaining = last - first
            while remaining:
                chunk = source.read(min(16 * 1024 * 1024, remaining))
                if not chunk:
                    raise EOFError("Transformer shard changed while extracting fusion weights")
                dest.write(chunk)
                remaining -= len(chunk)
            dest.flush()
            os.fsync(dest.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    read_header(output)
    print(f"Extracted {len(tensors)} fusion tensors to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract(args.shard, args.config, args.output)
