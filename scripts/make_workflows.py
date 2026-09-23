"""Write editable ComfyUI UI workflows and matching API prompts."""

from __future__ import annotations

import json
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workflows"
CHECKPOINT = "zen_image_edit_qwen21_single.safetensors"


def api(edit: bool) -> dict:
    encoder = {
        "clip": ["1", 1],
        "prompt": "turn the red square into a yellow circle" if edit else "a red fox in a snowy forest",
        "negative_prompt": "",
        "resolution": 512,
    }
    if edit:
        encoder.update({"vae": ["1", 2], "images": {"image_1": ["6", 0]}})
    result = {
        "1": {"class_type": "ZenImageEditCheckpointLoader",
              "inputs": {"checkpoint": CHECKPOINT, "te_precision": "auto", "shift": 5.0}},
        "2": {"class_type": "TextEncodeQwenImage21", "inputs": encoder},
        "3": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["2", 1],
            "latent_image": ["2", 2], "seed": 42, "steps": 20, "cfg": 4.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
        }},
        "4": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["1", 2]}},
        "5": {"class_type": "SaveImage", "inputs": {
            "images": ["4", 0], "filename_prefix": "ZenImageEdit",
        }},
    }
    if edit:
        result["6"] = {"class_type": "LoadImage", "inputs": {"image": "reference.png"}}
    return result


def ui(edit: bool) -> dict:
    nodes = []
    links = []

    def node(identifier, kind, x, y, width, height, inputs, outputs, widgets):
        nodes.append({
            "id": identifier, "type": kind, "pos": [x, y], "size": [width, height],
            "flags": {}, "order": identifier - 1, "mode": 0,
            "inputs": [{"name": name, "type": typ,
                        **({"widget": {"name": name}} if widget else {}), "link": None}
                       for name, typ, widget in inputs],
            "outputs": [{"name": name, "type": typ, "links": []}
                        for name, typ in outputs],
            "properties": {"Node name for S&R": kind}, "widgets_values": widgets,
        })

    def link(source, source_slot, target, target_slot, typ):
        lid = len(links) + 1
        links.append([lid, source, source_slot, target, target_slot, typ])
        nodes[source - 1]["outputs"][source_slot]["links"].append(lid)
        nodes[target - 1]["inputs"][target_slot]["link"] = lid

    node(1, "ZenImageEditCheckpointLoader", 0, 0, 340, 170,
         [("checkpoint", "COMBO", True), ("te_precision", "COMBO", True),
          ("shift", "FLOAT", True)],
         [("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")],
         [CHECKPOINT, "auto", 5.0])
    encoder_inputs = [("clip", "CLIP", False), ("prompt", "STRING", True),
                      ("negative_prompt", "STRING", True), ("vae", "VAE", False),
                      ("resolution", "INT", True)]
    if edit:
        encoder_inputs += [("canvas_mode", "COMBO", True), ("width", "INT", True),
                           ("height", "INT", True)]
        encoder_inputs.append(("image_1", "IMAGE", False))
    node(2, "ZenImageEditEncodeAdvanced" if edit else "TextEncodeQwenImage21",
         410, 0, 420, 320,
         encoder_inputs,
         [("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
          ("latent", "LATENT")],
         ["turn the red square into a yellow circle" if edit else "a red fox in a snowy forest",
          "", 512] + (["auto", 512, 512] if edit else []))
    node(3, "KSampler", 900, 0, 320, 450,
         [("model", "MODEL", False), ("positive", "CONDITIONING", False),
          ("negative", "CONDITIONING", False), ("latent_image", "LATENT", False),
          ("seed", "INT", True), ("steps", "INT", True), ("cfg", "FLOAT", True),
          ("sampler_name", "COMBO", True), ("scheduler", "COMBO", True),
          ("denoise", "FLOAT", True)],
         [("LATENT", "LATENT")], [42, "fixed", 20, 4.0, "euler", "simple", 1.0])
    node(4, "VAEDecode", 1280, 0, 270, 100,
         [("samples", "LATENT", False), ("vae", "VAE", False)],
         [("IMAGE", "IMAGE")], [])
    node(5, "SaveImage", 1600, 0, 340, 300,
         [("images", "IMAGE", False), ("filename_prefix", "STRING", True)],
         [], ["ZenImageEdit"])
    if edit:
        node(6, "LoadImage", 0, 260, 340, 350,
             [("image", "COMBO", True), ("upload", "IMAGEUPLOAD", True)],
             [("IMAGE", "IMAGE"), ("MASK", "MASK")],
             ["reference.png", "image"])
    link(1, 1, 2, 0, "CLIP")
    link(1, 0, 3, 0, "MODEL")
    link(2, 0, 3, 1, "CONDITIONING")
    link(2, 1, 3, 2, "CONDITIONING")
    link(2, 2, 3, 3, "LATENT")
    link(3, 0, 4, 0, "LATENT")
    link(1, 2, 4, 1, "VAE")
    link(4, 0, 5, 0, "IMAGE")
    if edit:
        link(1, 2, 2, 3, "VAE")
        link(6, 0, 2, 8, "IMAGE")
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"zen-image-edit:{'edit-v2' if edit else 'text'}")),
        "revision": 0,
        "last_node_id": 6 if edit else 5, "last_link_id": len(links),
        "nodes": nodes, "links": links, "groups": [], "config": {},
        "definitions": {}, "extra": {"ue_links": []}, "version": 0.4,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, edit in (("text_to_image", False), ("image_edit", True)):
        for kind, data in (("api", api(edit)), ("ui", ui(edit))):
            (OUT / f"{name}_{kind}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
