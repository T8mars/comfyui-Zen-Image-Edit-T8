"""Qwen3.5 student and Zen fusion exposed through ComfyUI's CLIP contract."""

from __future__ import annotations

import json
import re

import torch

from .checkpoint import assets_path, copy_component_to_module, metadata
from .fusion_lib import build_fusion


VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
SYSTEM_PROMPT = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
T2I_TEMPLATE = SYSTEM_PROMPT + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


def fusion_config(meta: dict[str, str]) -> dict:
    raw = json.loads(meta["zen_fusion_config"])
    config = {}
    for key, value in raw.items():
        if key == "student_layers":
            config[key] = [int(item) for item in str(value).strip("[]").replace(" ", "").split(",")]
        elif key == "norm":
            config[key] = str(value)
        else:
            config[key] = int(value)
    if "attn_max_len" in config:
        config["max_len"] = config.pop("attn_max_len")
    config.setdefault("max_len", 2304)
    config.setdefault("mixer_ffn", 2)
    config.setdefault("drop_idx", 14)
    required = {"in_dim", "out_dim", "hidden", "proj_layers", "norm", "n_slices", "student_layers"}
    if required - config.keys():
        raise ValueError(f"Zen fusion checkpoint lacks configuration: {sorted(required - config.keys())}")
    if config["in_dim"] != 6144 or config["out_dim"] != 4096:
        raise ValueError("This Zen adapter is incompatible with Qwen-Image-2.1")
    if len(config["student_layers"]) != config["n_slices"]:
        raise ValueError("Zen adapter student layer count does not match its fusion input")
    return config


class ZenTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        del embedding_directory, tokenizer_data

    def tokenize_with_weights(self, text, return_word_ids=False, images=None,
                              keep_vision=False, prevent_empty_text=False, **kwargs):
        del return_word_ids, prevent_empty_text, kwargs
        if re.search(r"\([^()]+:[+-]?\d+(?:\.\d+)?\)", text):
            raise ValueError("Zen Image Edit does not support Comfy prompt weight syntax")
        return {"text": text, "images": list(images or []), "keep_vision": keep_vision}

    def state_dict(self):
        return {}


class ZenClipModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None, model_options=None, checkpoint_path=None):
        super().__init__()
        del model_options
        from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration

        if checkpoint_path is None:
            raise ValueError("Zen checkpoint path is required")
        self.checkpoint_path = str(checkpoint_path)
        self.assets = assets_path(self.checkpoint_path)
        self.config = fusion_config(metadata(self.checkpoint_path))
        self.dtype = dtype or torch.bfloat16
        self.dtypes = [self.dtype]
        self.clip_options = {}

        student_config = AutoConfig.from_pretrained(str(self.assets / "text_encoder"), local_files_only=True)
        with torch.device("meta"):
            self.student = Qwen3_5ForConditionalGeneration(student_config)
        self.student.to_empty(device=device)
        self.student.to(dtype=self.dtype)
        self.student.lm_head.weight = self.student.model.language_model.embed_tokens.weight
        self.student.eval()
        self.fusion = build_fusion(**self.config).to(device=device, dtype=self.dtype).eval()
        self.processor = AutoProcessor.from_pretrained(str(self.assets / "processor"), local_files_only=True)
        self.layers = self.config["student_layers"]
        self.drop_idx = self.config["drop_idx"]
        self.max_len = self.config["max_len"]
        self.im_start_id = int(self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>"))
        self.image_pad_id = int(self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))

    def load_checkpoint_weights(self):
        for part, module in (("student", self.student), ("fusion", self.fusion)):
            copy_component_to_module(self.checkpoint_path, part, module)
        self.student.eval()
        self.fusion.eval()

    def load_sd(self, state):
        return self.load_state_dict(state, strict=False)

    def reset_clip_options(self):
        self.clip_options = {}

    def set_clip_options(self, options):
        self.clip_options.update(options)

    def memory_estimation_function(self, tokens, device=None):
        del device
        images = tokens.get("images", [])
        pixel_count = sum(int(image.shape[1] * image.shape[2]) for image in images)
        return 512 * 1024 * 1024 + pixel_count * 1024

    @staticmethod
    def _pil(image):
        from PIL import Image

        if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
            raise ValueError("Each Zen reference must be one RGB image [1,H,W,3]")
        pixels = (image[0].detach().cpu().float().clamp(0, 1).numpy() * 255.0).round().astype("uint8")
        return Image.fromarray(pixels)

    def encode_token_weights(self, token_weight_pairs):
        if self.clip_options.get("layer") is not None:
            raise ValueError("Zen Image Edit requires its fixed six student layers; CLIP skip is unsupported")
        text = token_weight_pairs["text"] or " "
        images = token_weight_pairs.get("images", [])
        if images and token_weight_pairs.get("keep_vision", False):
            raise ValueError("Connect the Qwen-Image-2.1 VAE when using reference images")
        refs = " ".join(f"<image{i + 1}>{VISION_BLOCK}" for i in range(len(images)))
        template = T2I_TEMPLATE.replace("{}", refs + "{}", 1) if refs else T2I_TEMPLATE
        prompt = template.format(text)
        image_pil = [self._pil(image) for image in images]
        kwargs = {"text": [prompt], "padding": True, "padding_side": "right", "return_tensors": "pt"}
        if image_pil:
            kwargs["images"] = image_pil
        inputs = self.processor(**kwargs)
        if inputs.input_ids.shape[1] > self.max_len:
            raise ValueError(f"Zen condition has {inputs.input_ids.shape[1]} tokens; adapter capacity is {self.max_len}. Reduce reference resolution or prompt length.")

        if image_pil:
            grid = inputs.image_grid_thw.tolist()
            expected = [[1, int(image.height / 16), int(image.width / 16)] for image in image_pil]
            if grid != expected:
                raise ValueError(f"Zen processor resized reference images: grid={grid}, expected={expected}")

        ids = inputs.input_ids[0].tolist()
        starts = [index for index, token in enumerate(ids) if token == self.im_start_id]
        if len(starts) < 2 or starts[1] != self.drop_idx:
            raise ValueError(f"Zen tokenizer system prefix changed: {starts[:2]} vs drop_idx={self.drop_idx}")
        runs = []
        for index, token in enumerate(ids):
            if token != self.image_pad_id:
                continue
            if runs and index == runs[-1][0] + runs[-1][1]:
                runs[-1][1] += 1
            else:
                runs.append([index, 1])
        if len(runs) != len(images):
            raise ValueError(f"Zen processor produced {len(runs)} image slots for {len(images)} images")
        for image, (_, span) in zip(image_pil, runs):
            expected_span = (image.width // 32) * (image.height // 32)
            if span != expected_span:
                raise ValueError(f"Zen image slot has {span} tokens; expected {expected_span}")

        device = next(self.student.parameters()).device
        inputs = inputs.to(device)
        forward = {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask,
                   "output_hidden_states": True}
        if image_pil:
            forward["pixel_values"] = inputs.pixel_values.to(self.dtype)
            forward["image_grid_thw"] = inputs.image_grid_thw
        if hasattr(inputs, "mm_token_type_ids"):
            forward["mm_token_type_ids"] = inputs.mm_token_type_ids
        with torch.no_grad():
            hidden = self.student(**forward).hidden_states
            stacked = torch.stack([hidden[index] for index in self.layers], dim=1)
            stacked = stacked.permute(0, 2, 1, 3).reshape(stacked.shape[0], stacked.shape[2], -1)
            condition = self.fusion(stacked.to(self.dtype), inputs.attention_mask.bool())

        keep = torch.ones(len(ids), dtype=torch.bool)
        keep[:self.drop_idx] = False
        slots = []
        for start, span in runs:
            slots.append(int(keep[:start].sum()))
            keep[start:start + span] = False
        keep = keep.to(condition.device)
        condition = condition[:, keep]
        attention_mask = inputs.attention_mask[:, keep]
        extra = {}
        if slots:
            extra["image_slots"] = slots
        if not bool(attention_mask.all()):
            extra["attention_mask"] = attention_mask
        return condition, None, extra
