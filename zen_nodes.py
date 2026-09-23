"""ComfyUI nodes for the native single-file Zen Image Edit checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

import torch

import comfy.model_management
import comfy.model_detection
import comfy.model_patcher
import comfy.sd
import folder_paths
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
from comfy_extras.nodes_qwen import TextEncodeQwenImage21
from comfy.supported_models_base import ClipTarget

from .checkpoint import component, metadata
from .clip import ZenClipModel, ZenTokenizer


def checkpoint_path(name: str) -> Path:
    if os.path.isabs(name):
        path = Path(name)
    else:
        path = Path(folder_paths.get_full_path_or_raise("checkpoints", name))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def checkpoint_choices() -> list[str]:
    names = []
    for name in folder_paths.get_filename_list("checkpoints"):
        if not name.lower().endswith(".safetensors"):
            continue
        try:
            if metadata(folder_paths.get_full_path_or_raise("checkpoints", name)):
                names.append(name)
        except (OSError, ValueError, KeyError):
            continue
    return sorted(names) or ["<no Zen single checkpoint found>"]


def checkpoint_fingerprint(name: str):
    path = checkpoint_path(name)
    stat = path.stat()
    return (str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def load_clip(path: Path, precision: str):
    metadata(path)
    if precision == "auto":
        precision = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    target = ClipTarget(ZenTokenizer, ZenClipModel)
    target.params["checkpoint_path"] = str(path)
    model_options = {"dtype": dtype, "initial_device": torch.device("cpu"),
                     "offload_device": comfy.model_management.text_encoder_offload_device()}
    clip = comfy.sd.CLIP(target, model_options=model_options, disable_dynamic=True)
    clip.cond_stage_model.load_checkpoint_weights()
    return clip


def load_diffusion(path: Path):
    # Construct on meta first: a second 14 GB CPU allocation can exceed the
    # Windows commit limit while the single checkpoint is memory-mapped.
    weights = component(path, "diffusion")
    config = comfy.model_detection.model_config_from_unet(weights, "")
    if config is None or config.unet_config.get("image_model") != "qwen_image21":
        raise ValueError("The embedded diffusion model is not native Qwen-Image-2.1")
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    dtype = torch.bfloat16
    manual_cast = comfy.model_management.unet_manual_cast(
        dtype, load_device, config.supported_inference_dtypes
    )
    config.set_inference_dtype(dtype, manual_cast, device=load_device)
    model = config.get_model(weights, device=torch.device("meta"))
    model.load_model_weights(weights, "", assign=True)
    if weights:
        raise ValueError(f"Unused Qwen-Image-2.1 weights: {list(weights)[:8]}")
    model.device = torch.device("cpu")
    return comfy.model_patcher.ModelPatcher(
        model, load_device=load_device, offload_device=offload_device
    )


class ZenImageEditCheckpointLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ZenImageEditCheckpointLoader",
            display_name="Zen Image Edit — Single Checkpoint Loader",
            category="loaders/zen image edit",
            inputs=[
                io.Combo.Input("checkpoint", options=checkpoint_choices()),
                io.Combo.Input("te_precision", options=["auto", "bf16", "fp16"], default="auto"),
                io.Float.Input("shift", default=5.0, min=0.01, max=20.0, step=0.01),
            ],
            outputs=[io.Model.Output(display_name="MODEL"),
                     io.Clip.Output(display_name="CLIP"),
                     io.Vae.Output(display_name="VAE")],
        )

    @classmethod
    def fingerprint_inputs(cls, checkpoint, **kwargs):
        return checkpoint_fingerprint(checkpoint)

    @classmethod
    def execute(cls, checkpoint, te_precision="auto", shift=5.0):
        path = checkpoint_path(checkpoint)
        metadata(path)
        model = load_diffusion(path)
        vae = comfy.sd.VAE(sd=component(path, "vae"))
        vae.throw_exception_if_invalid()
        clip = load_clip(path, te_precision)
        model = ModelSamplingAuraFlow().patch_aura(model, shift)[0]
        return io.NodeOutput(model, clip, vae)


class ZenImageEditCLIPLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ZenImageEditCLIPLoader",
            display_name="Zen Image Edit — CLIP Loader",
            category="loaders/zen image edit",
            inputs=[
                io.Combo.Input("checkpoint", options=checkpoint_choices()),
                io.Combo.Input("te_precision", options=["auto", "bf16", "fp16"], default="auto"),
            ],
            outputs=[io.Clip.Output(display_name="CLIP")],
        )

    @classmethod
    def fingerprint_inputs(cls, checkpoint, **kwargs):
        return checkpoint_fingerprint(checkpoint)

    @classmethod
    def execute(cls, checkpoint, te_precision="auto"):
        return io.NodeOutput(load_clip(checkpoint_path(checkpoint), te_precision))


class ZenImageEditEncodeAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ZenImageEditEncodeAdvanced",
            display_name="Zen Image Edit — Text Encode / Canvas",
            category="model/conditioning/qwen image",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("negative_prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Int.Input("resolution", default=1024, min=0, max=4096, step=32),
                io.Combo.Input("canvas_mode", options=["auto", "explicit"], default="auto"),
                io.Int.Input("width", default=1024, min=32, max=4096, step=32),
                io.Int.Input("height", default=1024, min=32, max=4096, step=32),
                io.Autogrow.Input("images", template=io.Autogrow.TemplateNames(
                    io.Image.Input("image"), names=[f"image_{i}" for i in range(1, 17)], min=0)),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"),
                     io.Conditioning.Output(display_name="negative"),
                     io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, clip, prompt, negative_prompt, vae=None, resolution=1024,
                canvas_mode="auto", width=1024, height=1024, images=None, **kwargs):
        image_inputs = dict(images or {})
        image_inputs.update({key: value for key, value in kwargs.items() if key.startswith("image_")})
        image_inputs = {name: value for name, value in image_inputs.items() if value is not None}
        for name, image in image_inputs.items():
            if image.shape[0] != 1:
                raise ValueError(f"{name} has batch size {image.shape[0]}; use one reference per input")
        result = TextEncodeQwenImage21.execute(clip, prompt, negative_prompt, vae=vae,
                                               resolution=resolution, images=image_inputs)
        if canvas_mode == "explicit":
            if width % 32 or height % 32:
                raise ValueError("Zen canvas width and height must be multiples of 32")
            latent = torch.zeros([1, 64, height // 16, width // 16],
                                 device=comfy.model_management.intermediate_device())
            return io.NodeOutput(result[0], result[1], {"samples": latent})
        return result


class ZenImageEditExtension(ComfyExtension):
    async def get_node_list(self):
        return [ZenImageEditCheckpointLoader, ZenImageEditCLIPLoader, ZenImageEditEncodeAdvanced]


async def comfy_entrypoint():
    return ZenImageEditExtension()
