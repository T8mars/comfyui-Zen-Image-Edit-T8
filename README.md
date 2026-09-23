# Zen Image Edit for ComfyUI / Zen Image Edit ComfyUI 节点

[English](#english) · [简体中文](#简体中文)

**Model / 模型下载:** [t8star/Zen-Image-Edit-Comfy](https://huggingface.co/t8star/Zen-Image-Edit-Comfy/)

## English

Zen Image Edit brings [AiArtLab/zen-image-edit](https://huggingface.co/AiArtLab/zen-image-edit) to ComfyUI as a single-file checkpoint. Its Qwen3.5-0.8B multimodal text encoder and fusion adapter produce a native `CLIP` output; image generation and editing continue through ComfyUI's Qwen-Image-2.1 conditioning, sampler, and VAE nodes. No Diffusers pipeline is required at runtime.

### Install

1. Update ComfyUI to a build with native Qwen-Image-2.1 support (validated on ComfyUI `0.37.0`). Install **Zen Image Edit T8** from ComfyUI Manager, or clone this repository into `ComfyUI/custom_nodes/`.
2. If installing manually, run `python -m pip install -r ComfyUI/custom_nodes/comfyui-Zen-Image-Edit-T8/requirements.txt` with ComfyUI's Python interpreter.
3. Download `zen_image_edit_qwen21_single.safetensors` from [t8star/Zen-Image-Edit-Comfy](https://huggingface.co/t8star/Zen-Image-Edit-Comfy/) and place it in `ComfyUI/models/checkpoints/`. Restart ComfyUI.
4. Open the [text-to-image](workflows/text_to_image_ui.json) or [image-edit](workflows/image_edit_ui.json) workflow. For editing, select an image in `LoadImage`; the [image-edit API example](api_workflows/image_edit_api.json) expects an uploaded `reference.png`. [Text-to-image API example](api_workflows/text_to_image_api.json).

The checkpoint contains the Qwen-Image-2.1 DiT and VAE plus the Zen student encoder, fusion adapter, and tokenizer/processor assets. You do not need to download a separate text encoder. The checkpoint loader outputs `MODEL`, `CLIP`, and `VAE`; the CLIP-only loader works with separate native diffusion and VAE loaders. The advanced encoder supports an explicit output canvas. Text encoding uses BF16; the CUDA/PyTorch runtime must support it.

For editing, connect the first image as the edit target and any following images as references. The stock `TextEncodeQwenImage21` node remains supported. Each reference must contain at least 65,536 pixels after native resizing; the adapter supports up to 2,304 processor tokens. Use plain prompt text: text inversion, prompt weights, CLIP skip, and text-encoder LoRA are unsupported. English prompts are recommended by the upstream model card. The validated GPU was an RTX 5090 Laptop with 24 GB VRAM; memory use rises with resolution and reference count.

The node code is Apache-2.0. The checkpoint combines weights from [AiArtLab](https://huggingface.co/AiArtLab/zen-image-edit) and [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1). Qwen-derived weights are limited to non-commercial research/evaluation under the Qwen Research License; commercial use requires separate permission. See the [model license](https://huggingface.co/t8star/Zen-Image-Edit-Comfy/blob/main/LICENSE) and [NOTICE.md](NOTICE.md).

## 简体中文

本节点将 [AiArtLab/zen-image-edit](https://huggingface.co/AiArtLab/zen-image-edit) 接入 ComfyUI。单文件模型内含 Qwen-Image-2.1 DiT、VAE、Qwen3.5-0.8B 多模态文本编码器及融合适配器。节点输出原生 `CLIP`，可继续使用 ComfyUI 的 Qwen-Image-2.1 条件编码、采样与 VAE 节点；运行时不依赖 Diffusers 管道。

### 安装

1. 更新到原生支持 Qwen-Image-2.1 的 ComfyUI（已在 `0.37.0` 验证）。在 ComfyUI Manager 安装 **Zen Image Edit T8**，或将本仓库克隆至 `ComfyUI/custom_nodes/`。
2. 手动安装时，使用 ComfyUI 对应的 Python 执行 `python -m pip install -r ComfyUI/custom_nodes/comfyui-Zen-Image-Edit-T8/requirements.txt`。
3. 从 [t8star/Zen-Image-Edit-Comfy 模型仓库](https://huggingface.co/t8star/Zen-Image-Edit-Comfy/)下载 `zen_image_edit_qwen21_single.safetensors`，放入 `ComfyUI/models/checkpoints/`，然后重启 ComfyUI。
4. 打开[文生图](workflows/text_to_image_ui.json)或[图片编辑](workflows/image_edit_ui.json)工作流。编辑时先在 `LoadImage` 选择图片；[图片编辑 API 示例](api_workflows/image_edit_api.json)需要预先上传名为 `reference.png` 的图片。另附[文生图 API 示例](api_workflows/text_to_image_api.json)。

无需另外下载文本编码器。整合加载器输出 `MODEL`、`CLIP`、`VAE`；独立 CLIP 加载器可搭配原生 DiT/VAE 加载器使用；高级编码器可指定输出画布。文本编码使用 BF16，CUDA/PyTorch 运行环境须支持该精度。编辑时，第一张图是编辑目标，后续图片是参考图。兼容原生 `TextEncodeQwenImage21`。每张参考图经原生缩放后须至少有 65,536 像素，融合适配器上限为 2,304 个处理器 token。请使用普通提示词；暂不支持文本反演、提示词权重、CLIP skip 或文本编码器 LoRA。原模型建议使用英文提示词。验收显卡为 24 GB 显存的 RTX 5090 Laptop；分辨率和参考图数量越高，显存需求越大。

节点代码采用 Apache-2.0。模型权重分别来自 [AiArtLab](https://huggingface.co/AiArtLab/zen-image-edit) 与 [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)。Qwen 衍生权重依照 Qwen Research License 仅限非商业研究/评估；商业使用须另行取得许可。详见[模型许可证](https://huggingface.co/t8star/Zen-Image-Edit-Comfy/blob/main/LICENSE)和 [NOTICE.md](NOTICE.md)。

## T8star

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [Hugging Face](https://huggingface.co/t8star) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery / 免费画廊](https://www.openzhenzhen.com) · [Online AI apps / 在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI portable bundle / ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd)
