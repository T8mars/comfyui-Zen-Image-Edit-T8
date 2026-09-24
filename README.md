# Zen Image Edit for ComfyUI / Zen Image Edit ComfyUI 节点

[English](#english) · [简体中文](#简体中文)

**Model / 模型下载:** [t8star/Zen-Image-Edit-Comfy](https://huggingface.co/t8star/Zen-Image-Edit-Comfy/)

## English

Zen Image Edit brings [AiArtLab/zen-image-edit](https://huggingface.co/AiArtLab/zen-image-edit) to ComfyUI as a single-file checkpoint. Its Qwen3.5-0.8B multimodal text encoder and fusion adapter produce a native `CLIP` output; image generation and editing continue through ComfyUI's Qwen-Image-2.1 conditioning, sampler, and VAE nodes. No Diffusers pipeline is required at runtime.

### Viggle Turbo v0.2.1 — native LoRA

[Download LoRAs](https://huggingface.co/t8star/Qwen-Image-2.1-viggle-turbo-4step-r64-comfy/tree/main) · [Text-to-image workflow](workflows/viggle_v021_text_to_image_ui.json) · [Image-edit workflow](workflows/viggle_v021_image_edit_ui.json)

These converted [Viggle](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) adapters work with **ComfyUI's built-in nodes**. The Zen extension is not required. Use ComfyUI `0.37.0` with frontend `1.53.6` or newer.

| File | Size | Folder |
|---|---:|---|
| `qwen_image_2.1_viggle_turbo_v0.2.1_r256_comfy.safetensors` | 1.76 GB | `models/loras/` |
| `qwen_image_2.1_viggle_turbo_v0.2.1_r128_comfy.safetensors` | 881 MB | `models/loras/` |

Choose one adapter. Download the separate base files from [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1): `qwen_image_2.1_bf16.safetensors` → `models/diffusion_models/`, `qwen3vl_8b_int8_convrot.safetensors` → `models/text_encoders/`, and `qwen_image_2.1_vae_bf16.safetensors` → `models/vae/`. Open a workflow and select the files; for editing, select your reference image in `LoadImage`.

**Required loader for these workflows:** the built-in **Load LoRA (Bypass, Model Only) (for debugging)** (`LoraLoaderBypassModelOnly`) at strength **1.0**. It computes the adapter separately, preserving updates that ordinary merged LoRA loading can round away. This conversion retains all 227 source projection adapters, including both halves of ComfyUI's fused MLP; it contains no base weights. The larger file size comes from lossless block-diagonal packing.

The workflows use **Euler, six steps, no CFG**, automatically shift the author's sigma schedule to the actual output size, and disable Qwen prefix caching. Keep the schedule group connected. The supported base is **BF16 DiT**; INT8 DiT has a fused MLP path that can skip bypass hooks. Use the original Qwen3-VL-8B encoder, not the Zen student encoder. The built-in bypass loader is experimental. [API examples](api_workflows/) are also included.

Validated with all custom nodes disabled: 512 px text-to-image and editing, complete native adapter mapping, and schedule equivalence. This is functional validation, not a full image-quality benchmark. Weights retain the upstream **Qwen Research License**; see the [conversion notice](https://huggingface.co/t8star/Qwen-Image-2.1-viggle-turbo-4step-r64-comfy/blob/main/NOTICE-v0.2.1).

### Zen installation

1. Update ComfyUI to a build with native Qwen-Image-2.1 support (validated on ComfyUI `0.37.0`). Install **Zen Image Edit T8** from ComfyUI Manager, or clone this repository into `ComfyUI/custom_nodes/`.
2. If installing manually, run `python -m pip install -r ComfyUI/custom_nodes/comfyui-Zen-Image-Edit-T8/requirements.txt` with ComfyUI's Python interpreter.
3. Download `zen_image_edit_qwen21_single.safetensors` from [t8star/Zen-Image-Edit-Comfy](https://huggingface.co/t8star/Zen-Image-Edit-Comfy/) and place it in `ComfyUI/models/checkpoints/`. Restart ComfyUI.
4. Open the [text-to-image](workflows/text_to_image_ui.json) or [image-edit](workflows/image_edit_ui.json) workflow. For editing, select an image in `LoadImage`; the [image-edit API example](api_workflows/image_edit_api.json) expects an uploaded `reference.png`. [Text-to-image API example](api_workflows/text_to_image_api.json).

The checkpoint contains the Qwen-Image-2.1 DiT and VAE plus the Zen student encoder, fusion adapter, and tokenizer/processor assets. You do not need to download a separate text encoder. The checkpoint loader outputs `MODEL`, `CLIP`, and `VAE`; the CLIP-only loader works with separate native diffusion and VAE loaders. The advanced encoder supports an explicit output canvas. Text encoding uses BF16; the CUDA/PyTorch runtime must support it.

For editing, connect the first image as the edit target and any following images as references. The stock `TextEncodeQwenImage21` node remains supported. Each reference must contain at least 65,536 pixels after native resizing; the adapter supports up to 2,304 processor tokens. Use plain prompt text: text inversion, prompt weights, CLIP skip, and text-encoder LoRA are unsupported. English prompts are recommended by the upstream model card. The validated GPU was an RTX 5090 Laptop with 24 GB VRAM; memory use rises with resolution and reference count.

The node code is Apache-2.0. The checkpoint combines weights from [AiArtLab](https://huggingface.co/AiArtLab/zen-image-edit) and [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1). Qwen-derived weights are limited to non-commercial research/evaluation under the Qwen Research License; commercial use requires separate permission. See the [model license](https://huggingface.co/t8star/Zen-Image-Edit-Comfy/blob/main/LICENSE) and [NOTICE.md](NOTICE.md).

## 简体中文

本节点将 [AiArtLab/zen-image-edit](https://huggingface.co/AiArtLab/zen-image-edit) 接入 ComfyUI。单文件模型内含 Qwen-Image-2.1 DiT、VAE、Qwen3.5-0.8B 多模态文本编码器及融合适配器。节点输出原生 `CLIP`，可继续使用 ComfyUI 的 Qwen-Image-2.1 条件编码、采样与 VAE 节点；运行时不依赖 Diffusers 管道。

### Viggle Turbo v0.2.1：原生 LoRA

[下载 LoRA](https://huggingface.co/t8star/Qwen-Image-2.1-viggle-turbo-4step-r64-comfy/tree/main) · [文生图工作流](workflows/viggle_v021_text_to_image_ui.json) · [图片编辑工作流](workflows/viggle_v021_image_edit_ui.json)

转换后的 [Viggle](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) LoRA **直接使用 ComfyUI 内置节点，无需安装 Zen 插件**。请使用 ComfyUI `0.37.0`、前端 `1.53.6` 或更新版本。

| 文件 | 大小 | 存放目录 |
|---|---:|---|
| `qwen_image_2.1_viggle_turbo_v0.2.1_r256_comfy.safetensors` | 1.76 GB | `models/loras/` |
| `qwen_image_2.1_viggle_turbo_v0.2.1_r128_comfy.safetensors` | 881 MB | `models/loras/` |

两种 LoRA 任选其一。另从 [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) 下载底座：`qwen_image_2.1_bf16.safetensors` 放入 `models/diffusion_models/`，`qwen3vl_8b_int8_convrot.safetensors` 放入 `models/text_encoders/`，`qwen_image_2.1_vae_bf16.safetensors` 放入 `models/vae/`。导入工作流并选择文件；编辑时在 `LoadImage` 选择参考图。

**工作流指定加载器：** ComfyUI 内置 **Load LoRA (Bypass, Model Only) (for debugging)**（`LoraLoaderBypassModelOnly`），强度 **1.0**。它独立计算 LoRA 分支，保留普通合并加载可能因舍入而丢失的更新。转换完整保留原始 227 组投影适配器，包括 ComfyUI 融合 MLP 的两个分支；文件不含底座权重。体积增加来自无损块对角打包。

工作流采用 **Euler、六步、无 CFG**，按实际输出尺寸自动调整作者的 sigma 日程，并关闭 Qwen 前缀缓存。保留调度组连线。支持的底座为 **BF16 DiT**；INT8 DiT 的融合 MLP 路径可能跳过 bypass hook。文本编码器使用原版 Qwen3-VL-8B，不使用 Zen student。内置 bypass 加载器目前标为实验性功能。另附 [API 示例](api_workflows/)。

已在禁用全部自定义节点的环境验证 512 像素文生图与编辑、全部适配器映射和调度等价性；这属于功能验证，不代表完整画质基准。权重沿用上游 **Qwen Research License**，详见[转换声明](https://huggingface.co/t8star/Qwen-Image-2.1-viggle-turbo-4step-r64-comfy/blob/main/NOTICE-v0.2.1)。

### Zen 安装

1. 更新到原生支持 Qwen-Image-2.1 的 ComfyUI（已在 `0.37.0` 验证）。在 ComfyUI Manager 安装 **Zen Image Edit T8**，或将本仓库克隆至 `ComfyUI/custom_nodes/`。
2. 手动安装时，使用 ComfyUI 对应的 Python 执行 `python -m pip install -r ComfyUI/custom_nodes/comfyui-Zen-Image-Edit-T8/requirements.txt`。
3. 从 [t8star/Zen-Image-Edit-Comfy 模型仓库](https://huggingface.co/t8star/Zen-Image-Edit-Comfy/)下载 `zen_image_edit_qwen21_single.safetensors`，放入 `ComfyUI/models/checkpoints/`，然后重启 ComfyUI。
4. 打开[文生图](workflows/text_to_image_ui.json)或[图片编辑](workflows/image_edit_ui.json)工作流。编辑时先在 `LoadImage` 选择图片；[图片编辑 API 示例](api_workflows/image_edit_api.json)需要预先上传名为 `reference.png` 的图片。另附[文生图 API 示例](api_workflows/text_to_image_api.json)。

无需另外下载文本编码器。整合加载器输出 `MODEL`、`CLIP`、`VAE`；独立 CLIP 加载器可搭配原生 DiT/VAE 加载器使用；高级编码器可指定输出画布。文本编码使用 BF16，CUDA/PyTorch 运行环境须支持该精度。编辑时，第一张图是编辑目标，后续图片是参考图。兼容原生 `TextEncodeQwenImage21`。每张参考图经原生缩放后须至少有 65,536 像素，融合适配器上限为 2,304 个处理器 token。请使用普通提示词；暂不支持文本反演、提示词权重、CLIP skip 或文本编码器 LoRA。原模型建议使用英文提示词。验收显卡为 24 GB 显存的 RTX 5090 Laptop；分辨率和参考图数量越高，显存需求越大。

节点代码采用 Apache-2.0。模型权重分别来自 [AiArtLab](https://huggingface.co/AiArtLab/zen-image-edit) 与 [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)。Qwen 衍生权重依照 Qwen Research License 仅限非商业研究/评估；商业使用须另行取得许可。详见[模型许可证](https://huggingface.co/t8star/Zen-Image-Edit-Comfy/blob/main/LICENSE)和 [NOTICE.md](NOTICE.md)。

## T8star

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [Hugging Face](https://huggingface.co/t8star) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery / 免费画廊](https://www.openzhenzhen.com) · [Online AI apps / 在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI portable bundle / ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd)
