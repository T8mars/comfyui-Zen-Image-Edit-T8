# Attribution

`fusion_lib.py` is adapted from the Apache-2.0 implementation in
[recoilme/zen-image-edit-comfyui](https://github.com/recoilme/zen-image-edit-comfyui/blob/main/fusion_lib.py).
The accompanying `LICENSE` is retained. The model weights have separate terms below.

The converted checkpoint combines the following independently published weights:

- Qwen-Image-2.1 DiT and VAE: [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1), revision `9a44dbdb47cefd046be9c0a13476192f34c8db8e`.
- Qwen3.5 student text encoder and Zen text-fusion adapter: [AiArtLab/zen-image-edit](https://huggingface.co/AiArtLab/zen-image-edit), revision `be4afb16f1f8d5032e7d77af4d5f63caa81ff25a`.

Review the upstream model cards and licenses before redistributing the converted checkpoint. The checkpoint is kept locally under `models/` and is excluded from this source tree's Git tracking.

The optional Viggle v0.2.1 LoRA conversion uses adapters from
[Viggle/Qwen-Image-2.1-viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo),
revision `b667ee50696e8f4ccd161e02f8af0e43bb5665ed`. Built with Qwen.
T8star changes tensor names and packs gate/up adapters into a block-diagonal
LoRA for ComfyUI's native fused layer. No base weights, training, rank
truncation, or weight merging are added. These derivative weights retain the
Qwen Research License and upstream NOTICE, distributed alongside the files
on the model repository. This source repository's Apache license does not
relicense model weights.
