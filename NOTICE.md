# Attribution

`fusion_lib.py` is adapted from the text-fusion implementation in
[AiArtLab/zen-image-edit](https://huggingface.co/AiArtLab/zen-image-edit),
published under Apache-2.0. The accompanying `LICENSE` is retained.

The converted checkpoint combines the following independently published weights:

- Qwen-Image-2.1 DiT and VAE: [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1), revision `9a44dbdb47cefd046be9c0a13476192f34c8db8e`.
- Qwen3.5 student text encoder and Zen text-fusion adapter: [AiArtLab/zen-image-edit](https://huggingface.co/AiArtLab/zen-image-edit), revision `be4afb16f1f8d5032e7d77af4d5f63caa81ff25a`.

Review the upstream model cards and licenses before redistributing the converted checkpoint. The checkpoint is kept locally under `models/` and is excluded from this source tree's Git tracking.
