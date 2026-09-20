<p align="center">
    <img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
</p>
<p align="center">
    💜 <a href="https://chat.qwen.ai/">Qwen Chat</a>&nbsp;&nbsp;|
    &nbsp;&nbsp;🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-2.1">HuggingFace</a>&nbsp;&nbsp;|
    &nbsp;&nbsp;🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-2.1">ModelScope</a>&nbsp;&nbsp;|
    &nbsp;&nbsp;📑 <a href="https://qwen.ai/blog?id=qwen-image-2.1">Blog</a>&nbsp;&nbsp;
    <br>
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-2.1">Demo</a>&nbsp;&nbsp;|
    &nbsp;&nbsp;💬 <a href="https://github.com/QwenLM/Qwen-Image-2.1/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp;&nbsp;|
    &nbsp;&nbsp;🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>
</p>

## Introduction

We are excited to open-source **Qwen-Image-2.1**, a unified text-to-image generation and image editing model in the Qwen family. With just **7B parameters in its visual generation component** (32 Single-Stream DiT layers), Qwen-Image-2.1 balances generation quality, inference efficiency, and versatility.

Four key improvements define this release:

- **Compact and Efficient** — A lightweight architecture with mixed-granularity attention and prefix KV cache reuse delivers strong image quality at low computational cost.
- **Native Transparency, Unified Creation and Editing** — Generate regular or transparent (RGBA) images from text, edit transparent layers, and extract subjects from photographs—all in one model.
- **Versatile Editing** — Support up to **10 reference images**, specify local edits via circles, painted annotations, or separate masks, and preserve identity for people and products.
- **Realistic Textures and Refined Aesthetics** — Improved typography, portrait lighting, and fine details for more visually compelling results.

<p align="center">
    <img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen-Image/image2.1/images/example-01.png" width="100%"/>
</p>

## News

- 2026.09.20: We released Qwen-Image-2.1! Check our [Blog](https://qwen.ai/blog?id=qwen-image-2.1) for more details. Weights available at [HuggingFace](https://huggingface.co/Qwen/Qwen-Image-2.1) and [ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image-2.1).
- 2026.09.20: [Diffusers](https://github.com/huggingface/diffusers) supports Qwen-Image-2.1 from Day 0 via `QwenImage21Pipeline`. See [PR #14804](https://github.com/huggingface/diffusers/pull/14804).
- 2026.09.20: [ComfyUI](https://github.com/Comfy-Org/ComfyUI) supports Qwen-Image-2.1 from Day 0, with block-causal attention, prefix KV caching, and RGBA output. See [PR #16400](https://github.com/Comfy-Org/ComfyUI/pull/16400).
- 2026.09.20: [vLLM-Omni](https://github.com/vllm-project/vllm-omni) supports high-performance Qwen-Image-2.1 inference from Day 0, with step-wise execution, prefix KV caching, CUDA Graph decode, FP8 quantization, and TP/Ulysses parallelism.
- 2026.09.20: [SGLang](https://github.com/sgl-project/sglang) provides Day-0 native support for Qwen-Image-2.1, including prefix caching, Cache-DiT, CUDA graphs, TP/Ulysses/Ring/CFG parallelism, and component offload. See [PR #39983](https://github.com/sgl-project/sglang/pull/39983).

## Quick Start

### Requirements

```bash
pip install torch>=2.4.0
pip install transformers>=5.17
pip install git+https://github.com/huggingface/diffusers
pip install accelerate
pip install pillow
```

### Text-to-Image

```python
import torch
from diffusers import QwenImage21Pipeline

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16
).to("cuda")

image = pipe(
    prompt="A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement",
    num_inference_steps=40,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("t2i_example.png")
```

### Image Editing (Single Image)

```python
import torch
from PIL import Image
from diffusers import QwenImage21Pipeline

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16
).to("cuda")

input_image = Image.open("input.png")

image = pipe(
    prompt="Change the background to a sunset beach",
    image=input_image,
    num_inference_steps=40,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("edit_example.png")
```

### Image Editing (Multiple Reference Images)

Qwen-Image-2.1 supports up to **10 reference images** for multi-subject composition:

```python
import torch
from PIL import Image
from diffusers import QwenImage21Pipeline

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16
).to("cuda")

images = [Image.open(f"ref_{i}.png") for i in range(3)]

result = pipe(
    prompt="These three characters are sitting around a campfire in a forest",
    image=images,
    num_inference_steps=40,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

result.save("multi_ref_example.png")
```

### Transparent Image Generation (RGBA)

The model natively generates transparent images when prompted:

```python
import torch
from diffusers import QwenImage21Pipeline

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16
).to("cuda")

# The model infers transparency from the prompt
image = pipe(
    prompt="A cute cartoon dragon sticker with transparent background, PNG asset",
    num_inference_steps=40,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("transparent_example.png")  # Saved as RGBA when the model generates transparency
```

### Supported Aspect Ratios

Qwen-Image-2.1 natively supports 2K resolution. Recommended sizes:

```python
aspect_ratios = {
    "1:1":  (2048, 2048),
    "4:3":  (2400, 1792),
    "3:4":  (1792, 2400),
    "3:2":  (2528, 1696),
    "2:3":  (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

width, height = aspect_ratios["1:1"]

image = pipe(
    prompt="A panoramic mountain landscape",
    width=width,
    height=height,
    num_inference_steps=40,
).images[0]
```

### Default Parameters

| Parameter | Default | Notes |
|---|---|---|
| `num_inference_steps` | 40 | Number of denoising steps |
| `width` / `height` | 2048 × 2048 | Native 2K resolution; see aspect ratio table above |

## Prompt Rewriting

For best results, we recommend using a prompt rewriting model to expand short prompts into detailed descriptions. This section will be updated with the official prompt rewriting tool and system prompts once available.

<!-- TODO: Add prompt rewriting tool, system prompt examples, and integration code -->

## Advanced Usage

### Memory Optimization

For GPUs with limited memory, use model offloading:

```python
pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
```

### Prefix KV Cache

The transformer automatically caches the text and condition-image prefix across denoising steps when the checkpoint has `causal_condition: true` (the default). This provides significant speedup for image editing tasks with multiple condition images — the condition context is encoded once and reused for all denoising steps.

## Inference with vLLM

[vLLM-Omni](https://github.com/vllm-project/vllm-omni) supports high-performance serving with prefix KV caching, CUDA Graph decode, FP8 quantization, and tensor parallelism.

### Offline Inference

```bash
# Text-to-image
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image-2.1 \
  --prompt "A ceramic teapot on a wooden table" \
  --output qwen21_t2i.png \
  --num-inference-steps 40

# Image editing
python examples/offline_inference/image_to_image/image_edit.py \
  --model Qwen/Qwen-Image-2.1 \
  --color-format RGBA \
  --seed 42 \
  --image input.png \
  --prompt "Let this mascot dance under the moon" \
  --output qwen21_edit.png \
  --num-inference-steps 40
```

### Online Serving

```bash
vllm serve Qwen/Qwen-Image-2.1 --omni --port 8091
```

```bash
curl http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-Image-2.1",
    "prompt": "A ceramic teapot on a wooden table",
    "size": "1024x1024",
    "num_inference_steps": 40,
    "seed": 42
  }'
```

For step-wise execution (batch-level scheduling):

```bash
vllm serve Qwen/Qwen-Image-2.1 --omni \
  --port 8091 \
  --step-execution \
  --max-num-seqs 8
```

See the [vLLM-Omni recipe](https://github.com/vllm-project/vllm-omni/blob/qwen-image-2.1/recipes/Qwen/Qwen-Image-2.1.md) for FP8 quantization, prefix KV cache options, multi-GPU parallelism, and detailed benchmarks.

## Inference with SGLang

**SGLang-Diffusion** provides native, high-performance inference for Qwen-Image 2.1, supporting text-to-image generation, multi-image editing, and transparent RGBA output. It offers multi-GPU parallelism, memory offloading, and optimized kernels across datacenter and consumer GPUs.

Generate a 1024×1024 image:

```bash
sglang generate \
  --model-path Qwen/Qwen-Image-2.1 \
  --prompt "A capybara reading a book by candlelight" \
  --height 1024 --width 1024 \
  --num-inference-steps 40 --guidance-scale 1 \
  --seed 42 --save-output
```

For image editing, add `--image-path input.png`. See the [Qwen-Image 2.1 cookbook](https://docs.sglang.io/cookbook/diffusion/Qwen-Image/Qwen-Image-2.1) for installation, GPU-specific commands, image editing, and transparent-background examples.

## Architecture

Qwen-Image-2.1 is a single-stream DiT with the following design:

- **Transformer**: 32 layers, 7B parameters, single-stream architecture with block-causal attention (`(q_idx >= kv_idx) or same_image_block`). Text uses token-level causal mask; images use chunk-level bidirectional mask.
- **Text Encoder**: Qwen3-VL 8B (vision-language model) — encodes both text instructions and condition images into a unified representation.
- **VAE**: 64-channel RGBA autoencoder with 16× spatial compression, supporting native transparency.
- **Scheduler**: Flow Matching with Euler discrete scheduling and dynamic shifting.

The mixed-granularity attention architecture enables efficient **prefix KV cache reuse**: input images and text instructions are computed once at the first denoising step and cached for all subsequent steps.

<p align="center">
    <img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen-Image/image2.1/images/example-03.png" width="100%"/>
</p>

## Showcase

### Native Transparency

<p align="center">
<img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen-Image/image2.1/images/example-04.png" width="30%"/>
<img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen-Image/image2.1/images/example-05.png" width="30%"/>
<img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen-Image/image2.1/images/example-06.png" width="30%"/>
</p>

### Multi-Reference Editing

<p align="center">
<img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen-Image/image2.1/images/example-15.png" width="100%"/>
</p>
<p align="center"><em>Group photograph generated from six individual portrait references</em></p>

<p align="center">
<img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen-Image/image2.1/images/example-16.png" width="100%"/>
</p>
<p align="center"><em>Complete outfit assembled from five reference images (model, clothing, shoes, bag, hat)</em></p>

### Local Editing

<p align="center">
<img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen-Image/image2.1/images/example-18.png" width="48%"/>
<img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen-Image/image2.1/images/example-19.png" width="48%"/>
</p>
<p align="center"><em>Circle-guided multi-region editing: remove watch, change hair color, replace clothing</em></p>

### Portrait and Product Fidelity

<p align="center">
<img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen-Image/image2.1/images/example-25.png" width="48%"/>
<img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen-Image/image2.1/images/example-26.png" width="48%"/>
</p>

<p align="center">
<img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen-Image/image2.1/images/example-31.png" width="48%"/>
<img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen-Image/image2.1/images/example-32.png" width="48%"/>
</p>

### Text Rendering

<p align="center">
<img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen-Image/image2.1/images/example-43.png" width="48%"/>
<img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen-Image/image2.1/images/example-44.png" width="48%"/>
</p>

### Panorama and Storyboard

<p align="center">
<img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen-Image/image2.1/images/example-38.png" width="100%"/>
</p>
<p align="center"><em>Panorama generated from a selfie</em></p>

<p align="center">
<img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen-Image/image2.1/images/example-42.png" width="100%"/>
</p>
<p align="center"><em>Storyboard generated from a three-view character reference</em></p>

## Community Support

### Diffusers (Recommended)

[Diffusers](https://github.com/huggingface/diffusers) supports Qwen-Image-2.1 via `QwenImage21Pipeline`, handling both text-to-image and image-conditioned generation in a single pipeline. See [PR #14804](https://github.com/huggingface/diffusers/pull/14804).

### ComfyUI

[ComfyUI](https://github.com/Comfy-Org/ComfyUI) provides native node-based support for Qwen-Image-2.1, with block-causal attention, prefix KV caching (~1.7× speedup on edits), and RGBA output. See [PR #16400](https://github.com/Comfy-Org/ComfyUI/pull/16400).

### vLLM-Omni

[vLLM-Omni](https://github.com/vllm-project/vllm-omni) provides production-grade serving with step-wise execution, CUDA Graph decode, FP8 quantization, and tensor/sequence parallelism. See the [Qwen-Image-2.1 recipe](https://github.com/vllm-project/vllm-omni/blob/qwen-image-2.1/recipes/Qwen/Qwen-Image-2.1.md).

### SGLang

[SGLang-Diffusion](https://github.com/sgl-project/sglang) provides native, high-performance inference with multi-GPU parallelism, memory offloading, and optimized kernels. See the [Qwen-Image 2.1 cookbook](https://docs.sglang.io/cookbook/diffusion/Qwen-Image/Qwen-Image-2.1) and [PR #39983](https://github.com/sgl-project/sglang/pull/39983).

### ModelScope

* [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) — Low-VRAM inference with layer-by-layer offload, FP8 quantization, LoRA/full training.
* [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine) — Advanced inference optimizations including FBCache acceleration.

## License Agreement

The code in this repository is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

The model weights are licensed under the [Qwen Research License Agreement](./LICENSE).

## Contact and Join Us

If you'd like to get in touch with our research team, join our [Discord](https://discord.gg/z3GAxXZ9Ce) or connect via [WeChat](assets/wechat.png). We welcome issues and pull requests on GitHub.

If you're passionate about fundamental research, we're hiring full-time employees and research interns. Reach out at fulai.hr@alibaba-inc.com.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image-2.1&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image-2.1&Date)
