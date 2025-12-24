# FlashVSR Video Upscaling Plugin for Wan2GP

AI-powered 4x video upscaling using FlashVSR models with Sparse SageAttention.

**Version**: 1.0.1  
**License**: Apache 2.0  
**Minimum VRAM**: 8GB

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Model Downloads](#model-downloads)
- [Usage Guide](#usage-guide)
- [Pipeline Variants](#pipeline-variants)
- [Advanced Settings](#advanced-settings)
- [Sparse SageAttention vs Block-Sparse Attention](#sparse-sageattention-vs-block-sparse-attention)
- [Performance Benchmarks](#performance-benchmarks)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)

---

## Features

- **4x Video Upscaling** - AI-powered upscaling using FlashVSR diffusion models
- **2x Mode Available** - Lower scale factor for faster processing
- **8GB VRAM Support** - Tiled processing enables operation on consumer GPUs
- **Three Pipeline Variants**:
  - **Tiny**: 8-10GB VRAM, fastest processing (recommended for most users)
  - **Tiny-Long**: 10-12GB VRAM, optimized for videos >120 frames
  - **Full**: 18-24GB VRAM, highest quality output
- **Pre-Flight Resource Check** - Estimates VRAM/RAM requirements before processing with recommendations
- **Cancellation Support** - Stop button to cancel long-running operations
- **Memory-Efficient Output** - Streaming frame writer prevents RAM exhaustion on large videos
- **Sparse SageAttention** - Triton-based attention requiring no CUDA compilation
- **Automatic Model Downloads** - Models downloaded from HuggingFace on first run
- **VAE Sharing** - Reuses Wan2GP's existing VAE checkpoint to save disk space
- **Color Correction** - Optional wavelet-based color fix for improved results
- **Audio Pass-Through** - Preserves original audio track in upscaled videos

---

## Installation

### Via Wan2GP Plugin Manager (Recommended)

1. Open Wan2GP
2. Navigate to the **Plugins** tab
3. Click **Install New Plugin**
4. Enter the GitHub URL: `https://github.com/Hakaze/wan2gp-flashvsr`
5. Click **Download and Install**
6. Restart Wan2GP

### Manual Installation

1. Clone or download this repository
2. Copy the `wan2gp-flashvsr` folder to `plugins/` in your Wan2GP installation
3. Install dependencies:
   ```bash
   pip install -r plugins/wan2gp-flashvsr/requirements.txt
   ```
4. Restart Wan2GP

### Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| PyTorch | 2.0+ with CUDA |
| CUDA | 11.8+ recommended |
| VRAM | 8GB minimum |
| Triton | 2.0+ |

### Dependencies

Most dependencies are already included with Wan2GP:

```
torch>=2.0.0
transformers>=4.40.0
accelerate>=0.20.0
einops>=0.7.0
safetensors>=0.4.0
diffsynth>=0.1.0
triton>=2.0.0
huggingface_hub>=0.16.0
```

---

## Model Downloads

Models are automatically downloaded from HuggingFace on first use. The download process:

1. **First Run**: When you click "Upscale Video", models download automatically
2. **Download Location**: `ckpts/flashvsr/`
3. **Total Size**: ~3GB (DiT model + TCDecoder)
4. **VAE Reuse**: Uses existing `ckpts/Wan2.1_VAE.safetensors` (saves ~1.5GB)

### Model Files

| Model | Size | Description |
|-------|------|-------------|
| `flashvsr_dit.safetensors` | ~2.5GB | Main diffusion transformer |
| `tcdecoder.safetensors` | ~300MB | Fast decoder for Tiny variants |
| `posi_prompt.pth` | 47KB | Pre-computed prompt embeddings (bundled) |

### Manual Model Download (Optional)

If you prefer to download models manually:

```bash
huggingface-cli download JunhaoZhuang/FlashVSR --local-dir ckpts/flashvsr/
```

---

## Usage Guide

### Basic Usage

1. **Open FlashVSR Tab**: Navigate to "FlashVSR Upscaling" in Wan2GP
2. **Upload Video**: Click "Input Video" and select your video file
3. **Choose Settings**:
   - Select model variant (Tiny recommended)
   - Choose scale factor (4x recommended)
   - Enable Tiled DiT for videos with 50+ frames
4. **Click Upscale**: Press "🚀 Upscale Video" button
5. **Wait for Processing**: Progress bar shows current status
   - A "⬛ Stop Processing" button appears during upscaling
   - Click it to cancel (stops after current operation)
6. **Download Result**: Upscaled video appears in output panel

### Input Requirements

| Parameter | Recommendation |
|-----------|----------------|
| **Input Resolution** | 480p-720p (will be upscaled to 1920p-2880p) |
| **Minimum Frames** | 21 frames (FlashVSR requirement) |
| **Video Format** | MP4, AVI, MOV, MKV |
| **Frame Rate** | Any (preserved in output) |

> **Note**: This plugin currently supports video files only. Image sequence input (folder of images) is available in the upstream FlashVSR_plus CLI but is not yet implemented in this Wan2GP plugin. The "Output FPS" setting serves as a fallback when video metadata cannot be read.

### Example Workflow

**Scenario**: Upscale a 480p video to 1920p

1. Upload `my_video_480p.mp4` (320x480, 60 frames)
2. Select "Tiny (8-10GB VRAM)" variant
3. Keep "4x" scale factor
4. Enable "Color Fix" for better colors
5. Click upscale
6. Output: 1280x1920 video with enhanced details

---

## Pipeline Variants

Choose the pipeline that matches your GPU and use case:

### Tiny (Recommended)

- **VRAM**: 8-10GB
- **Speed**: Fastest (8-12 FPS on RTX 4090)
- **Quality**: Good
- **Decoder**: TCDecoder (fast, minimal quality loss)
- **Best For**: Most users, 8GB GPUs

### Tiny-Long

- **VRAM**: 10-12GB
- **Speed**: Fast (6-10 FPS on RTX 4090)
- **Quality**: Good
- **Features**: Streaming buffer for memory efficiency
- **Best For**: Long videos (>120 frames)

### Full

- **VRAM**: 18-24GB
- **Speed**: Slower (4-6 FPS on RTX 4090)
- **Quality**: Highest
- **Decoder**: Full Wan2.1 VAE
- **Best For**: Quality-priority, high-end GPUs

---

## Advanced Settings

### VRAM Optimization

| Setting | Default | Description |
|---------|---------|-------------|
| **Tiled VAE** | ✅ On | Tiles VAE processing for large resolutions |
| **Tiled DiT** | ❌ Off | Tiles DiT processing; **recommended for 50+ frames** |
| **Tile Size** | 256 | Smaller = less VRAM, slower (128-512) |
| **Tile Overlap** | 24px | Reduces seam artifacts at tile boundaries |

**For 8GB GPUs**: Enable both Tiled VAE and Tiled DiT with tile size 256.

**For 50+ frame videos**: Enable Tiled DiT regardless of VRAM. Without it, DiT attention is O(n²) across all frames, causing very slow processing even on high-end GPUs.

### Quality Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Color Fix** | ✅ On | Wavelet-based color correction |
| **Output Quality** | 6 | Video compression quality (1-10) |
| **Output FPS** | 30 | Fallback FPS when video metadata is unavailable |

### Processing Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Data Type** | bf16 | Use fp16 for older GPUs without bf16 support |
| **Unload DiT** | ❌ Off | Free VRAM during decode (slower) |

### Sparse Attention Parameters

These control the efficiency of the attention mechanism:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Sparse Ratio** | 2.0 | 0.5-5.0 | Controls sparsity; lower = more sparse, faster |
| **KV Ratio** | 3 | 1-8 | KV cache compression ratio |
| **Local Range** | 11 | 3-15 | Local attention window size (odd numbers) |

**Tip**: Keep defaults unless you experience quality issues or need more speed.

### Model Version

| Version | Description |
|---------|-------------|
| **FlashVSR** | Original model (buffer-based) |
| **FlashVSR-v1.1** | Improved model with causal attention (recommended) |

---

## Sparse SageAttention vs Block-Sparse Attention

This plugin uses **Sparse SageAttention** instead of the original Block-Sparse Attention (BSA). Here's why:

### What is Block-Sparse Attention?

The original FlashVSR paper uses Block-Sparse Attention, which:
- Requires pre-compiling CUDA kernels for specific GPU architectures
- Has compatibility issues across different NVIDIA GPU generations
- Provides theoretical maximum performance but with installation complexity

### What is Sparse SageAttention?

Sparse SageAttention is an alternative implementation that:
- Uses **Triton kernels** that compile on-the-fly
- Works across all modern NVIDIA GPUs without special compilation
- Uses INT8 quantization for efficient memory usage
- Part of the well-maintained SageAttention ecosystem

### Comparison

| Aspect | Block-Sparse | Sparse SageAttention |
|--------|--------------|----------------------|
| **Installation** | Complex (CUDA compilation) | Simple (pip install) |
| **GPU Compatibility** | Limited | Broad |
| **Speed** | Fastest | ~10-15% slower |
| **Memory** | Low | Low (INT8 quantization) |
| **Maintenance** | Difficult | Active community |

### Fallback Chain

If Sparse SageAttention encounters issues, the plugin falls back through:

1. **Sparse SageAttention** (default, Triton-based)
2. **SageAttention** (standard version)
3. **Flash Attention 2** (if installed)
4. **PyTorch scaled_dot_product_attention** (guaranteed to work)

This ensures the plugin works on all systems, even without optimal performance.

---

## Performance Benchmarks

Tested with 480p → 1920p upscaling (4x), 60 frames:

### GPU Performance (FPS)

| GPU | VRAM | Tiny | Tiny-Long | Full |
|-----|------|------|-----------|------|
| RTX 4090 | 24GB | 10-12 | 8-10 | 5-6 |
| RTX 4080 | 16GB | 8-10 | 6-8 | 3-4* |
| RTX 3090 | 24GB | 8-10 | 6-8 | 4-5 |
| RTX 3080 | 10GB | 6-8 | 5-7 | N/A |
| RTX 3070 | 8GB | 4-6† | 3-5† | N/A |
| RTX 3060 | 12GB | 5-7 | 4-6 | N/A |

\* Requires Tiled DiT  
† Requires Tiled DiT and Tiled VAE

### Processing Time Examples

| Input | Output | Variant | GPU | Time |
|-------|--------|---------|-----|------|
| 480p, 60 frames | 1920p | Tiny | RTX 4090 | ~6 sec |
| 480p, 60 frames | 1920p | Tiny | RTX 3080 | ~10 sec |
| 480p, 120 frames | 1920p | Tiny-Long | RTX 4090 | ~15 sec |
| 720p, 60 frames | 2880p | Tiny | RTX 4090 | ~12 sec |

### VRAM Usage

| Configuration | Approximate VRAM |
|---------------|------------------|
| Tiny, 1080p output | 8-9 GB |
| Tiny-Long, 1080p | 10-11 GB |
| Full, 1080p | 18-20 GB |
| Tiny + Tiled DiT, 1080p | 6-7 GB |

---

## Troubleshooting

For comprehensive troubleshooting, see **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** which covers:
- Detailed solutions for all common issues
- VRAM optimization strategies by GPU tier
- DiffSynth compatibility notes
- Advanced debugging techniques

### Quick Solutions

| Issue | Quick Fix |
|-------|-----------|
| "Not enough frames" | Use video with 21+ frames, or loop short videos |
| Out of Memory (OOM) | Enable **Tiled DiT**, reduce **Tile Size** to 128-192 |
| Very slow processing | Enable **Tiled DiT** for videos with 50+ frames |
| Slow first run | Wait for model download (~3GB), one-time only |
| Cancellation not working | Cannot interrupt mid-inference; restart app if stuck |
| Triton errors | `pip install triton>=2.0.0 --force-reinstall` |
| Pixelated output | Increase **Sparse Ratio** to 3.0+, increase **Tile Size** |
| Color issues | Enable **Color Fix**, try **FlashVSR-v1.1** model |
| No audio in output | Verify ffmpeg is installed: `ffmpeg -version` |

### Getting Help

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions
2. Review the [Wan2GP Issues](https://github.com/Wan-Video/Wan2.1/issues) page
3. Enable verbose logging: `python wgp.py --verbose 2`

---

## Credits

### Research & Models

- **FlashVSR Paper**: [FlashVSR: Efficient Video Super-Resolution](https://arxiv.org/abs/2510.12747)
- **FlashVSR Models**: [JunhaoZhuang/FlashVSR on HuggingFace](https://huggingface.co/JunhaoZhuang/FlashVSR)

### Reference Implementation

- **FlashVSR_plus**: [lihaoyun6/FlashVSR_plus](https://github.com/lihaoyun6/FlashVSR_plus) - The proven implementation this plugin is based on

### Open-Source Projects & Libraries 

* **FlashVSR** — [https://github.com/OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR)
- **Sparse SageAttention**: [jt-zhang/Sparse_SageAttention_API](https://github.com/jt-zhang/Sparse_SageAttention_API)
- **SageAttention Team**: Apache 2.0 licensed attention kernels
- **taehv** — [https://github.com/madebyollin/taehv](https://github.com/madebyollin/taehv)
- **DiffSynth Studio**: Model management and pipeline abstractions
- **Wan2GP**: Plugin system and video generation platform

### 📜 Citation

```bibtex
@misc{zhuang2025flashvsrrealtimediffusionbasedstreaming,
      title={FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution}, 
      author={Junhao Zhuang and Shi Guo and Xin Cai and Xiaohui Li and Yihao Liu and Chun Yuan and Tianfan Xue},
      year={2025},
      eprint={2510.12747},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.12747}, 
}
```

---

## License

This plugin is licensed under the **Apache License 2.0**, the same license as FlashVSR.

```
Copyright 2025 FlashVSR Authors and Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

See [LICENSE.txt](src/models/sparse_sage/LICENSE.txt) for the full license text.
