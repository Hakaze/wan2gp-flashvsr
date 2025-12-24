# Changelog

All notable changes to the FlashVSR Plugin for Wan2GP will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-23

### Added

- **4x Video Upscaling** using FlashVSR diffusion models
- **Three Pipeline Variants**:
  - Tiny (8-10GB VRAM) - Fastest, recommended for most users
  - Tiny-Long (10-12GB VRAM) - Optimized for videos >120 frames
  - Full (18-24GB VRAM) - Highest quality output
- **Sparse SageAttention** - Triton-based attention requiring no CUDA compilation
- **Tiled Processing** - Enable 8GB VRAM support via Tiled VAE and Tiled DiT
- **Automatic Model Downloads** - Models downloaded from HuggingFace on first run
- **VAE Sharing** - Reuses Wan2GP's existing `Wan2.1_VAE.safetensors` checkpoint
- **Color Correction** - Optional wavelet-based color fix
- **Audio Pass-Through** - Preserves original audio track in upscaled videos
- **FlashVSR-v1.1 Support** - Improved model with causal attention
- Comprehensive README with usage guide and benchmarks
- Detailed TROUBLESHOOTING.md for common issues

### Technical Details

- Based on [FlashVSR_plus](https://github.com/lihaoyun6/FlashVSR_plus) reference implementation
- Embedded Sparse SageAttention module (no external CUDA compilation required)
- Fallback chain: Sparse SageAttention → SageAttention → Flash Attention 2 → PyTorch SDPA
- Minimum 21 frames required (FlashVSR model constraint)

### Notes

- Image sequence input (folder of images) is not yet supported; video files only

[1.0.0]: https://github.com/Hakaze/wan2gp-flashvsr/releases/tag/v1.0.0
