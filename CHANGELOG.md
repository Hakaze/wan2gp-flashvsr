# Changelog

All notable changes to the FlashVSR Plugin for Wan2GP will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2025-12-23

### Added

- **Pre-Flight Resource Check** - Estimates VRAM/RAM requirements before processing starts
  - Uses SAFETY_FACTOR of 4.0 for accurate estimation (based on ComfyUI-FlashVSR_Stable)
  - Displays warnings with recommended settings when resources are insufficient
  - Logs processing summary with peak VRAM usage on completion
- **Cancellation Support** - Stop button to cancel long-running upscaling operations
  - Two-button pattern: Upscale button hides, Stop button appears during processing
  - Graceful cancellation with cleanup of partial files
  - Note: Cannot interrupt mid-inference; stops between operations
- **Adaptive Batch Sizing** - Dynamically calculates safe batch sizes based on available RAM
  - Queries system RAM at runtime using `psutil`
  - Prevents memory exhaustion during video output phase

### Changed

- **Memory-Efficient Video Output** - Streaming frame writer replaces batch-to-numpy conversion
  - Eliminates 10-15GB+ RAM spike during 4K output
  - Progressive cleanup of intermediate tensors
- **Improved Tiled DiT Guidance** - UI now recommends Tiled DiT for 50+ frame videos
  - Updated info text explaining O(n²) complexity without tiling
  - Pre-flight warns when >100 frames without Tiled DiT enabled
- **Status Feedback** - Replaced status textbox with Gradio toast notifications
  - `gr.Info` for success messages
  - `gr.Warning` for warnings and recommendations  
  - `gr.Error` for error messages

### Fixed

- **Memory Exhaustion on 4x Upscale** - 100+ frame 4K videos no longer cause system lockups
- **No Way to Cancel** - Users can now stop processing (between operations)

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
