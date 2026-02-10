# Changelog

All notable changes to the FlashVSR Plugin for Wan2GP will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.3] - 2026-02-09

### Fixes & Optimized
- `snapshot_download` now excludes the pickle VAE — Added `ignore_patterns=["Wan2.1_VAE.pth"]` so the ~0.5 GB redundant file is no longer downloaded from the FlashVSR HuggingFace repo.
- **Fixed** the `required_files` validation — Was checking for "dit.safetensors" and "config.json" which don't match the actual filenames, causing unnecessary re-download attempts every launch. Now checks for the real filenames: diffusion_pytorch_model_streaming_dmd.safetensors, LQ_proj_in.ckpt, TCDecoder.ckpt.
- Rewrote `get_vae_path()` with download fallback — Now always returns a valid path:
  - First checks Wan2.1_VAE.safetensors (most common case)
  - Falls back to Wan2.1_VAE.pth (legacy)
  - Removed the dead check for ckpts/flashvsr/vae.pth (never existed)
  - If neither found, downloads the safetensors VAE from DeepBeepMeep/Wan2.1 using hf_hub_download — the same source Wan2GP's model manager uses
- Removed `convert_vae_safetensors_to_pth()` — The FlashVSR ModelManager → `load_state_dict()` already handles .safetensors files natively via safe_open, so the pickle conversion was unnecessary and insecure.
- Simplified `get_model_paths()` VAE resolution — Replaced the 20-line if/else block with a single line: `paths["vae"] = get_vae_path()`.
- Added a stale-file notice — If a previous Wan2.1_VAE.pth exists from an older download, prints a message suggesting the user can delete it to reclaim ~0.5 GB.

## [1.0.2] - 2025-12-29

### Fixed

- **Output Video Quality = 10 produced audio-only files** - Updated MP4 encoding to use explicit H.264 settings (libx264 + yuv420p) and a stable quality→CRF mapping for reliable video streams.

### Housekeeping
- Fixed some incorrect links and formatting issues in `README` and `TROUBLESHOOTING`.

[1.0.2]: https://github.com/Hakaze/wan2gp-flashvsr/releases/tag/v1.0.2

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

[1.0.1]: https://github.com/Hakaze/wan2gp-flashvsr/releases/tag/v1.0.1

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
