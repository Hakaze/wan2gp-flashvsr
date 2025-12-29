# FlashVSR Plugin Troubleshooting Guide

A comprehensive guide for diagnosing and resolving common issues with the FlashVSR video upscaling plugin.

---

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Common Issues and Solutions](#common-issues-and-solutions)
  - [Installation Issues](#installation-issues)
  - [Model Download Issues](#model-download-issues)
  - [Runtime Errors](#runtime-errors)
  - [Quality Problems](#quality-problems)
  - [Performance Issues](#performance-issues)
- [VRAM Optimization Guide](#vram-optimization-guide)
  - [Understanding VRAM Usage](#understanding-vram-usage)
  - [Optimization Strategies by GPU](#optimization-strategies-by-gpu)
  - [Tiled Processing Deep Dive](#tiled-processing-deep-dive)
- [DiffSynth Compatibility](#diffsynth-compatibility)
- [Advanced Debugging](#advanced-debugging)
- [Getting Help](#getting-help)

---

## Quick Diagnostics

Before diving into specific issues, run these quick checks:

### 1. Verify Plugin Loading
```
Open Wan2GP → Plugins tab → Check "FlashVSR Upscaling" is enabled
```

### 2. Check GPU Detection
```python
# In Python console or script:
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 3. Verify Triton Installation
```bash
python -c "import triton; print(f'Triton version: {triton.__version__}')"
```

### 4. Test Basic Imports
```python
# Should complete without errors
from plugins.wan2gp_flashvsr.src.models.sparse_sage import sparse_sageattn
from plugins.wan2gp_flashvsr.src.pipelines import FlashVSRTinyPipeline
print("All imports successful!")
```

---

## Common Issues and Solutions

### Installation Issues

#### Plugin Not Appearing in Tab List

**Symptoms**: FlashVSR tab doesn't appear after installing the plugin.

**Solutions**:
1. **Restart Wan2GP completely** - Close and reopen the application
2. **Check `__init__.py`** exists in `plugins/wan2gp-flashvsr/`
3. **Check `plugin.py`** exists and has the correct class structure
4. **Review console output** for import errors during startup

```bash
# Run with verbose logging
python wgp.py --verbose 2
```

#### Import Errors on Startup

**Symptoms**: Console shows `ModuleNotFoundError` or `ImportError`.

**Common causes and fixes**:

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: diffsynth` | DiffSynth not installed | `pip install diffsynth` |
| `ModuleNotFoundError: triton` | Triton not installed | `pip install triton>=2.0.0` |
| `ImportError: sparse_sage` | Corrupted plugin files | Re-download plugin |
| `ModuleNotFoundError: einops` | Missing dependency | `pip install einops>=0.7.0` |

#### Requirements Installation Fails

**Symptoms**: `pip install -r requirements.txt` shows errors.

**Solutions**:
1. **Update pip**: `pip install --upgrade pip`
2. **Install PyTorch first** (with CUDA support):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```
3. **Install requirements one-by-one** to identify the problematic package
4. **Check Python version** - requires Python 3.10+

---

### Model Download Issues

#### Download Stalls or Times Out

**Symptoms**: Progress bar stops, or "Connection timeout" errors.

**Solutions**:
1. **Check internet connection** - HuggingFace CDN may be slow in some regions
2. **Use a VPN** if HuggingFace is blocked in your region
3. **Manual download**:
   ```bash
   huggingface-cli download JunhaoZhuang/FlashVSR --local-dir ckpts/flashvsr/
   ```
4. **Set HuggingFace mirror** (for China users):
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

#### "Model file not found" After Download

**Symptoms**: Error about missing `.safetensors` files even after download completed.

**Solutions**:
1. **Verify download location**: Files should be in `ckpts/flashvsr/`
2. **Check file sizes** - incomplete downloads will be smaller than expected:
   - `flashvsr_dit.safetensors`: ~2.5GB
   - `tcdecoder.safetensors`: ~300MB
3. **Clear and re-download**:
   ```bash
   rm -rf ckpts/flashvsr/
   # Then restart and let plugin re-download
   ```

#### posi_prompt.pth Missing

**Symptoms**: Error about `posi_prompt.pth` not found.

**Solution**: This file is bundled with the plugin at `plugins/wan2gp-flashvsr/prompt/posi_prompt.pth`. If missing:
1. Re-download the plugin from the repository
2. Or manually download from FlashVSR source repository

---

### Runtime Errors

#### "Not enough frames" Error

**Symptoms**: `RuntimeError: Not enough frames. Got X.`

**Cause**: FlashVSR requires a minimum of 21 frames for temporal processing.

**Solutions**:
1. **Use longer video** - at least 1 second at 24fps
2. **Loop short videos**:
   ```bash
   ffmpeg -stream_loop 3 -i short.mp4 -c copy looped.mp4
   ```
3. **Pad with duplicate frames** using video editor

#### Out of Memory (OOM) / CUDA Error

**Symptoms**: `RuntimeError: CUDA out of memory` or sudden application crash.

**Cause**: VRAM exhausted during processing.

**Immediate Solutions**:
1. Enable **Tiled DiT** in Advanced Settings
2. Reduce **Tile Size** to 128 or 192
3. Switch to **Tiny** pipeline variant
4. Close other GPU applications (games, browsers with hardware acceleration)
5. Enable **Unload DiT before Decoding** option

**See [VRAM Optimization Guide](#vram-optimization-guide) for detailed strategies.**

#### Triton Compilation Errors

**Symptoms**: Errors mentioning `triton`, `autotuning`, or `kernel compilation failed`.

**Common errors and fixes**:

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: triton.language` | Old Triton version | `pip install triton>=2.0.0` |
| `Compilation failed` | Incompatible CUDA | Update CUDA toolkit to 11.8+ |
| `autotuning failed` | Cache corruption | Delete `~/.triton/` cache folder |

**Nuclear option** - disable Triton entirely:
```bash
# The plugin will fall back to PyTorch attention
export TRITON_DISABLE=1
python wgp.py
```

#### "reinit_cross_kv" AttributeError

**Symptoms**: `AttributeError: WanModel lacks reinit_cross_kv method`

**Cause**: Model initialization failed or corrupted model files.

**Solutions**:
1. Delete and re-download model files:
   ```bash
   rm -rf ckpts/flashvsr/
   ```
2. Verify you have the correct FlashVSR model version (v1.1 recommended)

#### Frame Count Mismatch Errors

**Symptoms**: Errors about tensor shape mismatch during color correction or blending.

**Cause**: Known issue with frame alignment during decoder processing.

**Solution**: This was fixed in plugin v2.0.0. Update to the latest version. If you still encounter it:
1. Disable **Color Fix** temporarily
2. Try **Tiny-Long** pipeline instead of **Tiny**

---

### Quality Problems

#### Output Looks Pixelated/Blocky

**Symptoms**: Upscaled video has visible tile boundaries or blocky artifacts.

**Causes and Solutions**:

1. **Sparse attention parameters too aggressive**:
   - Increase **Sparse Ratio** from 2.0 to 3.0 or higher
   - Increase **KV Ratio** from 3 to 5
   - Decrease **Local Range** if > 11

2. **Tile size too small**:
   - Increase **Tile Size** to 256 or 320 (if VRAM allows)
   - Increase **Tile Overlap** to 32 or 48 pixels

3. **Wrong pipeline variant**:
   - Switch to **Full** pipeline for best quality (needs 18GB+ VRAM)

#### Colors Look Wrong/Washed Out

**Symptoms**: Output video has different colors than input.

**Solutions**:
1. **Enable Color Fix** in Quality Settings (wavelet-based correction)
2. **Try FlashVSR-v1.1** model version (improved color handling)
3. **Check input video** - some codecs may have color space issues

#### Output Has Temporal Flickering

**Symptoms**: Frame-to-frame inconsistency or flickering in output.

**Solutions**:
1. Use **Tiny-Long** pipeline for better temporal consistency
2. Enable **Color Fix** for more stable colors
3. Ensure input video is high quality (avoid over-compressed sources)

#### Visible Seams Between Tiles

**Symptoms**: Grid pattern or visible lines in output when using tiled processing.

**Solutions**:
1. Increase **Tile Overlap** to 32-48 pixels (higher = better blending, slower)
2. Increase **Tile Size** if VRAM allows
3. The overlap is scaled with upscale factor (24px input = 96px at 4x scale)

---

### Performance Issues

#### Extremely Slow Processing

**Symptoms**: Processing takes much longer than benchmark times.

**Possible causes**:

1. **First run compilation** - Triton compiles kernels on first use (one-time ~2-3 min)
2. **CPU fallback** - Check if attention is falling back to PyTorch:
   ```
   Look for "Using PyTorch scaled_dot_product_attention" in console
   ```
3. **Thermal throttling** - GPU overheating, check temperatures
4. **Power limit** - Laptop GPUs may throttle. Connect to power adapter.

**Solutions**:
- Wait for first-run compilation to complete
- Install Flash Attention 2 for faster fallback: `pip install flash-attn`
- Improve GPU cooling
- Use Desktop GPU if possible

#### Processing Seems Stuck

**Symptoms**: Progress bar stops moving for extended periods.

**Possible causes**:
1. **Large tile processing** - Each tile takes time, especially on lower-end GPUs
2. **Model loading** - First frame may take longer due to lazy model loading
3. **Memory swapping** - System is using disk as swap (very slow)

**Solutions**:
- Wait longer (first tile is slowest)
- Enable **Unload DiT before Decoding** to reduce memory pressure
- Reduce tile size to process faster per tile

---

## VRAM Optimization Guide

### Understanding VRAM Usage

FlashVSR VRAM consumption depends on:

| Factor | Impact |
|--------|--------|
| **Output Resolution** | Higher res = more VRAM (quadratic scaling) |
| **Pipeline Variant** | Full > Tiny-Long > Tiny |
| **Tiled Processing** | Reduces peak VRAM significantly |
| **Number of Frames** | More frames = more memory for buffers |
| **Data Type** | fp16 vs bf16 (similar VRAM, bf16 recommended) |

### Approximate VRAM Requirements

| Configuration | Peak VRAM |
|---------------|-----------|
| Tiny, 1080p, no tiling | 10-12 GB |
| Tiny, 1080p, tiled | 6-8 GB |
| Tiny-Long, 1080p, no tiling | 12-14 GB |
| Tiny-Long, 1080p, tiled | 8-10 GB |
| Full, 1080p, no tiling | 20-24 GB |

### Optimization Strategies by GPU

#### 8GB GPUs (RTX 3060, RTX 3070, RTX 4060)

**Recommended Settings**:
```
Pipeline: Tiny
Tiled VAE: ✅ ON
Tiled DiT: ✅ ON
Tile Size: 192-256
Tile Overlap: 24
Unload DiT: ✅ ON
Data Type: fp16 (if bf16 causes issues)
```

**Tips**:
- Close all other GPU applications
- Use 2x upscaling instead of 4x for faster processing
- Process in shorter segments if still OOM

#### 10-12GB GPUs (RTX 3080 10GB, RTX 3060 12GB)

**Recommended Settings**:
```
Pipeline: Tiny or Tiny-Long
Tiled VAE: ✅ ON
Tiled DiT: OFF (try ON if needed)
Tile Size: 256
Tile Overlap: 24
```

#### 16GB GPUs (RTX 4080, RTX 4070 Ti)

**Recommended Settings**:
```
Pipeline: Tiny-Long (best balance)
Tiled VAE: ✅ ON
Tiled DiT: OFF
Full pipeline: Possible for short videos
```

#### 24GB+ GPUs (RTX 4090, RTX 3090)

**Recommended Settings**:
```
Pipeline: Any (Full for best quality)
Tiled VAE: OFF (or ON for 4K output)
Tiled DiT: OFF
```

### Tiled Processing Deep Dive

#### How Tiled DiT Works

Tiled DiT divides the video spatially into overlapping tiles:

```
┌──────────────────────────────┐
│  Tile 1  │overlap│  Tile 2  │
├──────────┼───────┼──────────┤
│ overlap  │ blend │ overlap  │
├──────────┼───────┼──────────┤
│  Tile 3  │overlap│  Tile 4  │
└──────────────────────────────┘
```

- Each tile is processed independently (lower VRAM)
- Overlapping regions are blended with feathered weights
- Trade-off: slower processing, potential seam artifacts

#### Tile Size vs VRAM Trade-off

| Tile Size | VRAM Reduction | Processing Speed | Quality |
|-----------|----------------|------------------|---------|
| 128 | Maximum | Slowest | Good (more blending) |
| 192 | High | Slow | Very Good |
| 256 | Medium | Medium | Excellent |
| 320 | Low | Fast | Best |
| 512 | Minimal | Fastest | Best |

#### Calculating Optimal Tile Size

For a target VRAM budget:

```
Available VRAM for tiles ≈ Total VRAM - 4GB (model overhead)
Tile Size ≈ sqrt(Available VRAM × 1024 / frame_count / 3)
```

Example for 8GB GPU processing 60 frames:
```
Available = 8 - 4 = 4GB = 4096 MB
Tile Size ≈ sqrt(4096 × 1024 / 60 / 3) ≈ 150px → round to 192
```

---

## DiffSynth Compatibility

### What is DiffSynth?

DiffSynth Studio is a framework for diffusion models that provides:
- Model management and loading utilities
- Pipeline abstractions
- State dict conversion helpers

FlashVSR uses DiffSynth for model management.

### Common DiffSynth Issues

#### Version Compatibility

**Recommended**: DiffSynth >= 0.1.0

```bash
pip install --upgrade diffsynth
```

#### ModelManager Conflicts

**Symptoms**: Errors about model registry or duplicate model loading.

**Cause**: Multiple DiffSynth versions or conflicting model managers.

**Solution**:
```bash
pip uninstall diffsynth
pip install diffsynth==0.1.0
```

#### State Dict Loading Errors

**Symptoms**: `KeyError` or `size mismatch` during model loading.

**Cause**: FlashVSR model format differs from DiffSynth expectations.

**Solution**: The plugin includes custom state dict converters. If issues persist:
1. Verify you have FlashVSR-v1.1 models (not v1.0)
2. Delete `ckpts/flashvsr/` and re-download

### Working Without DiffSynth

If DiffSynth causes persistent issues, the core pipeline can work without it:

1. The plugin has fallback model loading in `download_manager.py`
2. State dict conversion is handled internally
3. Only advanced features may be affected

---

## Advanced Debugging

### Enable Verbose Logging

```bash
python wgp.py --verbose 2
```

This shows:
- Model loading progress
- Attention mechanism selection
- Tile processing progress
- VRAM usage at each stage

### Memory Profiling

Add to your workflow:

```python
import torch

def print_vram():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

### Check Attention Backend

```python
from plugins.wan2gp_flashvsr.src.models.sparse_sage import sparse_sageattn

# Test which attention is being used
try:
    # Small test
    import torch
    q = torch.randn(1, 8, 64, 64, device='cuda', dtype=torch.float16)
    k = torch.randn(1, 8, 64, 64, device='cuda', dtype=torch.float16)
    v = torch.randn(1, 8, 64, 64, device='cuda', dtype=torch.float16)
    result = sparse_sageattn(q, k, v)
    print("Sparse SageAttention working!")
except Exception as e:
    print(f"Fallback will be used: {e}")
```

### Force Specific Attention Backend

```python
import os

# Force PyTorch attention (slowest but most compatible)
os.environ['FLASHVSR_FORCE_PYTORCH_ATTN'] = '1'

# Disable Triton
os.environ['TRITON_DISABLE'] = '1'
```

### Profile Individual Tiles

For debugging tiled processing issues:

```python
import time

# Before each tile
start = time.time()
print(f"Processing tile {tile_idx}...")

# After tile
elapsed = time.time() - start
print(f"Tile {tile_idx} completed in {elapsed:.2f}s")
```

---

## Getting Help

### Before Asking for Help

1. **Collect system info**:
   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
   pip freeze | grep -E "torch|triton|diffsynth|einops"
   ```

2. **Check console output** for the exact error message

3. **Try with default settings** before reporting issues

### Where to Get Help

1. **wan2gp-flashvsr Github Issues**: [Hakaze/wan2gp-flashvsr/issues](https://github.com/Hakaze/wan2gp-flashvsr/issues)
2. **Wan2GP GitHub Issues**: [deepbeepmeep/Wan2GP/issues](https://github.com/deepbeepmeep/Wan2GP/issues)
3. **FlashVSR_plus Reference**: [lihaoyun6/FlashVSR_plus](https://github.com/lihaoyun6/FlashVSR_plus)
4. **FlashVSR Paper**: [arXiv:2510.12747](https://arxiv.org/abs/2510.12747)

### Reporting Bugs

Include in your bug report:
- GPU model and VRAM
- PyTorch and CUDA versions
- Exact error message and stack trace
- Settings used (pipeline, tile size, etc.)
- Input video details (resolution, frame count, format)

---

## Quick Reference Card

### Recommended Settings by Scenario

| Scenario | Pipeline | Tiled | Tile Size | Notes |
|----------|----------|-------|-----------|-------|
| 8GB GPU, 1080p output | Tiny | DiT + VAE | 192 | Enable Unload DiT |
| 12GB GPU, 1080p output | Tiny-Long | VAE only | 256 | Best quality/speed |
| 24GB GPU, 4K output | Full | VAE only | 256 | Maximum quality |
| Long video (>200 frames) | Tiny-Long | VAE | 256 | Streaming buffer |
| Maximum speed | Tiny | None | - | Needs 12GB+ |
| Maximum quality | Full | None | - | Needs 24GB+ |

### Error Quick Reference

| Error | Likely Cause | Quick Fix |
|-------|--------------|-----------|
| OOM | Insufficient VRAM | Enable Tiled DiT |
| "Not enough frames" | < 21 frames | Loop video |
| Triton compile fail | Old Triton | `pip install triton>=2.0` |
| Model not found | Download incomplete | Re-download models |
| Pixelated output | Settings too aggressive | Increase Sparse Ratio |
| Color issues | Color space mismatch | Enable Color Fix |
