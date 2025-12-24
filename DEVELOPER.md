# FlashVSR Plugin Developer Guide

This document provides technical details for developers who want to understand, extend, or contribute to the FlashVSR plugin for Wan2GP.

## Architecture Overview

The FlashVSR plugin follows the standard Wan2GP plugin architecture, inheriting from `WAN2GPPlugin`.

### Core Components

1.  **`FlashVSRPlugin` (plugin.py)**: The main entry point.
    *   **`setup_ui`**: Registers the "FlashVSR Upscaling" tab.
    *   **`post_ui_setup`**: Wires up event handlers (buttons, dropdowns).
    *   **`upscale_video`**: The main orchestration method that handles video loading, preprocessing, pipeline execution, and saving.
    *   **`on_tab_select` / `on_tab_deselect`**: Manages VRAM by loading/unloading models when the user switches tabs.

2.  **Pipelines (`src/pipelines/`)**:
    *   **`BasePipeline`**: Abstract base class for all FlashVSR pipelines.
    *   **`FlashVSRTinyPipeline`**: Optimized for 8-10GB VRAM. Uses `TCDecoder` for fast, low-memory decoding.
    *   **`FlashVSRTinyLongPipeline`**: Similar to Tiny but with streaming buffers for long videos (>120 frames).
    *   **`FlashVSRFullPipeline`**: Uses the full Wan2.1 VAE. Requires 24GB+ VRAM.

3.  **Models (`src/models/`)**:
    *   **`WanModel`**: The Diffusion Transformer (DiT) backbone.
    *   **`WanVideoVAE`**: The Variational Autoencoder (used in Full pipeline).
    *   **`TCDecoder`**: Tiny Consistency Decoder (used in Tiny pipelines).
    *   **`sparse_sage`**: Custom sparse attention implementation for efficiency.

## Pipeline System

The plugin uses a strategy pattern for pipelines. All pipelines share a common interface but implement different memory/performance trade-offs.

### Key Methods in Pipelines

*   **`__init__`**: Loads model components (DiT, VAE/Decoder).
*   **`__call__`**: Runs the generation loop.
*   **`encode_video`**: Encodes input video to latents (usually via simple bicubic upscaling + convolution in FlashVSR).
*   **`decode_video`**: Decodes latents back to pixel space.

### Adding a New Pipeline

To add a new pipeline variant (e.g., "Medium"):

1.  Create `src/pipelines/flashvsr_medium.py`.
2.  Inherit from `BasePipeline`.
3.  Implement `__init__` to load your specific combination of models.
4.  Implement `__call__` for the inference logic.
5.  Register the new pipeline in `src/pipelines/__init__.py`.
6.  Update `download_manager.py` to handle model downloads for the new variant.
7.  Update `plugin.py` to add the new variant to the dropdown and `load_pipeline` logic.

## Customizing the Model

### Sparse Attention

The `sparse_sage` module implements the core optimization. You can tune:
*   **`sparse_ratio`**: Controls the sparsity level (higher = faster/less VRAM, lower = better quality).
*   **`kv_ratio`**: Ratio of key/value cache compression.
*   **`local_range`**: Window size for local attention.

These are exposed in the UI but can also be hardcoded in `src/configs/model_config.py` if needed.

## Integration

### Using FlashVSR from Other Plugins

Currently, the FlashVSR plugin is designed as a standalone tool within its own tab. However, you can access its components if needed:

```python
# In your plugin
def setup_ui(self):
    # Request access to FlashVSR components if they exist
    # Note: Component IDs are prefixed with the plugin name usually, 
    # but this plugin creates its own isolated Blocks.
    pass
```

*Note: Direct programmatic access to the upscaler from other plugins is not currently exposed via global functions. If you need this, consider refactoring `upscale_video` to be a global utility.*

### Shared Resources

*   **VAE**: The plugin attempts to share the `Wan2.1_VAE.safetensors` from the main Wan2GP installation to save disk space.
*   **Models**: FlashVSR specific models are stored in `ckpts/flashvsr/`.

## Debugging

*   Enable `debug=True` in `plugin.py` (if available) or add print statements.
*   Check `wgp.log` or the console output for Triton/CUDA errors.
*   Use `clean_vram()` helper in `plugin.py` to force garbage collection during development.
