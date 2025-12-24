"""
FlashVSR Video Upscaling Plugin for Wan2GP

This plugin provides 4x video upscaling using FlashVSR models.
Based on the FlashVSR_plus implementation by lihaoyun6.

Copyright 2025 Wan2GP Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Features:
- 4x video upscaling with AI models
- Support for 8GB GPUs with tile_dit optimization
- Three pipeline variants (Tiny/Tiny-Long/Full)
- Sparse SageAttention for efficient processing
"""

from shared.utils.plugins import WAN2GPPlugin
import gradio as gr
import torch
import torch.nn.functional as F_torch
import numpy as np
import math


def create_feather_mask(size, overlap):
    """
    Create a feather mask for blending overlapping tiles.
    Matches the upstream FlashVSR_plus implementation.
    
    Args:
        size: Tuple of (height, width) of the tile
        overlap: Overlap in pixels (already scaled to output resolution)
        
    Returns:
        Tensor of shape (1, 1, H, W) with linear ramp feather weights
    """
    H, W = size
    mask = torch.ones(1, 1, H, W)
    
    if overlap <= 0:
        return mask
    
    ramp = torch.linspace(0, 1, overlap)
    
    # Left edge
    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
    # Right edge
    mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
    # Top edge
    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    # Bottom edge
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    
    return mask


def calculate_tile_coords(height, width, tile_size, overlap):
    """
    Calculate tile coordinates for spatial tiling with overlap.
    Matches the upstream FlashVSR_plus implementation.
    
    Note: These are coordinates at the ORIGINAL (source) resolution.
    The pipeline will upscale each tile, and results are stitched
    at the scaled resolution.
    
    Args:
        height: Total height of the source image/video
        width: Total width of the source image/video
        tile_size: Size of each tile (at source resolution)
        overlap: Overlap between adjacent tiles in pixels (at source resolution)
        
    Returns:
        List of (x1, y1, x2, y2) tuples (note: x1, y1 order matches upstream)
    """
    coords = []
    stride = tile_size - overlap
    num_rows = math.ceil((height - overlap) / stride)
    num_cols = math.ceil((width - overlap) / stride)
    
    for r in range(num_rows):
        for c in range(num_cols):
            y1 = r * stride
            x1 = c * stride
            y2 = min(y1 + tile_size, height)
            x2 = min(x1 + tile_size, width)
            
            # Adjust start if tile is smaller than tile_size at boundary
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
            
            coords.append((x1, y1, x2, y2))
    
    return coords


def largest_8n1_leq(n):
    """Return largest value <= n of form 8k+1."""
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def next_8n5(n):
    """Return next value >= n of form 8k+5."""
    return 21 if n < 21 else ((n - 5 + 7) // 8) * 8 + 5


def get_input_params(image_tensor, scale):
    """
    Calculate input parameters for FlashVSR pipeline.
    Matches upstream FlashVSR_plus implementation.
    
    Args:
        image_tensor: Input video tensor of shape (N, H, W, C)
        scale: Upscale factor (2 or 4)
        
    Returns:
        Tuple of (target_height, target_width, num_frames)
    """
    N0, h0, w0, _ = image_tensor.shape
    multiple = 128
    sW, sH = w0 * scale, h0 * scale
    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)
    F = largest_8n1_leq(N0 + 4)
    if F == 0:
        raise RuntimeError(f"Not enough frames. Got {N0 + 4}.")
    return tH, tW, F


def prepare_input_tensor(image_tensor, device, scale=4, dtype=torch.bfloat16):
    """
    Prepare input tensor for FlashVSR pipeline.
    Matches upstream FlashVSR_plus implementation - prepares LQ_video
    with bicubic upscaling to target resolution.
    
    Args:
        image_tensor: Input video tensor of shape (N, H, W, C) in [0, 1] range
        device: Target device
        scale: Upscale factor
        dtype: Target dtype
        
    Returns:
        Tuple of (LQ_video, target_height, target_width, num_frames)
        LQ_video shape: (1, C, F, H, W) in [-1, 1] range
    """
    N0, h0, w0, _ = image_tensor.shape
    tH, tW, Fs = get_input_params(image_tensor, scale)
    
    frames = []
    for i in range(Fs):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_bchw = frame_slice.permute(2, 0, 1).unsqueeze(0)
        
        # Bicubic upscale to scaled dimensions
        upscaled_tensor = F_torch.interpolate(
            tensor_bchw, 
            size=(h0 * scale, w0 * scale), 
            mode='bicubic', 
            align_corners=False
        )
        
        # Center crop to aligned target dimensions
        l = max(0, (w0 * scale - tW) // 2)
        t = max(0, (h0 * scale - tH) // 2)
        cropped_tensor = upscaled_tensor[:, :, t:t + tH, l:l + tW]
        
        # Normalize to [-1, 1]
        tensor_out = (cropped_tensor.squeeze(0) * 2.0 - 1.0).to('cpu').to(dtype)
        frames.append(tensor_out)
    
    vid_stacked = torch.stack(frames, 0)
    vid_final = vid_stacked.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, F, H, W)
    
    # Clean VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return vid_final, tH, tW, Fs


def tensor2video(frames_tensor):
    """
    Convert output tensor to video frames.
    Matches upstream FlashVSR_plus implementation.
    
    Args:
        frames_tensor: Tensor of shape (C, F, H, W) or (1, C, F, H, W) in [-1, 1] range
        
    Returns:
        Tensor of shape (F, H, W, C) in [0, 1] range
    """
    from einops import rearrange
    video_squeezed = frames_tensor.squeeze(0) if frames_tensor.dim() == 5 else frames_tensor
    video_permuted = rearrange(video_squeezed, "C F H W -> F H W C")
    video_final = (video_permuted.float() + 1.0) / 2.0
    return video_final


def clean_vram():
    """Clean VRAM by emptying CUDA cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tensor_upscale_then_center_crop(frame_slice, scale, tW, tH):
    """
    Upscale a frame tensor using bicubic interpolation, then center crop.
    
    Args:
        frame_slice: Frame tensor of shape (H, W, C) in [0, 1] range
        scale: Upscale factor
        tW: Target width after cropping
        tH: Target height after cropping
        
    Returns:
        Tensor of shape (C, H, W) in [0, 1] range
    """
    h0, w0, _ = frame_slice.shape
    tensor_bchw = frame_slice.permute(2, 0, 1).unsqueeze(0)
    
    # Bicubic upscale to scaled dimensions
    upscaled_tensor = F_torch.interpolate(
        tensor_bchw, 
        size=(h0 * scale, w0 * scale), 
        mode='bicubic', 
        align_corners=False
    )
    
    # Center crop to aligned target dimensions
    l = max(0, (w0 * scale - tW) // 2)
    t = max(0, (h0 * scale - tH) // 2)
    cropped_tensor = upscaled_tensor[:, :, t:t + tH, l:l + tW]
    
    return cropped_tensor.squeeze(0)


def input_tensor_generator(image_tensor, device, scale=4, dtype=torch.bfloat16):
    """
    Generator function that yields prepared frame tensors one at a time.
    Used by Tiny-Long pipeline for memory-efficient streaming.
    
    Args:
        image_tensor: Input video tensor of shape (N, H, W, C) in [0, 1] range
        device: Target device
        scale: Upscale factor
        dtype: Target dtype
        
    Yields:
        Tensor of shape (C, H, W) in [-1, 1] range for each frame
    """
    N0, h0, w0, _ = image_tensor.shape
    tH, tW, F = get_input_params(image_tensor, scale)
    
    for i in range(F):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_chw = tensor_upscale_then_center_crop(frame_slice, scale=scale, tW=tW, tH=tH)
        tensor_out = tensor_chw * 2.0 - 1.0
        del tensor_chw
        yield tensor_out.to('cpu').to(dtype)


def stitch_video_tiles(
    tile_paths, 
    tile_coords, 
    final_dims, 
    scale, 
    overlap, 
    output_path, 
    fps, 
    quality, 
    cleanup=True,
    chunk_size=40
):
    """
    Stitch multiple tile videos into a single output video.
    Used by Tiny-Long pipeline for tiled processing.
    
    Args:
        tile_paths: List of paths to tile video files
        tile_coords: List of (x1, y1, x2, y2) coordinates for each tile
        final_dims: Tuple of (width, height) for final output
        scale: Upscale factor used
        overlap: Tile overlap in pixels (at source resolution)
        output_path: Path to write the stitched video
        fps: Output video FPS
        quality: Output video quality (1-10)
        cleanup: Whether to remove temp tile files after stitching
        chunk_size: Number of frames to process at once (for memory efficiency)
    """
    import imageio
    import os
    from tqdm import tqdm
    
    if not tile_paths:
        print("[FlashVSR] No tile videos found to stitch.")
        return
    
    final_W, final_H = final_dims
    
    # Open all video files
    readers = [imageio.get_reader(p) for p in tile_paths]
    
    try:
        # Get total frame count
        num_frames = readers[0].count_frames()
        if num_frames is None or num_frames <= 0:
            num_frames = len([_ for _ in readers[0]])
            for r in readers:
                r.close()
            readers = [imageio.get_reader(p) for p in tile_paths]
        
        # Open output writer
        with imageio.get_writer(output_path, fps=fps, quality=quality) as writer:
            
            # Process in chunks for memory efficiency
            for start_frame in tqdm(range(0, num_frames, chunk_size), desc="[FlashVSR] Stitching Chunks"):
                end_frame = min(start_frame + chunk_size, num_frames)
                current_chunk_size = end_frame - start_frame
                
                # Create canvas for this chunk
                chunk_canvas = np.zeros((current_chunk_size, final_H, final_W, 3), dtype=np.float32)
                weight_canvas = np.zeros_like(chunk_canvas, dtype=np.float32)
                
                # Process each tile
                for i, reader in enumerate(readers):
                    try:
                        # Read frames for this chunk using get_data for random access
                        tile_chunk_frames = []
                        for frame_idx in range(start_frame, end_frame):
                            try:
                                frame = reader.get_data(frame_idx)
                                tile_chunk_frames.append(frame.astype(np.float32) / 255.0)
                            except IndexError:
                                # Reached end of video
                                break
                        
                        if not tile_chunk_frames:
                            print(f"[FlashVSR] Warning: No frames read from tile {i} for range {start_frame}-{end_frame}")
                            continue
                            
                        tile_chunk_np = np.stack(tile_chunk_frames, axis=0)
                    except Exception as e:
                        print(f"[FlashVSR] Warning: Could not read chunk from tile {i}: {e}")
                        continue
                    
                    if tile_chunk_np.shape[0] != current_chunk_size:
                        print(f"[FlashVSR] Warning: Tile {i} chunk has {tile_chunk_np.shape[0]} frames, expected {current_chunk_size}. Adjusting...")
                        # Adjust current_chunk_size for this iteration if needed
                        actual_chunk_size = tile_chunk_np.shape[0]
                    else:
                        actual_chunk_size = current_chunk_size
                    # Create feather mask
                    tile_H, tile_W, _ = tile_chunk_np.shape[1:]
                    scaled_overlap = overlap * scale
                    if scaled_overlap > 0:
                        ramp = np.linspace(0, 1, scaled_overlap, dtype=np.float32)
                        mask = np.ones((tile_H, tile_W, 1), dtype=np.float32)
                        mask[:, :scaled_overlap, :] *= ramp[np.newaxis, :, np.newaxis]
                        mask[:, -scaled_overlap:, :] *= np.flip(ramp)[np.newaxis, :, np.newaxis]
                        mask[:scaled_overlap, :, :] *= ramp[:, np.newaxis, np.newaxis]
                        mask[-scaled_overlap:, :, :] *= np.flip(ramp)[:, np.newaxis, np.newaxis]
                    else:
                        mask = np.ones((tile_H, tile_W, 1), dtype=np.float32)
                    mask_4d = mask[np.newaxis, :, :, :]
                    
                    # Blend into canvas
                    x1_orig, y1_orig, _, _ = tile_coords[i]
                    out_y1, out_x1 = y1_orig * scale, x1_orig * scale
                    out_y2, out_x2 = out_y1 + tile_H, out_x1 + tile_W
                    
                    chunk_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += tile_chunk_np * mask_4d
                    weight_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_4d
                
                # Normalize and write frames
                weight_canvas[weight_canvas == 0] = 1.0
                stitched_chunk = chunk_canvas / weight_canvas
                
                for frame_idx_in_chunk in range(current_chunk_size):
                    frame_uint8 = (np.clip(stitched_chunk[frame_idx_in_chunk], 0, 1) * 255).astype(np.uint8)
                    writer.append_data(frame_uint8)
                    
    finally:
        print("[FlashVSR] Closing all tile reader instances...")
        for reader in readers:
            reader.close()
    
    if cleanup:
        print("[FlashVSR] Cleaning up temporary tile files...")
        for path in tile_paths:
            try:
                os.remove(path)
            except OSError as e:
                print(f"[FlashVSR] Could not remove temporary file '{path}': {e}")


class FlashVSRPlugin(WAN2GPPlugin):
    """
    FlashVSR video upscaling plugin for Wan2GP.
    
    This plugin provides AI-powered 4x video upscaling using FlashVSR models,
    based on the FlashVSR_plus implementation by lihaoyun6. It supports multiple
    pipeline variants optimized for different VRAM configurations:
    
    - Tiny (8-10GB VRAM): Fastest, uses TCDecoder for efficient decoding
    - Tiny-Long (10-12GB VRAM): Optimized for long videos (>120 frames)
    - Full (18-24GB VRAM): Highest quality, uses full VAE decoder
    
    Key Features:
        - Sparse SageAttention for efficient memory usage
        - Tile-based processing for low-VRAM GPUs (8GB minimum)
        - Automatic model downloading from HuggingFace
        - VAE sharing with Wan2GP installation
        - Dedicated upscaling tab in Wan2GP interface
    
    Attributes:
        name (str): Plugin display name
        version (str): Plugin version (semantic versioning)
        description (str): Short plugin description
        current_pipeline: Currently loaded FlashVSR pipeline instance
        models_loaded (bool): Whether models have been downloaded/initialized
    
    Example:
        The plugin is automatically discovered and loaded by Wan2GP's plugin
        system. Users access it via the "FlashVSR Upscaling" tab.
    """
    
    def __init__(self):
        """
        Initialize the FlashVSR plugin.
        
        Sets up plugin metadata and initializes the plugin state.
        Model loading is deferred until first use to minimize startup time.
        """
        super().__init__()
        self.name = "FlashVSR Upscaling"
        self.version = "2.0.0"
        self.description = "AI-powered 4x video upscaling with FlashVSR models (8GB+ VRAM)"
        
        # Plugin state
        self.current_pipeline = None
        self.models_loaded = False
        
        # Load config
        self.config = self.load_config()
        
    def load_config(self):
        """
        Load configuration from config.json file.
        
        Reads the plugin configuration file and merges it with default values.
        Handles missing or corrupted config files gracefully by returning defaults.
        
        Returns:
            dict: Configuration dictionary with all settings
        """
        import json
        from pathlib import Path
        
        # Default configuration - matches FlashVSR_plus defaults
        default_config = {
            "model_variant": "tiny",
            "model_version": "FlashVSR-v1.1",
            "scale_factor": 4,
            "vram_optimization": {
                "tiled_vae": True,
                "tiled_dit": False,
                "tile_size": 256,
                "overlap": 24
            },
            "quality_settings": {
                "color_fix": True,
                "output_quality": 6,
                "output_fps": 30
            },
            "sparse_attention": {
                "sparse_ratio": 2.0,
                "kv_ratio": 3,
                "local_range": 11
            },
            "processing": {
                "dtype": "bf16",
                "unload_dit": False
            }
        }
        
        # Get plugin directory
        plugin_dir = Path(__file__).parent
        config_path = plugin_dir / "config.json"
        
        # Try to load config file
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    
                # Check if config has default field (schema file)
                if "default" in config_data:
                    user_config = config_data["default"]
                else:
                    user_config = config_data
                
                # Merge with defaults (user config takes precedence)
                merged_config = default_config.copy()
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in merged_config:
                        # Deep merge for nested dicts
                        merged_config[key].update(value)
                    else:
                        merged_config[key] = value
                
                print(f"[FlashVSR] Loaded configuration from {config_path}")
                return merged_config
            else:
                print("[FlashVSR] Config file not found, using defaults")
                return default_config
                
        except json.JSONDecodeError as e:
            print(f"[FlashVSR] Warning: Failed to parse config.json: {e}")
            print("[FlashVSR] Using default configuration")
            return default_config
        except Exception as e:
            print(f"[FlashVSR] Warning: Error loading config: {e}")
            print("[FlashVSR] Using default configuration")
            return default_config
    
    def save_config(self, config=None):
        """
        Save configuration to config.json file.
        
        Writes the current plugin configuration to disk for persistence
        across sessions. Creates the config file if it doesn't exist.
        
        Args:
            config: Configuration dictionary to save. If None, uses self.config.
            
        Returns:
            bool: True if save succeeded, False otherwise
        """
        import json
        from pathlib import Path
        
        if config is None:
            config = self.config
        
        # Get plugin directory
        plugin_dir = Path(__file__).parent
        config_path = plugin_dir / "config.json"
        
        try:
            # Read existing file to preserve schema if present
            existing_data = {}
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        existing_data = json.load(f)
                except Exception:
                    pass  # If read fails, we'll create new file
            
            # Check if this is a schema file (has $schema field)
            if "$schema" in existing_data:
                # Update the default field instead of replacing entire file
                existing_data["default"] = config
                data_to_write = existing_data
            else:
                # Just write the config directly
                data_to_write = config
            
            # Write config file
            with open(config_path, 'w') as f:
                json.dump(data_to_write, f, indent=2)
            
            print(f"[FlashVSR] Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            print(f"[FlashVSR] Warning: Failed to save config: {e}")
            return False
    
    def update_config_from_ui(self, **kwargs):
        """
        Update configuration from UI component values.
        
        Extracts settings from UI components and updates the plugin config.
        Automatically saves the updated config to disk.
        
        Args:
            **kwargs: Keyword arguments with setting names and values
            
        Returns:
            dict: Updated configuration dictionary
        """
        # Map UI values to config structure
        if "model_variant" in kwargs:
            variant_map = {
                "Tiny (8-10GB VRAM)": "tiny",
                "Tiny-Long (10-12GB VRAM)": "tiny-long",
                "Full (18-24GB VRAM)": "full"
            }
            self.config["model_variant"] = variant_map.get(kwargs["model_variant"], "tiny")
        
        if "scale_factor" in kwargs:
            self.config["scale_factor"] = int(kwargs["scale_factor"].replace("x", ""))
        
        if "tiled_vae" in kwargs:
            self.config["vram_optimization"]["tiled_vae"] = kwargs["tiled_vae"]
        
        if "tiled_dit" in kwargs:
            self.config["vram_optimization"]["tiled_dit"] = kwargs["tiled_dit"]
        
        if "tile_size" in kwargs:
            self.config["vram_optimization"]["tile_size"] = int(kwargs["tile_size"])
        
        if "overlap" in kwargs:
            self.config["vram_optimization"]["overlap"] = int(kwargs["overlap"])
        
        if "color_fix" in kwargs:
            self.config["quality_settings"]["color_fix"] = kwargs["color_fix"]
        
        if "output_quality" in kwargs:
            self.config["quality_settings"]["output_quality"] = int(kwargs["output_quality"])
        
        if "output_fps" in kwargs:
            self.config["quality_settings"]["output_fps"] = int(kwargs["output_fps"])
        
        if "sparse_ratio" in kwargs:
            self.config["sparse_attention"]["sparse_ratio"] = float(kwargs["sparse_ratio"])
        
        if "kv_ratio" in kwargs:
            self.config["sparse_attention"]["kv_ratio"] = int(kwargs["kv_ratio"])
        
        if "local_range" in kwargs:
            self.config["sparse_attention"]["local_range"] = int(kwargs["local_range"])
        
        if "dtype" in kwargs:
            self.config["processing"]["dtype"] = kwargs["dtype"]
        
        if "unload_dit" in kwargs:
            self.config["processing"]["unload_dit"] = kwargs["unload_dit"]
        
        if "model_version" in kwargs:
            self.config["model_version"] = kwargs["model_version"]
        
        # Save updated config
        self.save_config()
        
        return self.config
    
    def get_config_defaults(self):
        """
        Get default values from config for UI initialization.
        
        Returns a dictionary mapping UI component names to their default
        values from the configuration file.
        
        Returns:
            dict: Default values for UI components
        """
        variant_map = {
            "tiny": "Tiny (8-10GB VRAM)",
            "tiny-long": "Tiny-Long (10-12GB VRAM)",
            "full": "Full (18-24GB VRAM)"
        }
        
        # Handle backwards compatibility for old config format
        sparse_attn = self.config.get("sparse_attention", {})
        quality = self.config.get("quality_settings", {})
        processing = self.config.get("processing", {})
        
        return {
            "model_variant": variant_map.get(self.config["model_variant"], "Tiny (8-10GB VRAM)"),
            "model_version": self.config.get("model_version", "FlashVSR-v1.1"),
            "scale_factor": f"{self.config['scale_factor']}x",
            "tiled_vae": self.config["vram_optimization"]["tiled_vae"],
            "tiled_dit": self.config["vram_optimization"]["tiled_dit"],
            "tile_size": self.config["vram_optimization"]["tile_size"],
            "overlap": self.config["vram_optimization"]["overlap"],
            "color_fix": quality.get("color_fix", True),
            "output_quality": quality.get("output_quality", 6),
            "output_fps": quality.get("output_fps", 30),
            "sparse_ratio": sparse_attn.get("sparse_ratio", 2.0),
            "kv_ratio": sparse_attn.get("kv_ratio", 3),
            "local_range": sparse_attn.get("local_range", 11),
            "dtype": processing.get("dtype", "bf16"),
            "unload_dit": processing.get("unload_dit", False)
        }
        
    def setup_ui(self):
        """
        Setup UI components before the main Wan2GP UI is built.
        
        This method is called during plugin initialization to register
        custom tabs and request access to shared components.
        
        Currently adds a dedicated "FlashVSR Upscaling" tab at position 5.
        """
        # Add dedicated FlashVSR tab
        self.add_tab(
            tab_id="flashvsr_upscaling",
            label="FlashVSR Upscaling",
            component_constructor=self.create_flashvsr_ui,
            position=5  # After main generation tabs
        )
    
    def create_flashvsr_ui(self):
        """
        Create the FlashVSR upscaling tab user interface.
        
        Builds the Gradio UI components for the FlashVSR upscaling functionality.
        Includes all controls for video upscaling with FlashVSR models.
        
        Features:
        - Video file upload with validation
        - Model variant selection (Tiny/Tiny-Long/Full)
        - Scale factor selection (2x/4x)
        - Advanced settings:
          - Tiled VAE/DiT for VRAM optimization
          - Tile size and overlap controls
          - Color correction toggle
          - Sparse attention parameters
        - Progress bar for upscaling operation
        - Output video display with download
        
        Returns:
            gr.Blocks: Gradio Blocks component containing the FlashVSR UI
        """
        # Get default values from config
        defaults = self.get_config_defaults()
        
        with gr.Blocks() as demo:
            gr.Markdown("""
            ## FlashVSR Video Upscaling
            
            Upload a video and upscale it using AI-powered FlashVSR models.
            
            **Features:**
            - 4x upscaling (2x also supported)
            - Support for 8GB+ VRAM GPUs
            - Automatic model downloading from HuggingFace
            - Tile-based processing for low VRAM scenarios
            """)
            
            with gr.Row():
                # Left column - Input controls
                with gr.Column(scale=1):
                    gr.Markdown("### Input Settings")
                    
                    video_input = gr.File(
                        label="Input Video",
                        file_types=["video"],
                        elem_id="flashvsr_input_video"
                    )
                    
                    with gr.Row():
                        model_variant = gr.Dropdown(
                            choices=[
                                "Tiny (8-10GB VRAM)",
                                "Tiny-Long (10-12GB VRAM)", 
                                "Full (18-24GB VRAM)"
                            ],
                            value=defaults["model_variant"],
                            label="Model Variant",
                            info="Tiny recommended for most users",
                            elem_id="flashvsr_model_variant"
                        )
                    
                    with gr.Row():
                        scale_factor = gr.Dropdown(
                            choices=["2x", "4x"],
                            value=defaults["scale_factor"],
                            label="Scale Factor",
                            info="4x recommended (native FlashVSR)",
                            elem_id="flashvsr_scale_factor"
                        )
                    
                    # Advanced Settings Accordion
                    with gr.Accordion("Advanced Settings", open=False):
                        gr.Markdown("#### VRAM Optimization")
                        
                        tiled_vae = gr.Checkbox(
                            label="Tiled VAE",
                            value=defaults["tiled_vae"],
                            info="Enable for high resolution (>1080p)",
                            elem_id="flashvsr_tiled_vae"
                        )
                        
                        tiled_dit = gr.Checkbox(
                            label="Tiled DiT",
                            value=defaults["tiled_dit"],
                            info="Required for 8GB GPUs at 1080p",
                            elem_id="flashvsr_tiled_dit"
                        )
                        
                        with gr.Row():
                            tile_size = gr.Slider(
                                minimum=128,
                                maximum=512,
                                value=defaults["tile_size"],
                                step=64,
                                label="Tile Size",
                                info="Smaller = less VRAM, slower",
                                elem_id="flashvsr_tile_size"
                            )
                            
                            overlap = gr.Slider(
                                minimum=8,
                                maximum=64,
                                value=defaults["overlap"],
                                step=8,
                                label="Tile Overlap (px)",
                                info="Reduces seam artifacts",
                                elem_id="flashvsr_overlap"
                            )
                        
                        gr.Markdown("#### Quality Settings")
                        
                        color_fix = gr.Checkbox(
                            label="Enable Color Fix",
                            value=defaults["color_fix"],
                            info="Wavelet-based color correction",
                            elem_id="flashvsr_color_fix"
                        )
                        
                        with gr.Row():
                            output_quality = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=defaults["output_quality"],
                                step=1,
                                label="Output Video Quality",
                                info="Higher = better quality, larger file",
                                elem_id="flashvsr_output_quality"
                            )
                            
                            output_fps = gr.Number(
                                value=defaults["output_fps"],
                                label="Output FPS",
                                info="Fallback when video metadata unavailable",
                                precision=0,
                                elem_id="flashvsr_output_fps"
                            )
                        
                        gr.Markdown("#### Processing Settings")
                        
                        with gr.Row():
                            dtype = gr.Radio(
                                choices=["fp16", "bf16"],
                                value=defaults["dtype"],
                                label="Data Type",
                                info="bf16 recommended for most GPUs",
                                elem_id="flashvsr_dtype"
                            )
                            
                            unload_dit = gr.Checkbox(
                                label="Unload DiT before Decoding",
                                value=defaults["unload_dit"],
                                info="Saves VRAM during decode",
                                elem_id="flashvsr_unload_dit"
                            )
                        
                        gr.Markdown("#### Sparse Attention Parameters")
                        
                        with gr.Row():
                            sparse_ratio = gr.Slider(
                                minimum=0.5,
                                maximum=5.0,
                                value=defaults["sparse_ratio"],
                                step=0.1,
                                label="Sparse Ratio",
                                info="Controls attention sparsity; smaller = more sparse",
                                elem_id="flashvsr_sparse_ratio"
                            )
                            
                            kv_ratio = gr.Slider(
                                minimum=1,
                                maximum=8,
                                value=defaults["kv_ratio"],
                                step=1,
                                label="KV Cache Ratio",
                                info="Controls the length of the KV cache",
                                elem_id="flashvsr_kv_ratio"
                            )
                        
                        local_range = gr.Slider(
                            minimum=3,
                            maximum=15,
                            value=defaults["local_range"],
                            step=2,
                            label="Local Range",
                            info="Size of the local attention window",
                            elem_id="flashvsr_local_range"
                        )
                        
                        gr.Markdown("#### Model Version")
                        
                        model_version = gr.Radio(
                            choices=["FlashVSR", "FlashVSR-v1.1"],
                            value=defaults.get("model_version", "FlashVSR-v1.1"),
                            label="Model Version",
                            info="FlashVSR-v1.1 uses causal attention for better temporal consistency",
                            elem_id="flashvsr_model_version"
                        )
                    
                    # Upscale button with progress
                    upscale_btn = gr.Button(
                        "🚀 Upscale Video",
                        variant="primary",
                        size="lg",
                        elem_id="flashvsr_upscale_btn"
                    )
                    
                    progress_bar = gr.Progress()
                
                # Right column - Output and status
                with gr.Column(scale=1):
                    gr.Markdown("### Output")
                    
                    video_output = gr.Video(
                        label="Upscaled Video",
                        elem_id="flashvsr_output_video"
                    )
                    
                    status_text = gr.Textbox(
                        label="Status",
                        value="Ready to upscale. Upload a video to begin.",
                        interactive=False,
                        lines=3,
                        elem_id="flashvsr_status"
                    )
                    
                    # Info box with VRAM estimates
                    vram_info = gr.Markdown(
                        """
                        **VRAM Estimates:**
                        - Tiny: 8-10GB for 1080p
                        - Tiny-Long: 10-12GB for long videos
                        - Full: 18-24GB for highest quality
                        
                        Enable Tiled DiT for 8GB GPUs.
                        """,
                        elem_id="flashvsr_vram_info"
                    )
            
            # Event handler - implements full upscaling functionality
            def upscale_video(
                video, variant, scale, t_vae, t_dit, 
                t_size, t_overlap, c_fix, out_quality, out_fps,
                data_type, do_unload_dit, sparse_r, kv, local_r,
                model_ver,
                progress=gr.Progress()
            ):
                """
                Upscale a video using FlashVSR models.
                
                Args:
                    video: Gradio File object containing the input video
                    variant: Model variant selection string
                    scale: Scale factor ("2x" or "4x")
                    t_vae: Enable tiled VAE
                    t_dit: Enable tiled DiT
                    t_size: Tile size for tiled processing
                    t_overlap: Tile overlap in pixels
                    c_fix: Enable color correction
                    out_quality: Output video quality (1-10)
                    out_fps: Fallback FPS when video metadata unavailable
                    data_type: Data type ("fp16" or "bf16")
                    do_unload_dit: Unload DiT before decoding
                    sparse_r: Sparse ratio for attention (0.5-5.0)
                    kv: KV cache ratio (1-8)
                    local_r: Local attention range (3-15)
                    progress: Gradio progress tracker
                
                Returns:
                    Tuple of (output_video_path, status_message)
                """
                import os
                import torch
                import imageio
                import numpy as np
                import ffmpeg
                from pathlib import Path
                
                # Validation
                if video is None:
                    return None, "❌ Please upload a video first."
                
                try:
                    progress(0, desc="Initializing...")
                    
                    # Parse variant
                    variant_map = {
                        "Tiny (8-10GB VRAM)": "tiny",
                        "Tiny-Long (10-12GB VRAM)": "tiny-long",
                        "Full (18-24GB VRAM)": "full"
                    }
                    selected_variant = variant_map.get(variant, "tiny")
                    is_tiny_long = (selected_variant == "tiny-long")
                    
                    # Parse scale factor
                    scale_factor = int(scale.replace("x", ""))
                    
                    # Determine dtype based on user selection
                    if data_type == "bf16" and torch.cuda.is_bf16_supported():
                        torch_dtype = torch.bfloat16
                    else:
                        torch_dtype = torch.float16
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    
                    if device == "cpu":
                        return None, "❌ CUDA GPU required for FlashVSR upscaling."
                    
                    # Import helper functions from download_manager
                    from .src.models.download_manager import load_pipeline
                    
                    progress(0.05, desc="Loading input video...")
                    
                    # Load input video
                    video_path = video.name if hasattr(video, 'name') else str(video)
                    
                    try:
                        reader = imageio.get_reader(video_path)
                        meta = reader.get_meta_data()
                        fps = int(round(meta.get('fps', out_fps)))
                        
                        # Load all frames
                        frames = []
                        for frame_data in reader:
                            frame_np = frame_data.astype(np.float32) / 255.0
                            frames.append(torch.from_numpy(frame_np).to(torch_dtype))
                        
                        reader.close()
                        
                        if len(frames) < 21:
                            return None, f"❌ Video must have at least 21 frames. Got {len(frames)} frames."
                        
                        video_tensor = torch.stack(frames, 0)  # Shape: (N, H, W, C)
                        
                    except Exception as e:
                        return None, f"❌ Error loading video: {str(e)}"
                    
                    progress(0.15, desc=f"Loading {selected_variant.upper()} pipeline (model: {model_ver})...")
                    
                    # Load pipeline (this will download models if needed)
                    # IMPORTANT: Reinitialize fresh each time to avoid state leakage (matches upstream)
                    try:
                        # Clean up any existing pipeline to avoid memory issues
                        if self.current_pipeline is not None:
                            del self.current_pipeline
                            self.current_pipeline = None
                            clean_vram()
                        
                        pipeline = load_pipeline(
                            variant=selected_variant,
                            device=device,
                            torch_dtype=torch_dtype,
                            model_version=model_ver
                        )
                        self.current_pipeline = pipeline
                        self.models_loaded = True
                    except Exception as e:
                        return None, f"❌ Error loading pipeline: {str(e)}"
                    
                    progress(0.25, desc="Preparing input frames...")
                    
                    # Prepare input tensor - matches upstream FlashVSR_plus approach
                    # Frame padding to ensure 8n+5 alignment for the pipeline
                    frame_count = int(video_tensor.shape[0])
                    N0, h0, w0, _ = video_tensor.shape

                    pad_to = next_8n5(frame_count)
                    add = pad_to - frame_count
                    if add > 0:
                        padding_frames = video_tensor[-1:, :, :, :].repeat(add, 1, 1, 1)
                        video_tensor = torch.cat([video_tensor, padding_frames], dim=0)
                    
                    # Clean VRAM before processing
                    clean_vram()
                    print(f"[FlashVSR] Processing {frame_count} frames...")
                    
                    progress(0.35, desc="Running upscaling inference...")
                    
                    # Build common pipe_kwargs matching upstream FlashVSR_plus
                    # Note: color_fix is handled differently for tiled vs non-tiled modes
                    pipe_kwargs = {
                        "prompt": "",
                        "negative_prompt": "",
                        "cfg_scale": 1.0,
                        "num_inference_steps": 1,
                        "seed": 0,
                        "tiled": t_vae,
                        "is_full_block": False,
                        "if_buffer": True,
                        "kv_ratio": int(kv),
                        "local_range": int(local_r),
                        "unload_dit": False,  # Don't unload between tiles
                        "fps": fps,  # CRITICAL: Pass fps for temporal consistency
                    }
                    
                    final_output_tensor = None
                    output_frames = None
                    output_written_directly = False  # Flag for Tiny-Long direct file output
                    
                    # Prepare output path early (needed for Tiny-Long mode)
                    output_dir = Path("outputs") / "flashvsr"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    import time as time_module
                    timestamp = time_module.strftime("%Y%m%d-%H%M%S")
                    output_filename = f"flashvsr_{selected_variant}_{scale}_{timestamp}.mp4"
                    output_path = output_dir / output_filename
                    
                    try:
                        if t_dit:
                            # ============================================================
                            # TILED DiT PROCESSING - matches upstream FlashVSR_plus
                            # ============================================================
                            N, H, W, C = video_tensor.shape
                            progress(0.35, desc=f"Initializing tiled processing (tile_size={t_size}, overlap={t_overlap})...")
                            
                            # Validate overlap
                            if t_overlap > t_size / 2:
                                return None, "❌ Overlap must be less than half of the tile size!"
                            
                            # Calculate tile coordinates at ORIGINAL resolution
                            tile_coords = calculate_tile_coords(H, W, t_size, t_overlap)
                            num_tiles = len(tile_coords)
                            
                            print(f"[FlashVSR] Tile-DiT: Processing {num_tiles} tiles at {W}x{H} (output: {W*scale_factor}x{H*scale_factor})")
                            
                            from tqdm import tqdm as tqdm_progress
                            
                            # Add color_fix to pipe_kwargs for tiled processing
                            tile_pipe_kwargs = {**pipe_kwargs, "color_fix": c_fix}
                            
                            if is_tiny_long:
                                # ============================================================
                                # TINY-LONG TILED MODE: Write each tile to temp file, then stitch
                                # ============================================================
                                import tempfile
                                import uuid
                                
                                temp_dir = Path(tempfile.gettempdir()) / f"flashvsr_tiles_{uuid.uuid4().hex}"
                                temp_dir.mkdir(parents=True, exist_ok=True)
                                temp_videos = []
                                
                                for tile_idx, (x1, y1, x2, y2) in enumerate(tqdm_progress(tile_coords, desc="[FlashVSR] Processing tiles")):
                                    progress(
                                        0.35 + 0.40 * (tile_idx / num_tiles),
                                        desc=f"Processing tile {tile_idx+1}/{num_tiles}"
                                    )
                                    
                                    # Extract tile from ORIGINAL frames
                                    input_tile = video_tensor[:, y1:y2, x1:x2, :]
                                    
                                    # Get input parameters for this tile
                                    th, tw, F = get_input_params(input_tile, scale=scale_factor)
                                    
                                    # Use generator for memory-efficient processing
                                    LQ_tile = input_tensor_generator(input_tile, device, scale=scale_factor, dtype=torch_dtype)
                                    
                                    # Temp output path for this tile
                                    temp_name = str(temp_dir / f"{tile_idx+1:05d}.mp4")
                                    
                                    # Calculate topk_ratio for this tile's resolution
                                    topk_ratio_tile = sparse_r * 768 * 1280 / (th * tw)
                                    
                                    # Run pipeline on tile - writes directly to temp file
                                    result = pipeline(
                                        LQ_video=LQ_tile,
                                        num_frames=F,
                                        height=th,
                                        width=tw,
                                        topk_ratio=topk_ratio_tile,
                                        output_path=temp_name,
                                        quality=int(out_quality),
                                        **tile_pipe_kwargs
                                    )
                                    
                                    temp_videos.append(temp_name)
                                    
                                    # Clean up
                                    del input_tile
                                    clean_vram()
                                
                                progress(0.75, desc="Stitching tiles...")
                                
                                # Stitch all tiles together
                                stitch_video_tiles(
                                    tile_paths=temp_videos,
                                    tile_coords=tile_coords,
                                    final_dims=(W * scale_factor, H * scale_factor),
                                    scale=scale_factor,
                                    overlap=t_overlap,
                                    output_path=str(output_path),
                                    fps=fps,
                                    quality=int(out_quality),
                                    cleanup=True
                                )
                                
                                # Clean up temp directory
                                import shutil
                                try:
                                    shutil.rmtree(temp_dir)
                                except:
                                    pass
                                
                                output_written_directly = True
                                print("[FlashVSR] Tile-DiT processing complete (Tiny-Long mode).")
                            
                            else:
                                # ============================================================
                                # STANDARD TILED MODE: Accumulate in memory, then save
                                # ============================================================
                                num_aligned_frames = largest_8n1_leq(N + 4) - 4
                                
                                # Create output canvas at SCALED resolution
                                final_output_canvas = torch.zeros(
                                    (num_aligned_frames, H * scale_factor, W * scale_factor, C),
                                    dtype=torch.float32
                                )
                                weight_sum_canvas = torch.zeros_like(final_output_canvas)
                                
                                for tile_idx, (x1, y1, x2, y2) in enumerate(tqdm_progress(tile_coords, desc="[FlashVSR] Processing tiles")):
                                    progress(
                                        0.35 + 0.50 * (tile_idx / num_tiles),
                                        desc=f"Processing tile {tile_idx+1}/{num_tiles}"
                                    )
                                    
                                    # Extract tile from ORIGINAL frames (not upscaled)
                                    input_tile = video_tensor[:, y1:y2, x1:x2, :]
                                    
                                    # Prepare the tile for the pipeline (bicubic upscale + normalize)
                                    LQ_tile, th, tw, F = prepare_input_tensor(
                                        input_tile, device, scale=scale_factor, dtype=torch_dtype
                                    )
                                    LQ_tile = LQ_tile.to(device)
                                    
                                    # Calculate topk_ratio for this tile's resolution
                                    topk_ratio_tile = sparse_r * 768 * 1280 / (th * tw)
                                    
                                    # Run pipeline on tile
                                    output_tile_gpu = pipeline(
                                        LQ_video=LQ_tile,
                                        num_frames=F,
                                        height=th,
                                        width=tw,
                                        topk_ratio=topk_ratio_tile,
                                        **tile_pipe_kwargs
                                    )
                                    
                                    # Check for pipeline error (returns boolean on failure)
                                    if not isinstance(output_tile_gpu, torch.Tensor):
                                        raise RuntimeError(f"Pipeline returned {type(output_tile_gpu).__name__} instead of tensor. This may indicate an incompatible pipeline variant or internal error.")
                                    
                                    # Convert output tile to video frames format
                                    processed_tile_cpu = tensor2video(output_tile_gpu).cpu()
                                    
                                    # Create feather mask for blending at SCALED resolution
                                    tile_out_h, tile_out_w = processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]
                                    mask = create_feather_mask(
                                        (tile_out_h, tile_out_w),
                                        t_overlap * scale_factor
                                    ).cpu()
                                    # Reshape mask for broadcasting: (1, 1, H, W) -> (1, H, W, 1)
                                    mask = mask.permute(0, 2, 3, 1)
                                    
                                    # Calculate output coordinates at SCALED resolution
                                    x1_s, y1_s = x1 * scale_factor, y1 * scale_factor
                                    x2_s = x1_s + tile_out_w
                                    y2_s = y1_s + tile_out_h
                                    
                                    # Accumulate weighted tile into canvas
                                    actual_frames = processed_tile_cpu.shape[0]
                                    canvas_frames = final_output_canvas.shape[0]
                                    use_frames = min(actual_frames, canvas_frames)
                                    
                                    final_output_canvas[:use_frames, y1_s:y2_s, x1_s:x2_s, :] += processed_tile_cpu[:use_frames] * mask
                                    weight_sum_canvas[:use_frames, y1_s:y2_s, x1_s:x2_s, :] += mask
                                    
                                    # Clean up tile to free VRAM
                                    del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile, mask
                                    clean_vram()
                                
                                # Normalize by weight sum
                                weight_sum_canvas[weight_sum_canvas == 0] = 1.0
                                final_output_tensor = final_output_canvas / weight_sum_canvas
                                
                                # Trim to original frame count (before padding)
                                output_frames = final_output_tensor[:frame_count]
                                
                                print("[FlashVSR] Tile-DiT processing complete.")
                            
                            # Clean up pipeline if requested
                            if do_unload_dit and hasattr(pipeline, 'offload_model'):
                                pipeline.offload_model(keep_vae=True)
                        
                        else:
                            # ============================================================
                            # STANDARD (NON-TILED) PROCESSING
                            # ============================================================
                            # Get input parameters
                            tH, tW, F = get_input_params(video_tensor, scale_factor)
                            
                            # Calculate topk_ratio
                            topk_ratio_adjusted = sparse_r * 768 * 1280 / (tH * tW)
                            
                            # Add color_fix and unload_dit for non-tiled mode
                            full_pipe_kwargs = {
                                **pipe_kwargs,
                                "color_fix": c_fix,
                                "unload_dit": do_unload_dit,
                            }
                            
                            if is_tiny_long:
                                # ============================================================
                                # TINY-LONG NON-TILED: Write directly to output file
                                # ============================================================
                                # Use generator for memory-efficient processing
                                LQ_video = input_tensor_generator(video_tensor, device, scale=scale_factor, dtype=torch_dtype)
                                
                                # Run pipeline with output_path - writes directly to file
                                result = pipeline(
                                    LQ_video=LQ_video,
                                    num_frames=F,
                                    height=tH,
                                    width=tW,
                                    topk_ratio=topk_ratio_adjusted,
                                    output_path=str(output_path),
                                    quality=int(out_quality),
                                    **full_pipe_kwargs
                                )
                                
                                if result == False:
                                    raise RuntimeError("Pipeline returned False, indicating an error during processing. Check console for details.")
                                
                                output_written_directly = True
                                print("[FlashVSR] Processing complete (Tiny-Long mode).")
                            
                            else:
                                # ============================================================
                                # STANDARD NON-TILED: Process in memory, then save
                                # ============================================================
                                # Prepare full-frame input tensor
                                LQ_video, tH, tW, F = prepare_input_tensor(
                                    video_tensor, device, scale=scale_factor, dtype=torch_dtype
                                )
                                LQ_video = LQ_video.to(device)
                                
                                # Run full pipeline
                                output_tensor = pipeline(
                                    LQ_video=LQ_video,
                                    num_frames=F,
                                    height=tH,
                                    width=tW,
                                    topk_ratio=topk_ratio_adjusted,
                                    **full_pipe_kwargs
                                )
                                
                                # Check for pipeline error (returns boolean on failure)
                                if not isinstance(output_tensor, torch.Tensor):
                                    raise RuntimeError(f"Pipeline returned {type(output_tensor).__name__} instead of tensor. Check console for error details.")
                                
                                # Convert output to video frames
                                output_frames = tensor2video(output_tensor).cpu()
                                # Trim to original frame count
                                output_frames = output_frames[:frame_count]
                            
                            del pipeline
                            clean_vram()
                            
                    except Exception as e:
                        import traceback
                        error_trace = traceback.format_exc()
                        print(f"[FlashVSR] Error during upscaling: {error_trace}")
                        return None, f"❌ Error during upscaling: {str(e)}"
                    
                    progress(0.85, desc="Saving output video...")
                    
                    # Save video (skip if Tiny-Long already wrote directly)
                    if not output_written_directly:
                        # output_frames is in (F, H, W, C) format in [0, 1] range
                        # Get output dimensions for status message
                        out_h, out_w = output_frames.shape[1], output_frames.shape[2]
                        
                        # Write frames to video with user-specified quality
                        frames_np = (output_frames.cpu().float() * 255.0).clip(0, 255).numpy().astype(np.uint8)
                        writer = imageio.get_writer(str(output_path), fps=fps, quality=int(out_quality))
                        
                        from tqdm import tqdm as tqdm_save
                        for frame_np in tqdm_save(frames_np, desc="[FlashVSR] Saving video"):
                            writer.append_data(frame_np)
                        
                        writer.close()
                    
                    # Get output dimensions from file for status message
                    try:
                        probe = ffmpeg.probe(str(output_path))
                        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                        out_w = int(video_stream['width'])
                        out_h = int(video_stream['height'])
                    except:
                        out_w, out_h = 0, 0  # Fallback if probe fails
                    
                    progress(0.95, desc="Merging audio...")
                    
                    # Try to merge audio from source
                    try:
                        probe = ffmpeg.probe(video_path)
                        audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
                        
                        if audio_streams:
                            temp_path = str(output_path) + "_temp.mp4"
                            os.rename(str(output_path), temp_path)
                            
                            input_video = ffmpeg.input(temp_path)['v']
                            input_audio = ffmpeg.input(video_path)['a']
                            
                            ffmpeg.output(
                                input_video, input_audio, str(output_path),
                                vcodec='copy', acodec='copy'
                            ).run(overwrite_output=True, quiet=True)
                            
                            os.remove(temp_path)
                    except Exception as e:
                        # Audio merge failed, but video is still usable
                        print(f"[FlashVSR] Warning: Audio merge failed: {e}")
                    
                    progress(1.0, desc="Complete!")
                    
                    # Get output frame count for status
                    if output_written_directly:
                        # For Tiny-Long, get frame count from output file
                        try:
                            probe = ffmpeg.probe(str(output_path))
                            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                            output_frame_count = int(video_stream.get('nb_frames', frame_count))
                        except:
                            output_frame_count = frame_count
                    else:
                        output_frame_count = len(frames_np)
                    
                    # Generate status message
                    status = f"""
                    ✅ Upscaling complete!
                    
                    **Configuration:**
                    - Model: {selected_variant.upper()}
                    - Scale: {scale}
                    - Input: {frame_count} frames @ {w0}x{h0}
                    - Output: {output_frame_count} frames @ {out_w}x{out_h}
                    - FPS: {fps}
                    
                    **Settings:**
                    - Tiled VAE: {'Yes' if t_vae else 'No'}
                    - Tiled DiT: {'Yes' if t_dit else 'No'}
                    - Color Fix: {'Yes' if c_fix else 'No'}
                    
                    **Output:** {output_filename}
                    """
                    
                    return str(output_path), status
                    
                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    return None, f"❌ Error: {str(e)}\n\nTraceback:\n{error_trace}"
            
            # Wire up event
            upscale_btn.click(
                fn=upscale_video,
                inputs=[
                    video_input, model_variant, scale_factor,
                    tiled_vae, tiled_dit, tile_size, overlap,
                    color_fix, output_quality, output_fps,
                    dtype, unload_dit, sparse_ratio, kv_ratio, local_range,
                    model_version
                ],
                outputs=[video_output, status_text]
            )
            
            # Save config when settings change
            def save_settings(variant, scale, t_vae, t_dit, t_size, t_overlap, 
                            c_fix, out_qual, out_fps, data_type, do_unload, 
                            sparse_r, kv, local_r, model_ver):
                """Save current UI settings to config file"""
                self.update_config_from_ui(
                    model_variant=variant,
                    scale_factor=scale,
                    tiled_vae=t_vae,
                    tiled_dit=t_dit,
                    tile_size=t_size,
                    overlap=t_overlap,
                    color_fix=c_fix,
                    output_quality=out_qual,
                    output_fps=out_fps,
                    dtype=data_type,
                    unload_dit=do_unload,
                    sparse_ratio=sparse_r,
                    kv_ratio=kv,
                    local_range=local_r,
                    model_version=model_ver
                )
                return None
            
            # Attach change handlers to save config (debounced via change event)
            for component in [model_variant, scale_factor, tiled_vae, tiled_dit, 
                            tile_size, overlap, color_fix, output_quality, output_fps,
                            dtype, unload_dit, sparse_ratio, kv_ratio, local_range,
                            model_version]:
                component.change(
                    fn=save_settings,
                    inputs=[
                        model_variant, scale_factor, tiled_vae, tiled_dit, 
                        tile_size, overlap, color_fix, output_quality, output_fps,
                        dtype, unload_dit, sparse_ratio, kv_ratio, local_range,
                        model_version
                    ],
                    outputs=None
                )
        
        return demo
    
    def post_ui_setup(self, components: dict):
        """
        Perform post-UI setup after the main Wan2GP UI is built.
        
        This method is called after all UI components are created and allows
        the plugin to:
        - Access and wire events to existing components
        - Inject new UI elements into existing layouts
        - Configure cross-component interactions
        
        Args:
            components: Dictionary of Gradio components from the main UI,
                       keyed by their elem_id values
        
        Returns:
            dict: Empty dictionary (no components to expose currently)
        """
        return {}
    
    def on_tab_select(self, state):
        """
        Handle FlashVSR tab selection event.
        
        Called when the user navigates to the FlashVSR Upscaling tab.
        Pre-loads models to reduce first-upscale latency and prepares GPU resources.
        
        Args:
            state: Current application state (from Gradio)
        """
        # Check if we have a pipeline loaded
        if self.current_pipeline is not None:
            try:
                print("[FlashVSR] Tab selected - loading models to GPU...")
                
                # Move pipeline models to GPU
                if hasattr(self.current_pipeline, 'load_models_to_device'):
                    self.current_pipeline.load_models_to_device(['dit', 'vae', 'TCDecoder'])
                
                # Re-initialize cross-attention KV cache if it was offloaded
                if hasattr(self.current_pipeline, 'prompt_emb_posi'):
                    if self.current_pipeline.prompt_emb_posi is not None:
                        if self.current_pipeline.prompt_emb_posi.get('stats') == 'offload':
                            context = self.current_pipeline.prompt_emb_posi.get('context')
                            if context is not None:
                                print("[FlashVSR] Re-initializing cross-attention KV cache...")
                                self.current_pipeline.init_cross_kv(context_tensor=context)
                
                # Move LQ_proj_in to GPU if it exists
                if hasattr(self.current_pipeline, 'dit') and self.current_pipeline.dit is not None:
                    if hasattr(self.current_pipeline.dit, 'LQ_proj_in') and self.current_pipeline.dit.LQ_proj_in is not None:
                        device = self.current_pipeline.device
                        self.current_pipeline.dit.LQ_proj_in.to(device)
                
                # Move TCDecoder to GPU
                if hasattr(self.current_pipeline, 'TCDecoder') and self.current_pipeline.TCDecoder is not None:
                    device = self.current_pipeline.device
                    self.current_pipeline.TCDecoder.to(device)
                
                print("[FlashVSR] Models loaded to GPU. Ready for upscaling.")
                
            except Exception as e:
                print(f"[FlashVSR] Warning: Failed to pre-load models on tab select: {e}")
                # Non-critical error, models will load on first upscale anyway
    
    def on_tab_deselect(self, state):
        """
        Handle FlashVSR tab deselection event.
        
        Called when the user navigates away from the FlashVSR Upscaling tab.
        Offloads models to CPU to free VRAM for other Wan2GP operations.
        
        Args:
            state: Current application state (from Gradio)
        """
        import torch
        
        # Check if we have a pipeline loaded
        if self.current_pipeline is not None:
            try:
                print("[FlashVSR] Tab deselected - offloading models to CPU to free VRAM...")
                
                # Get current VRAM usage before offload
                vram_before = 0.0
                if torch.cuda.is_available():
                    vram_before = torch.cuda.memory_allocated() / 1024**3  # GB
                    print(f"[FlashVSR] VRAM before offload: {vram_before:.2f} GB")
                
                # Offload pipeline models to CPU
                if hasattr(self.current_pipeline, 'offload_model'):
                    self.current_pipeline.offload_model(keep_vae=False)
                else:
                    # Manual offload if method doesn't exist
                    if hasattr(self.current_pipeline, 'dit') and self.current_pipeline.dit is not None:
                        if hasattr(self.current_pipeline.dit, 'clear_cross_kv'):
                            self.current_pipeline.dit.clear_cross_kv()
                        self.current_pipeline.dit.to('cpu')
                    
                    if hasattr(self.current_pipeline, 'vae') and self.current_pipeline.vae is not None:
                        self.current_pipeline.vae.to('cpu')
                    
                    if hasattr(self.current_pipeline, 'TCDecoder') and self.current_pipeline.TCDecoder is not None:
                        self.current_pipeline.TCDecoder.to('cpu')
                    
                    # Update status
                    if hasattr(self.current_pipeline, 'prompt_emb_posi'):
                        if self.current_pipeline.prompt_emb_posi is not None:
                            self.current_pipeline.prompt_emb_posi['stats'] = 'offload'
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    vram_after = torch.cuda.memory_allocated() / 1024**3  # GB
                    freed = vram_before - vram_after
                    print(f"[FlashVSR] VRAM after offload: {vram_after:.2f} GB (freed {freed:.2f} GB)")
                
                print("[FlashVSR] Models offloaded to CPU. VRAM freed for other tasks.")
                
            except Exception as e:
                print(f"[FlashVSR] Warning: Failed to offload models on tab deselect: {e}")
                # Non-critical error, but VRAM may not be freed
