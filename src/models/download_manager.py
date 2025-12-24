"""
Model download and management utilities for FlashVSR plugin.
Handles automatic model downloads from HuggingFace and local caching.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from pathlib import Path


def get_ckpts_dir():
    """Get the ckpts directory from Wan2GP installation."""
    # Navigate up from plugin directory to Wan2GP root
    plugin_dir = Path(__file__).parent.parent.parent
    wan2gp_root = plugin_dir.parent.parent
    ckpts_dir = wan2gp_root / "ckpts"
    return ckpts_dir


def get_flashvsr_model_dir():
    """Get the FlashVSR model directory path."""
    ckpts_dir = get_ckpts_dir()
    flashvsr_dir = ckpts_dir / "flashvsr"
    return flashvsr_dir


def ensure_models_downloaded(model_version: str = "FlashVSR-v1.1", force_download: bool = False):
    """
    Ensure FlashVSR models are downloaded from HuggingFace.
    
    Args:
        model_version: Model version - "FlashVSR" or "FlashVSR-v1.1" (default: FlashVSR-v1.1)
        force_download: Force re-download even if files exist
        
    Returns:
        Path to model directory
    """
    from huggingface_hub import snapshot_download
    
    # Construct HuggingFace repo ID from model version
    model_name = f"JunhaoZhuang/{model_version}"
    
    model_dir = get_flashvsr_model_dir()
    
    # Check if already downloaded
    if model_dir.exists() and not force_download:
        # Basic validation - check for key files
        required_files = [
            "dit.safetensors",
            "config.json"
        ]
        
        all_exist = all((model_dir / f).exists() for f in required_files)
        
        if all_exist:
            print(f"[FlashVSR] Models already downloaded at: {model_dir}")
            return str(model_dir)
        else:
            print(f"[FlashVSR] Incomplete model files detected, re-downloading...")
    
    # Create directory if needed
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Download models
    print(f"[FlashVSR] Downloading models from HuggingFace ({model_name})...")
    print(f"[FlashVSR] This may take several minutes depending on your internet connection.")
    
    try:
        # Download to model version specific subdirectory for proper caching
        snapshot_download(
            repo_id=model_name,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"[FlashVSR] Models downloaded successfully to: {model_dir}")
    except Exception as e:
        print(f"[FlashVSR] Error downloading models: {str(e)}")
        raise RuntimeError(f"Failed to download FlashVSR models: {str(e)}")
    
    return str(model_dir)


def get_vae_path() -> Optional[str]:
    """
    Get path to Wan2.1 VAE, checking both .safetensors and .pth formats.
    
    Returns:
        Path to VAE file if found, None otherwise
    """
    ckpts_dir = get_ckpts_dir()
    
    # Check for .safetensors first (Wan2GP format)
    safetensors_path = ckpts_dir / "Wan2.1_VAE.safetensors"
    if safetensors_path.exists():
        print(f"[FlashVSR] Found Wan2.1 VAE (safetensors): {safetensors_path}")
        return str(safetensors_path)
    
    # Check for .pth format (FlashVSR native format)
    pth_path = ckpts_dir / "flashvsr" / "vae.pth"
    if pth_path.exists():
        print(f"[FlashVSR] Found Wan2.1 VAE (pth): {pth_path}")
        return str(pth_path)
    
    # Check alternative .pth location
    alt_pth_path = ckpts_dir / "Wan2.1_VAE.pth"
    if alt_pth_path.exists():
        print(f"[FlashVSR] Found Wan2.1 VAE (pth): {alt_pth_path}")
        return str(alt_pth_path)
    
    return None


def convert_vae_safetensors_to_pth(safetensors_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert VAE from .safetensors to .pth format for FlashVSR compatibility.
    
    Args:
        safetensors_path: Path to .safetensors file
        output_path: Optional output path (defaults to same location with .pth extension)
        
    Returns:
        Path to converted .pth file
    """
    from safetensors.torch import load_file
    
    if output_path is None:
        output_path = str(Path(safetensors_path).with_suffix('.pth'))
    
    print(f"[FlashVSR] Converting VAE from safetensors to pth format...")
    print(f"[FlashVSR] Input: {safetensors_path}")
    print(f"[FlashVSR] Output: {output_path}")
    
    try:
        # Load from safetensors
        state_dict = load_file(safetensors_path)
        
        # Save as .pth
        torch.save(state_dict, output_path)
        
        print(f"[FlashVSR] Conversion complete!")
        return output_path
    except Exception as e:
        print(f"[FlashVSR] Error converting VAE: {str(e)}")
        raise RuntimeError(f"Failed to convert VAE format: {str(e)}")


def get_posi_prompt_path() -> str:
    """
    Get path to bundled posi_prompt.pth file.
    
    Returns:
        Path to posi_prompt.pth
        
    Raises:
        FileNotFoundError if file is missing
    """
    # Path(__file__) is src/models/download_manager.py
    # .parent is src/models/, .parent.parent is src/, .parent.parent.parent is plugin root
    plugin_dir = Path(__file__).parent.parent.parent
    prompt_path = plugin_dir / "prompt" / "posi_prompt.pth"
    
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"posi_prompt.pth not found at: {prompt_path}\n"
            "This file should be bundled with the plugin."
        )
    
    return str(prompt_path)


def get_model_paths(variant: str = "tiny", model_version: str = "FlashVSR-v1.1") -> Dict[str, str]:
    """
    Get all required model file paths for the specified pipeline variant.
    
    Args:
        variant: Pipeline variant - "tiny", "tiny-long", or "full"
        model_version: Model version - "FlashVSR" or "FlashVSR-v1.1"
        
    Returns:
        Dictionary with paths to all required model files
        
    Raises:
        FileNotFoundError if required files are missing
    """
    variant = variant.lower()
    
    # Ensure models are downloaded
    model_dir = Path(ensure_models_downloaded(model_version=model_version))
    
    # Map to actual filenames from HuggingFace repo
    dit_path = model_dir / "diffusion_pytorch_model_streaming_dmd.safetensors"
    lq_proj_path = model_dir / "LQ_proj_in.ckpt"
    
    paths = {
        "dit": str(dit_path),
        "lq_proj": str(lq_proj_path),
        "posi_prompt": get_posi_prompt_path()
    }
    
    # Get VAE path (check for existing Wan2GP VAE first)
    vae_path = get_vae_path()
    
    if vae_path is None:
        # VAE not found, will be downloaded with models
        vae_from_download = model_dir / "vae.pth"
        if vae_from_download.exists():
            paths["vae"] = str(vae_from_download)
        else:
            raise FileNotFoundError(
                "VAE not found. Expected either:\n"
                f"  - ckpts/Wan2.1_VAE.safetensors (from Wan2GP)\n"
                f"  - {vae_from_download} (from FlashVSR download)"
            )
    else:
        # Check if conversion needed (.safetensors -> .pth)
        if vae_path.endswith('.safetensors'):
            ckpts_dir = get_ckpts_dir()
            pth_path = ckpts_dir / "Wan2.1_VAE.pth"
            
            if not pth_path.exists():
                print("[FlashVSR] VAE is in safetensors format, converting to pth...")
                vae_path = convert_vae_safetensors_to_pth(vae_path, str(pth_path))
            else:
                vae_path = str(pth_path)
        
        paths["vae"] = vae_path
    
    # Variant-specific requirements
    if variant in ["tiny", "tiny-long"]:
        # Tiny variants use TCDecoder
        tc_decoder_path = model_dir / "TCDecoder.ckpt"
        
        if tc_decoder_path.exists():
            paths["tc_decoder"] = str(tc_decoder_path)
        else:
            print("[FlashVSR] Warning: TC Decoder not found. Tiny pipeline may fall back to VAE decode.")
    
    # Validate required paths exist
    for key, path in paths.items():
        if key == "tc_decoder":
            continue  # Optional for tiny variants
        if not Path(path).exists():
            raise FileNotFoundError(f"Required model file not found: {path}")
    
    return paths


def load_pipeline(variant: str = "tiny", device: str = "cuda", torch_dtype = torch.float16, model_version: str = "FlashVSR-v1.1"):
    """
    Load and initialize the specified FlashVSR pipeline variant.
    
    This function matches the upstream FlashVSR_plus init_pipeline() implementation
    to ensure temporal consistency and correct model loading.
    
    Args:
        variant: Pipeline variant - "tiny", "tiny-long", or "full"
        device: Device to load models on (default: "cuda")
        torch_dtype: Data type for model weights (default: torch.float16)
        model_version: Model version - "FlashVSR" or "FlashVSR-v1.1" (default: "FlashVSR-v1.1")
        
    Returns:
        Initialized pipeline instance
        
    Raises:
        ValueError if variant is unknown
        FileNotFoundError if required models are missing
    """
    from ..models.model_manager import ModelManager
    from ..pipelines import FlashVSRTinyPipeline, FlashVSRTinyLongPipeline, FlashVSRFullPipeline
    from ..models.TCDecoder import build_tcdecoder
    from ..models.utils import Buffer_LQ4x_Proj, Causal_LQ4x_Proj
    
    variant = variant.lower()
    
    # Get all model paths (triggers download if needed)
    model_paths = get_model_paths(variant, model_version=model_version)
    
    print(f"[FlashVSR] Loading {variant.upper()} pipeline (model: {model_version})...")
    print(f"[FlashVSR] Device: {device}, dtype: {torch_dtype}")
    
    # === Match upstream init_pipeline() exactly ===
    
    # Initialize model manager - only load DiT for tiny/tiny-long, also VAE for full
    mm = ModelManager(torch_dtype=torch_dtype, device="cpu")
    
    if variant == "full":
        mm.load_models([model_paths["dit"], model_paths["vae"]])
        pipeline = FlashVSRFullPipeline.from_model_manager(mm, device=device)
    else:
        # Tiny and Tiny-Long variants
        mm.load_models([model_paths["dit"]])
        
        if variant == "tiny":
            pipeline = FlashVSRTinyPipeline.from_model_manager(mm, device=device)
        else:  # tiny-long
            pipeline = FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
        
        # Build TCDecoder with correct latent channels (16 + 768 = 784)
        # This is CRITICAL for proper decoding with conditioning
        print("[FlashVSR] Building TCDecoder (latent_channels=784 for conditioning)...")
        pipeline.TCDecoder = build_tcdecoder(
            new_channels=[512, 256, 128, 128],
            device=device,
            dtype=torch_dtype,
            new_latent_channels=16 + 768  # 784 channels for latent + conditioning
        )
        
        # Load TCDecoder weights
        if "tc_decoder" in model_paths:
            tc_state = torch.load(model_paths["tc_decoder"], map_location=device)
            pipeline.TCDecoder.load_state_dict(tc_state, strict=False)
            pipeline.TCDecoder.clean_mem()
            print("[FlashVSR] TCDecoder loaded successfully")
    
    # Initialize LQ_proj_in based on model version
    # CRITICAL: FlashVSR uses Buffer_LQ4x_Proj, FlashVSR-v1.1 uses Causal_LQ4x_Proj
    # Using the wrong one causes temporal ghosting!
    if model_version == "FlashVSR":
        print("[FlashVSR] Using Buffer_LQ4x_Proj (FlashVSR v1.0)")
        lq_proj = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=torch_dtype)
    else:
        print("[FlashVSR] Using Causal_LQ4x_Proj (FlashVSR v1.1)")
        lq_proj = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=torch_dtype)
    
    # Attach LQ_proj_in to the denoising model (DiT)
    pipeline.denoising_model().LQ_proj_in = lq_proj
    
    # Load LQ_proj_in weights if available
    if "lq_proj" in model_paths:
        lq_state = torch.load(model_paths["lq_proj"], map_location="cpu")
        pipeline.denoising_model().LQ_proj_in.load_state_dict(lq_state, strict=True)
        print("[FlashVSR] LQ_proj_in loaded successfully")
    
    # Move to device and enable VRAM management
    pipeline.to(device, dtype=torch_dtype)
    pipeline.enable_vram_management()
    
    # Initialize cross-attention KV cache
    print("[FlashVSR] Initializing cross-attention KV cache...")
    pipeline.init_cross_kv(prompt_path=model_paths["posi_prompt"])
    
    # Load models to device - ALWAYS load both dit and vae (matches upstream exactly)
    # This is critical for temporal consistency
    pipeline.load_models_to_device(["dit", "vae"])
    
    print(f"[FlashVSR] {variant.upper()} pipeline ready!")
    
    return pipeline


def check_vram_requirements(variant: str = "tiny") -> Tuple[int, str]:
    """
    Get VRAM requirements for the specified pipeline variant.
    
    Args:
        variant: Pipeline variant - "tiny", "tiny-long", or "full"
        
    Returns:
        Tuple of (required_vram_gb, description)
    """
    requirements = {
        "tiny": (8, "8-10GB VRAM (Recommended for most users, 8GB GPUs)"),
        "tiny-long": (10, "10-12GB VRAM (For long videos >120 frames)"),
        "full": (18, "18-24GB VRAM (Highest quality, requires high-end GPU)")
    }
    
    variant = variant.lower()
    if variant not in requirements:
        return (8, "Unknown variant - defaulting to Tiny requirements")
    
    return requirements[variant]


def get_available_variants() -> list:
    """
    Get list of available pipeline variants.
    
    Returns:
        List of variant names
    """
    return ["tiny", "tiny-long", "full"]


def estimate_vram_for_resolution(width: int, height: int, num_frames: int, variant: str = "tiny") -> int:
    """
    Estimate VRAM usage for given resolution and frame count.
    
    Args:
        width: Output width
        height: Output height
        num_frames: Number of frames
        variant: Pipeline variant
        
    Returns:
        Estimated VRAM in GB
    """
    # Base VRAM for model weights
    base_vram = {
        "tiny": 3.5,
        "tiny-long": 4.0,
        "full": 8.0
    }.get(variant.lower(), 3.5)
    
    # Estimate activation memory (rough approximation)
    # Latent size is 1/8 of original, 16 channels
    latent_elements = (width // 8) * (height // 8) * 16 * num_frames
    # Assume float16 (2 bytes per element)
    activation_vram = (latent_elements * 2) / (1024 ** 3)  # Convert to GB
    
    # Add safety margin (1.5x for intermediate tensors)
    total_vram = (base_vram + activation_vram) * 1.5
    
    return int(total_vram + 0.5)  # Round up
