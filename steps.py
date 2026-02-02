"""
holis_pipeline/steps.py

Shared step functions for the HOLiS pipeline.

Goal:
- These functions are used BOTH by:
  - the per-step CLI scripts (convert_scan.py, process_scan.py, segment_scan.py, …)
  - a future single-job / in-memory pipeline that chains all steps without
    writing intermediates to disk.

Conventions:
- Prefer Dask arrays for large image volumes.
- Nothing here should submit SLURM jobs.
- Nothing here should call subprocesses.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import dask.array as da
import numpy as np
import torch

# If you have these already, import them from your own modules:
# from holis_pipeline.read_data import read_fli_as_zarr, read_omehans
# from holis_pipeline.utils.zarr_related import write_omehans, write_zarr
# from holis_pipeline.preprocessing_functions import (
#     compute_bg_masks_for_scan,
#     compute_laser_patterns_for_scan,
#     apply_bg_and_pattern_correction,
# )
# from holis_pipeline.unet_inference import run_unet3d_on_volume


# ----------------------------------------------------------------------
# Config for steps (lightweight; you can reuse or extend your existing one)
# ----------------------------------------------------------------------

@dataclass
class StepConfig:
    # paths used when we *do* write to disk
    transforms_matlab_path: str
    matrices_matlab_path: str

    # directories
    # raw / fli / omehans / processed / segmentation etc.
    output_dir_ome: str
    output_dir_bg: str
    output_dir_lasers: str
    output_dir_process: str
    output_dir_segment: str

    # model / segmentation
    segmentation_model_path: str
    device: str = "cuda"

    # misc
    overwrite: bool = False


# ----------------------------------------------------------------------
# Utility: simple path helpers
# ----------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_ome_path(cfg: StepConfig, fli_path: str) -> str:
    """Map a .fli.zst path to an omehans output path."""
    base = os.path.basename(fli_path)
    # Example: NPBB328-...-272.fli.zst -> NPBB328-...-272.omehans
    root = re.sub(r"\.fli\.zst$", "", base)
    root = re.sub(r"\s+", "_", root)
    root = re.sub(r"[^A-Za-z0-9._-]+", "_", root)
    return os.path.join(cfg.output_dir_ome, root + ".omehans")


def make_processed_path(cfg: StepConfig, ome_path: str) -> str:
    """Map an omehans path to a processed omehans/zarr path."""
    base = os.path.basename(ome_path)
    root = re.sub(r"\.omehans$", "", base)
    return os.path.join(cfg.output_dir_process, root + "_processed.omehans")


def make_segmentation_path(cfg: StepConfig, ome_path: str) -> str:
    """Map an omehans path to a segmentation output (e.g., zarr or omehans)."""
    base = os.path.basename(ome_path)
    root = re.sub(r"\.omehans$", "", base)
    return os.path.join(cfg.output_dir_segment, root + "_segmentation.zarr")


# ----------------------------------------------------------------------
# 1. CONVERT: FLI → OMEHANS (or Dask array)
# ----------------------------------------------------------------------

def load_raw_fli_as_dask(fli_path: str) -> da.Array:
    """
    Lazy read of FLI as Dask array.

    Replace this with your actual loader (likely read_fli_as_zarr or read_omehans_via_env).
    """
    # TODO: plug in your real implementation
    # e.g.: img = read_fli_as_zarr(fli_path)  # returns dask array
    raise NotImplementedError("Implement load_raw_fli_as_dask with your real FLI reader")


def step_convert_to_ome_dask(fli_path: str, cfg: StepConfig) -> Tuple[da.Array, Dict[str, Any]]:
    """
    Convert a raw FLI file to an OMEHANS-like Dask array + metadata,
    but do NOT write to disk here. Pure-ish function.

    Returns
    -------
    image_da : dask.array.Array
    meta     : dict
        Anything you need to later write out (spacing, axes, transforms, etc).
    """
    img_da = load_raw_fli_as_dask(fli_path)

    # TODO: attach proper metadata using your Matlab transforms, etc.
    meta = {
        "source_fli": fli_path,
        "axes": "CZYX",  # or whatever is correct in your case
        # "spacing": ...,
        # "affine": ...,
    }

    return img_da, meta


def write_ome_from_dask(img_da: da.Array, meta: Dict[str, Any], out_path: str, cfg: StepConfig) -> None:
    """
    Write OMEHANS/Zarr from a Dask array and metadata.
    This is where you call your current write_omehans or write_zarr.
    """
    _ensure_dir(os.path.dirname(out_path))

    if not cfg.overwrite and os.path.exists(out_path):
        print(f"[convert] {out_path} exists, skipping.")
        return

    # TODO: replace with your real writer
    # write_omehans(img_da, out_path, metadata=meta)
    raise NotImplementedError("Implement write_ome_from_dask with your existing writer")


# ----------------------------------------------------------------------
# 2. BACKGROUND MASKS
# ----------------------------------------------------------------------

def step_get_bg_masks(img_da: da.Array, cfg: StepConfig) -> da.Array:
    """
    Compute per-pixel or per-strip background masks for a scan lazily.

    The return should be a Dask array or something that can broadcast against img_da.
    """
    # TODO: plug in your actual background computation (median over dark frames, etc.)
    # Example (totally fake):
    # bg = da.percentile(img_da, 5, axis=0)
    # return bg
    raise NotImplementedError("Implement step_get_bg_masks based on get_bg_masks.py logic")


def save_bg_mask(bg_da: da.Array, out_path: str, cfg: StepConfig) -> None:
    """
    Persist background mask to disk if you want an intermediate.
    """
    _ensure_dir(os.path.dirname(out_path))
    # You might want np.save, zarr, or OMEHANS, depending on what you use now.
    # Example:
    # bg_da.to_zarr(out_path, overwrite=cfg.overwrite)
    raise NotImplementedError("Implement save_bg_mask with your current bg saving")


# ----------------------------------------------------------------------
# 3. LASER PATTERNS
# ----------------------------------------------------------------------

def step_get_laser_patterns(img_da: da.Array, cfg: StepConfig) -> da.Array:
    """
    Compute laser illumination patterns (e.g., per-column profiles) lazily.
    """
    # TODO: plug in your real laser pattern computation.
    # Example:
    # pattern = da.mean(img_da, axis=0)  # fake
    # return pattern
    raise NotImplementedError("Implement step_get_laser_patterns based on get_laser_patterns.py")


def save_laser_patterns(pattern_da: da.Array, out_path: str, cfg: StepConfig) -> None:
    """
    Persist laser patterns if you want them on disk.
    """
    _ensure_dir(os.path.dirname(out_path))
    # Example:
    # pattern_da.to_zarr(out_path, overwrite=cfg.overwrite)
    raise NotImplementedError("Implement save_laser_patterns with your current writer")


# ----------------------------------------------------------------------
# 4. PROCESSING: BG subtraction + laser correction + misc
# ----------------------------------------------------------------------

def step_process_scan(
    img_da: da.Array,
    bg_da: Optional[da.Array],
    pattern_da: Optional[da.Array],
    cfg: StepConfig,
) -> da.Array:
    """
    Apply background subtraction, laser pattern correction, clipping, etc.

    All operations should stay lazy (Dask).
    """
    corrected = img_da

    if bg_da is not None:
        # Assumes bg_da broadcasts correctly
        corrected = da.maximum(corrected.astype("float32") - bg_da.astype("float32"), 0)

    if pattern_da is not None:
        # Example: divide by normalized pattern
        # (You’ll want to handle zeros / normalization in your real code.)
        corrected = corrected / (pattern_da + 1e-6)

    # TODO: clip, cast back to uint16 if you like:
    # corrected = da.clip(corrected, 0, 65535).astype("uint16")

    return corrected


def save_processed_scan(processed_da: da.Array, out_path: str, cfg: StepConfig) -> None:
    """
    Persist processed volume (background-corrected, laser-corrected) to disk.
    """
    _ensure_dir(os.path.dirname(out_path))
    # Example:
    # processed_da.to_zarr(out_path, overwrite=cfg.overwrite)
    raise NotImplementedError("Implement save_processed_scan using your current process_scan.py writer")


# ----------------------------------------------------------------------
# 5. SEGMENTATION
# ----------------------------------------------------------------------

def load_segmentation_model(cfg: StepConfig) -> torch.nn.Module:
    """
    Load your UNet3D / UNETR3D model for inference.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # TODO: replace with your real model definition:
    # from unet_class import UNet3D
    # from unetr_model import UNETR3D

    # Example placeholder:
    # model = UNETR3D(...)
    # model.load_state_dict(torch.load(cfg.segmentation_model_path, map_location=device))
    # model.to(device)
    # model.eval()
    raise NotImplementedError("Implement load_segmentation_model with your real UNet/UNETR")


def _run_model_on_block(block: np.ndarray, model: torch.nn.Module, device: torch.device) -> np.ndarray:
    """
    Helper used with Dask map_blocks: run segmentation model on a single block.

    block: numpy array (C, Z, Y, X) or (1, Z, Y, X) etc.
    """
    # Adjust this to your true layout.
    with torch.no_grad():
        x = torch.from_numpy(block).float().to(device)
        if x.ndim == 4:
            # add batch dimension
            x = x.unsqueeze(0)  # (1, C, Z, Y, X) or (1, 1, ...)
        logits = model(x)
        # binary example:
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).cpu().numpy().astype("uint8")
        # remove batch dimension
        preds = np.squeeze(preds, axis=0)
        return preds


def step_segment_scan(processed_da: da.Array, cfg: StepConfig) -> da.Array:
    """
    Build a lazy segmentation volume using Dask + PyTorch.

    This returns a Dask array of labels; computing it will run the model.
    """
    model = load_segmentation_model(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # You might want to rechunk for inference:
    # processed_da = processed_da.rechunk((1, 32, 128, 128))  # e.g., (C, Z, Y, X)

    def _wrap(block):
        return _run_model_on_block(block, model, device)

    # Choose the output dtype and chunking
    seg_da = processed_da.map_blocks(
        _wrap,
        dtype="uint8",
    )

    return seg_da


def save_segmentation(seg_da: da.Array, out_path: str, cfg: StepConfig) -> None:
    """
    Persist segmentation labels to disk (this will trigger compute()).
    """
    _ensure_dir(os.path.dirname(out_path))
    # Example: write as zarr
    seg_da.to_zarr(out_path, overwrite=cfg.overwrite)


# ----------------------------------------------------------------------
# High-level convenience wrappers (optional)
# ----------------------------------------------------------------------

def full_pipeline_one_scan(
    fli_path: str,
    cfg: StepConfig,
    write_intermediate: bool = False,
) -> None:
    """
    Example “all in one” function:
    FLI → OME (optional write) → BG → pattern → processed (optional write) → seg (write).

    This is what your single-job pipeline might call instead of staging everything.
    """
    # 1. convert
    img_da, meta = step_convert_to_ome_dask(fli_path, cfg)
    ome_path = make_ome_path(cfg, fli_path)

    if write_intermediate:
        write_ome_from_dask(img_da, meta, ome_path, cfg)

    # 2. bg masks
    bg_da = step_get_bg_masks(img_da, cfg)
    if write_intermediate:
        bg_path = os.path.join(cfg.output_dir_bg, os.path.basename(ome_path) + "_bg.zarr")
        save_bg_mask(bg_da, bg_path, cfg)

    # 3. laser patterns
    pattern_da = step_get_laser_patterns(img_da, cfg)
    if write_intermediate:
        pattern_path = os.path.join(cfg.output_dir_lasers, os.path.basename(ome_path) + "_pattern.zarr")
        save_laser_patterns(pattern_da, pattern_path, cfg)

    # 4. process
    processed_da = step_process_scan(img_da, bg_da, pattern_da, cfg)
    processed_path = make_processed_path(cfg, ome_path)
    if write_intermediate:
        save_processed_scan(processed_da, processed_path, cfg)

    # 5. segment (this one we always write)
    seg_da = step_segment_scan(processed_da, cfg)
    seg_path = make_segmentation_path(cfg, ome_path)
    save_segmentation(seg_da, seg_path, cfg)
