#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch

from .vits.lightning import VitsModel

_LOGGER = logging.getLogger("piper_train.export_onnx")

OPSET_VERSION = 15


def main() -> None:
    """Main entry point"""
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to model checkpoint (.ckpt)")
    parser.add_argument("output", help="Path to output model (.onnx)")
    parser.add_argument("--config", help="Path to config.json (required if not in same directory as checkpoint)")
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # -------------------------------------------------------------------------

    args.checkpoint = Path(args.checkpoint)
    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Find and load config.json
    config_path = None
    if args.config:
        config_path = Path(args.config)
    else:
        # Try to find config.json in the same directory as the checkpoint
        potential_config = args.checkpoint.parent / "config.json"
        if potential_config.exists():
            config_path = potential_config
    
    if not config_path or not config_path.exists():
        _LOGGER.error(
            "Could not find config.json. Please provide it with --config argument."
        )
        return
    
    # Load the config to get required parameters
    _LOGGER.info(f"Loading config from {config_path}")
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    
    num_symbols = config.get("num_symbols", 0)
    num_speakers = config.get("num_speakers", 1)
    sample_rate = config.get("audio", {}).get("sample_rate", 22050)
    
    if num_symbols == 0:
        _LOGGER.error("Could not find num_symbols in config.json")
        return
    
    _LOGGER.info(f"Model info: num_symbols={num_symbols}, num_speakers={num_speakers}, sample_rate={sample_rate}")
    
    # Check if it's an optimized model
    optimized = False
    if "optimization" in config:
        optimized = True
        opt_info = config["optimization"]
        _LOGGER.info(f"Optimized model detected: {opt_info.get('model_type', 'unknown')}")
        _LOGGER.info(f"Optimized model settings: layers={opt_info.get('n_layers', 'unknown')}, "
                    f"hidden={opt_info.get('hidden_dim', 'unknown')}, "
                    f"filter={opt_info.get('filter_dim', 'unknown')}")

    # Load the checkpoint
    _LOGGER.info(f"Loading checkpoint from {args.checkpoint}")
    
    # First try to load the state_dict directly to inspect what keys are available
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    # Check if this is a clean PyTorch checkpoint without Lightning wrapping
    if "state_dict" not in checkpoint and "model" in checkpoint:
        _LOGGER.info("Loading model from 'model' key in checkpoint (non-Lightning format)")
        # Create model with parameters from config
        model = VitsModel(
            num_symbols=num_symbols,
            num_speakers=num_speakers,
            sample_rate=sample_rate,
            dataset=None,
        )
        
        # Handle optimized model parameters if needed
        if optimized:
            if "hidden_dim" in opt_info:
                model.hidden_channels = opt_info["hidden_dim"]
            if "filter_dim" in opt_info:
                model.filter_channels = opt_info["filter_dim"]
            if "n_layers" in opt_info:
                model.n_layers = opt_info["n_layers"]
        
        # Manual loading of model weights
        state_dict = checkpoint["model"]
        
        # Split into generator and discriminator weights
        model_g_dict = {k.replace('model_g.', ''): v for k, v in state_dict.items() 
                        if isinstance(k, str) and (k.startswith('model_g.') or not '.' in k)}
        
        model_d_dict = {k.replace('model_d.', ''): v for k, v in state_dict.items() 
                        if isinstance(k, str) and k.startswith('model_d.')}
        
        # If we didn't find properly prefixed weights, try finding them directly
        if not model_g_dict and not model_d_dict:
            # Try to infer which keys belong to which model
            model_g_dict = {}
            model_d_dict = {}
            
            for k, v in state_dict.items():
                if not isinstance(k, str):
                    continue
                if any(x in k for x in ['enc_', 'dec.', 'flow.', 'dp.']):
                    model_g_dict[k] = v
                elif any(x in k for x in ['disc', 'mpd.', 'msd.', 'discriminator']):
                    model_d_dict[k] = v
        
        # Report weight stats
        _LOGGER.info(f"Found {len(model_g_dict)} generator weights and {len(model_d_dict)} discriminator weights")
        
        # Load weights with flexible handling for optimized models
        if model_g_dict:
            load_state_dict_flexible(model.model_g, model_g_dict)
        if model_d_dict:
            load_state_dict_flexible(model.model_d, model_d_dict)
    else:
        # Use standard Lightning loading
        _LOGGER.info("Loading model via Lightning checkpoint loader")
        # Add parameters required by the constructor
        model = VitsModel.load_from_checkpoint(
            args.checkpoint, 
            dataset=None,
            num_symbols=num_symbols,
            num_speakers=num_speakers,
            sample_rate=sample_rate
        )
    
    model_g = model.model_g

    # Inference only
    model_g.eval()

    with torch.no_grad():
        model_g.dec.remove_weight_norm()

    def infer_forward(text, text_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio = model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
        )[0].unsqueeze(1)

        return audio

    model_g.forward = infer_forward

    # Create dummy input for ONNX export
    dummy_input_length = 50
    sequences = torch.randint(
        low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long
    )
    sequence_lengths = torch.LongTensor([sequences.size(1)])

    sid: Optional[torch.LongTensor] = None
    if num_speakers > 1:
        sid = torch.LongTensor([0])

    # noise, noise_w, length
    scales = torch.FloatTensor([0.667, 1.0, 0.8])
    dummy_input = (sequences, sequence_lengths, scales, sid)

    # Export
    _LOGGER.info(f"Exporting model to {args.output} (ONNX opset {OPSET_VERSION})")
    torch.onnx.export(
        model=model_g,
        args=dummy_input,
        f=str(args.output),
        verbose=False,
        opset_version=OPSET_VERSION,
        input_names=["input", "input_lengths", "scales", "sid"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time"},
        },
    )

    _LOGGER.info("Export successful!")


def load_state_dict_flexible(model, saved_state_dict):
    """
    Enhanced state dict loading function for optimized models.
    
    This version has better handling for dimension mismatches, which is
    important when loading optimized models with reduced dimensions.
    """
    state_dict = model.state_dict()
    new_state_dict = {}
    
    # Track metrics for reporting
    matched_keys = 0
    missing_keys = 0
    size_mismatch = 0
    
    for k, v in state_dict.items():
        if k in saved_state_dict:
            saved_tensor = saved_state_dict[k]
            
            # Check for dimension mismatch
            if isinstance(v, torch.Tensor) and isinstance(saved_tensor, torch.Tensor):
                if v.shape == saved_tensor.shape:
                    # Shapes match, use saved value
                    new_state_dict[k] = saved_tensor
                    matched_keys += 1
                else:
                    # Shapes don't match, handle the mismatch
                    _LOGGER.debug(f"Size mismatch for {k}: model={v.shape}, checkpoint={saved_tensor.shape}")
                    size_mismatch += 1
                    
                    # Try to copy as much data as possible
                    if len(v.shape) == len(saved_tensor.shape):
                        # Same number of dimensions, copy overlapping parts
                        new_tensor = torch.zeros_like(v)
                        
                        # Get the minimum sizes in each dimension
                        min_sizes = [min(s1, s2) for s1, s2 in zip(v.shape, saved_tensor.shape)]
                        
                        # Create slices for copying
                        slices = tuple(slice(0, s) for s in min_sizes)
                        
                        # Copy data from saved tensor to new tensor
                        if all(s > 0 for s in min_sizes):
                            new_tensor[slices] = saved_tensor[slices]
                            new_state_dict[k] = new_tensor
                            matched_keys += 1
                        else:
                            # If any dimension is 0, use initialized value
                            new_state_dict[k] = v
                            missing_keys += 1
                    else:
                        # Different number of dimensions, use initialized value
                        new_state_dict[k] = v
                        missing_keys += 1
            else:
                # Not a tensor, use saved value
                new_state_dict[k] = saved_tensor
                matched_keys += 1
        else:
            # Not in saved state dict, use initialized value
            _LOGGER.debug("%s is not in the checkpoint", k)
            new_state_dict[k] = v
            missing_keys += 1
    
    _LOGGER.info(f"Loaded state dict: {matched_keys} matched keys, {missing_keys} missing keys, {size_mismatch} size mismatches")
    model.load_state_dict(new_state_dict)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()