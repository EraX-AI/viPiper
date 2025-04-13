#!/usr/bin/env python3

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from .vits import commons
from .vits.lightning import VitsModel

_LOGGER = logging.getLogger("piper_train.export_onnx_streaming")
OPSET_VERSION = 15


class VitsEncoder(nn.Module):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def forward(self, x, x_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]

        gen = self.gen
        x, m_p, logs_p, x_mask = gen.enc_p(x, x_lengths)
        if gen.n_speakers > 1:
            assert sid is not None, "Missing speaker id"
            g = gen.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        if gen.use_sdp:
            logw = gen.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = gen.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(
            commons.sequence_mask(y_lengths, y_lengths.max()), 1
        ).type_as(x_mask)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        return z_p, y_mask, g


class VitsDecoder(nn.Module):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def forward(self, z, y_mask, g=None):
        z = self.gen.flow(z, y_mask, g=g, reverse=True)
        output = self.gen.dec((z * y_mask), g=g)
        return output


def main() -> None:
    """Main entry point"""
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to model checkpoint (.ckpt)")
    parser.add_argument("output_dir", help="Path to output directory")
    parser.add_argument("--config", help="Path to config.json (optional if in same dir as checkpoint)")
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
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find and load config.json
    config_path = None
    if args.config:
        config_path = Path(args.config)
    else:
        potential_config = args.checkpoint.parent / "config.json"
        if potential_config.exists():
            config_path = potential_config
    
    model_params = {}
    if config_path and config_path.exists():
        _LOGGER.info(f"Loading config from {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        # Extract basic model parameters
        model_params["num_symbols"] = config.get("num_symbols", 0)
        model_params["num_speakers"] = config.get("num_speakers", 1)
        model_params["sample_rate"] = config.get("audio", {}).get("sample_rate", 22050)
        
        # Extract optimization parameters if available
        if "optimization" in config:
            opt = config["optimization"]
            _LOGGER.info(f"Optimized model detected: {opt.get('model_type', 'unknown')}")
            _LOGGER.info(f"Optimized model settings: layers={opt.get('n_layers', 'unknown')}, "
                        f"hidden={opt.get('hidden_dim', 'unknown')}, "
                        f"filter={opt.get('filter_dim', 'unknown')}")
            
            if "hidden_dim" in opt:
                model_params["hidden_channels"] = opt["hidden_dim"]
                model_params["inter_channels"] = opt["hidden_dim"]
            if "filter_dim" in opt:
                model_params["filter_channels"] = opt["filter_dim"]
            if "n_layers" in opt:
                model_params["n_layers"] = opt["n_layers"]
    
    # Load the model with the parameters from config
    _LOGGER.info(f"Loading model with parameters: {model_params}")
    
    # Check if this is a clean PyTorch checkpoint without Lightning wrapping
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    if "state_dict" not in checkpoint and "model" in checkpoint:
        _LOGGER.info("Loading model from 'model' key in checkpoint (non-Lightning format)")
        
        # Create model with parameters from config
        if "num_symbols" not in model_params or "num_speakers" not in model_params:
            _LOGGER.error("For non-Lightning checkpoints, num_symbols and num_speakers must be provided in config")
            return
            
        model = VitsModel(
            num_symbols=model_params["num_symbols"],
            num_speakers=model_params["num_speakers"],
            sample_rate=model_params.get("sample_rate", 22050),
            dataset=None,
            **{k: v for k, v in model_params.items() if k not in ["num_symbols", "num_speakers", "sample_rate"]}
        )
        
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
        # Use standard Lightning loading with our parameters
        model = VitsModel.load_from_checkpoint(
            args.checkpoint, 
            dataset=None, 
            **model_params
        )
    
    model_g = model.model_g

    with torch.no_grad():
        model_g.dec.remove_weight_norm()

    _LOGGER.info("Exporting encoder...")
    decoder_input = export_encoder(args, model_g)
    _LOGGER.info("Exporting decoder...")
    export_decoder(args, model_g, decoder_input)
    _LOGGER.info("Exported model to %s", str(args.output_dir))


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


def export_encoder(args, model_g):
    model = VitsEncoder(model_g)
    model.eval()

    num_symbols = model_g.n_vocab
    num_speakers = model_g.n_speakers

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

    output_names = [
        "z",
        "y_mask",
    ]
    if model_g.n_speakers > 1:
        output_names.append("g")

    onnx_path = os.fspath(args.output_dir.joinpath("encoder.onnx"))

    # Export
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=onnx_path,
        verbose=False,
        opset_version=OPSET_VERSION,
        input_names=["input", "input_lengths", "scales", "sid"],
        output_names=output_names,
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "z": {0: "batch_size", 2: "time"},
            "y_mask": {0: "batch_size", 2: "time"},
        },
    )
    _LOGGER.info("Exported encoder to %s", onnx_path)

    return model(*dummy_input)


def export_decoder(args, model_g, decoder_input):
    model = VitsDecoder(model_g)
    model.eval()

    input_names = [
        "z",
        "y_mask",
    ]
    if model_g.n_speakers > 1:
        input_names.append("g")

    onnx_path = os.fspath(args.output_dir.joinpath("decoder.onnx"))

    # Export
    torch.onnx.export(
        model=model,
        args=decoder_input,
        f=onnx_path,
        verbose=False,
        opset_version=OPSET_VERSION,
        input_names=input_names,
        output_names=["output"],
        dynamic_axes={
            "z": {0: "batch_size", 2: "time"},
            "y_mask": {0: "batch_size", 2: "time"},
            "output": {0: "batch_size", 1: "time"},
        },
    )

    _LOGGER.info("Exported decoder to %s", onnx_path)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()