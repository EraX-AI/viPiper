#!/usr/bin/env python3
"""
Piper Model Size Optimizer

Creates a compact model by:
1. Reducing the number of transformer layers (default: keep 3 layers)
2. Reducing hidden dimensions strategically
3. Simplifying architecture where appropriate
4. Maintaining float32 precision for training compatibility

All while preserving pre-trained knowledge for fine-tuning.
"""

import torch
import os
import logging
import argparse
import json
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def create_optimized_model(input_path, output_dir, config_path, num_layers=3, dimension_factor=0.5):
    """
    Create an optimized version of a Piper TTS model.
    
    Args:
        input_path: Path to the original model checkpoint
        output_dir: Directory to save the optimized model
        config_path: Path to the original config.json
        num_layers: Number of transformer layers to keep (default: 3)
        dimension_factor: Factor by which to reduce dimensions (default: 0.5)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # Get state dict
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Calculate current model size
    original_params = sum(tensor.numel() for key, tensor in state_dict.items() 
                         if isinstance(tensor, torch.Tensor))
    original_size_mb = original_params * 4 / (1024 * 1024)  # float32
    
    logger.info(f"Original model: {original_params:,} parameters ({original_size_mb:.2f}MB)")
    
    # 1. Identify transformer layers
    all_layers = set()
    for key in state_dict.keys():
        if isinstance(key, str) and ('.encoder.attn_layers.' in key or '.encoder.ffn_layers.' in key):
            parts = key.split('.')
            for part in parts:
                if part.isdigit():
                    all_layers.add(int(part))
    
    all_layers = sorted(list(all_layers))
    logger.info(f"Original transformer layers: {all_layers}")
    
    # Select which layers to keep
    if len(all_layers) <= num_layers:
        # If we already have fewer layers than requested, keep all
        keep_layers = all_layers
    else:
        # For 3 layers, keep first, middle, and last for best coverage
        if num_layers == 3 and len(all_layers) > 4:
            mid_idx = len(all_layers) // 2
            keep_layers = [all_layers[0], all_layers[mid_idx], all_layers[-1]]
        # For 2 layers, keep first and last
        elif num_layers == 2:
            keep_layers = [all_layers[0], all_layers[-1]]
        # For other cases, distribute evenly
        else:
            indices = [int(i * (len(all_layers) - 1) / (num_layers - 1)) for i in range(num_layers)]
            keep_layers = [all_layers[i] for i in indices]
    
    logger.info(f"Keeping transformer layers: {keep_layers}")
    
    # 2. Define dimension reduction targets
    # Get model dimensions from original model
    hidden_dim = 192  # Default from config
    filter_dim = 768  # Default from config
    upsample_dim = 512  # For high quality
    
    # Find actual dimensions in model
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            if 'hidden' in key and tensor.dim() > 0:
                hidden_dim = tensor.size(0)
            elif 'filter' in key and tensor.dim() > 0:
                filter_dim = tensor.size(0)
            elif 'ups.0' in key and 'weight' in key and tensor.dim() > 0:
                upsample_dim = tensor.size(0)
    
    # Record original dimensions for config
    original_dims = {
        "hidden": hidden_dim,
        "filter": filter_dim,
        "upsample": upsample_dim
    }
    
    # Set reduction targets based on the dimension_factor
    target_dims = {
        "hidden": int(hidden_dim * dimension_factor),
        "filter": int(filter_dim * dimension_factor),
        "upsample": int(upsample_dim * dimension_factor),
        "n_layers": len(keep_layers),
        "precision": "float32"
    }
    
    # Make dimensions divisible by 8 for better GPU performance
    target_dims["hidden"] = max(96, (target_dims["hidden"] // 8) * 8)
    target_dims["filter"] = max(384, (target_dims["filter"] // 8) * 8)
    target_dims["upsample"] = max(192, (target_dims["upsample"] // 8) * 8)
    
    logger.info(f"Original dimensions: hidden={hidden_dim}, filter={filter_dim}, upsample={upsample_dim}")
    logger.info(f"Target dimensions: {target_dims}")
    
    # 3. Process each tensor in the state dict
    new_state_dict = {}
    processed_count = 0
    skipped_count = 0
    size_mismatch_count = 0
    
    for key, tensor in state_dict.items():
        # Skip layers that should be removed
        skip = False
        if isinstance(key, str) and ('.encoder.attn_layers.' in key or '.encoder.ffn_layers.' in key):
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part.isdigit() and i > 0:
                    layer_idx = int(part)
                    if layer_idx not in keep_layers:
                        skip = True
                        skipped_count += 1
                        break
        
        if skip:
            continue
        
        # Process tensor dimensions
        if isinstance(tensor, torch.Tensor):
            modified = False
            original_shape = tensor.shape
            new_tensor = tensor
            
            if tensor.dim() >= 2:
                # Process dimensions based on the tensor's role
                new_shape = list(tensor.shape)
                
                # First dimension (usually output channels)
                if tensor.size(0) >= hidden_dim * 0.8:  # Allow some flexibility for matching
                    ratio = tensor.size(0) / hidden_dim
                    new_shape[0] = int(target_dims["hidden"] * ratio)
                    new_shape[0] = max(32, new_shape[0])  # Minimum reasonable size
                    modified = True
                
                # Second dimension (usually input channels)
                if tensor.dim() > 2 and tensor.size(1) >= hidden_dim * 0.8:
                    ratio = tensor.size(1) / hidden_dim
                    new_shape[1] = int(target_dims["hidden"] * ratio)
                    new_shape[1] = max(32, new_shape[1])
                    modified = True
                
                if modified:
                    # Create a new tensor with target dimensions
                    new_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
                    
                    # Copy values from original tensor (up to the minimum dimensions)
                    min_dims = [min(a, b) for a, b in zip(tensor.shape, new_shape)]
                    slices = tuple(slice(0, dim) for dim in min_dims)
                    
                    # Enhanced copying for attention layers and key network components
                    if any(pattern in key for pattern in ['attn', 'attention', 'query', 'key', 'value']):
                        # For attention matrices, try to preserve the most important information
                        if tensor.dim() == 2:  # For linear layers
                            # Simple copy for the overlapping part
                            new_tensor[slices] = tensor[slices]
                            
                            # If we're drastically reducing dimension, try to preserve the most important vectors
                            if new_shape[0] < tensor.shape[0] * 0.7:
                                # Calculate importance of each output dimension (L2 norm)
                                importance = torch.norm(tensor, dim=1)
                                _, indices = torch.topk(importance, k=min(new_shape[0], len(importance)))
                                
                                # For the remaining positions, copy from the most important vectors
                                for i in range(min_dims[0], new_shape[0]):
                                    if i < len(indices):
                                        idx = indices[i].item()
                                        new_tensor[i, :min_dims[1]] = tensor[idx, :min_dims[1]]
                        else:
                            # For conv layers or higher-dimensional tensors, use straight copy
                            new_tensor[slices] = tensor[slices]
                    else:
                        # Standard copy for other layers
                        new_tensor[slices] = tensor[slices]
                    
                    # Record that we processed this tensor
                    processed_count += 1
                    size_mismatch_count += 1
                    
                    if processed_count % 100 == 0:
                        logger.debug(f"Processed {processed_count} tensors...")
            
            elif tensor.dim() == 1:
                # Handle 1D tensors (biases, etc.)
                if tensor.size(0) >= hidden_dim * 0.8:
                    ratio = tensor.size(0) / hidden_dim
                    new_size = int(target_dims["hidden"] * ratio)
                    new_size = max(32, new_size)
                    
                    # Create new tensor and copy values
                    new_tensor = torch.zeros(new_size, dtype=tensor.dtype, device=tensor.device)
                    min_size = min(tensor.size(0), new_size)
                    new_tensor[:min_size] = tensor[:min_size]
                    
                    modified = True
                    processed_count += 1
                    size_mismatch_count += 1
                
            # Save the processed tensor
            new_state_dict[key] = new_tensor
        else:
            # Keep non-tensor data
            new_state_dict[key] = tensor
    
    # Calculate final model size
    final_params = sum(tensor.numel() for key, tensor in new_state_dict.items() 
                      if isinstance(tensor, torch.Tensor))
    
    # Size calculation with float32 precision
    final_size_mb = final_params * 4 / (1024 * 1024)  # 4 bytes per float32
    
    logger.info(f"Final model: {final_params:,} parameters ({final_size_mb:.2f}MB)")
    logger.info(f"Processed {processed_count} tensors, skipped {skipped_count} tensors, resized {size_mismatch_count} tensors")
    
    # Create a clean model dictionary
    clean_dict = {"model": {}}
    for key, tensor in new_state_dict.items():
        if isinstance(tensor, torch.Tensor):
            # Create a fresh tensor with no history
            clean_dict["model"][key] = tensor.clone().detach()
        else:
            clean_dict["model"][key] = tensor
    
    # Save model files
    model_pt = os.path.join(output_dir, "model.pt")
    model_ckpt = os.path.join(output_dir, "model.ckpt")
    
    logger.info(f"Saving model to {model_pt} and {model_ckpt}")
    
    # Use new zipfile serialization for smaller files
    torch.save(clean_dict, model_pt, _use_new_zipfile_serialization=True)
    torch.save(clean_dict, model_ckpt, _use_new_zipfile_serialization=True)
    
    # Check actual file sizes
    pt_size = os.path.getsize(model_pt) / (1024 * 1024)
    ckpt_size = os.path.getsize(model_ckpt) / (1024 * 1024)
    
    logger.info(f"Saved model file sizes:")
    logger.info(f"  - PyTorch (.pt): {pt_size:.2f}MB")
    logger.info(f"  - Checkpoint (.ckpt): {ckpt_size:.2f}MB")
    
    # Update config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update config to match the compact model
    config["audio"]["quality"] = "optimized"
    
    # Switch to 3-stage upsampling architecture
    if "audio" not in config:
        config["audio"] = {}
    
    config["audio"]["upsample_rates"] = [8, 8, 4]
    config["audio"]["upsample_kernel_sizes"] = [16, 16, 8]
    config["audio"]["upsample_initial_channel"] = target_dims["upsample"]
    
    # Add optimization info with more detailed information
    config["optimization"] = {
        "hidden_dim": target_dims["hidden"],
        "filter_dim": target_dims["filter"],
        "n_layers": target_dims["n_layers"],
        "n_heads": 2,  # Unchanged from original
        "original_hidden_dim": original_dims["hidden"],
        "original_filter_dim": original_dims["filter"],
        "original_n_layers": len(all_layers),
        "original_params": int(original_params),
        "optimized_params": int(final_params),
        "reduction_percent": round(((original_params - final_params) / original_params) * 100, 1),
        "model_type": f"optimized-{num_layers}layers-{int(dimension_factor*100)}pct",
        "precision": target_dims["precision"],
        "kept_layers": keep_layers,
        "pruning_factor": dimension_factor
    }
    
    # Save config
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Config saved to {config_file}")
    logger.info("Model optimization complete!")
    
    return {
        "pt_path": model_pt,
        "ckpt_path": model_ckpt,
        "original_params": original_params,
        "final_params": final_params,
        "original_size_mb": original_size_mb,
        "final_size_mb": final_size_mb,
        "pt_size": pt_size,
        "ckpt_size": ckpt_size,
        "config_path": config_file,
        "hidden_dim": target_dims["hidden"],
        "filter_dim": target_dims["filter"],
        "n_layers": target_dims["n_layers"]
    }

def main():
    parser = argparse.ArgumentParser(description="Create optimized Piper model")
    parser.add_argument("--input", required=True, help="Input model path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", required=True, help="Config path")
    parser.add_argument("--layers", type=int, default=3, help="Number of transformer layers to keep (default: 3)")
    parser.add_argument("--factor", type=float, default=0.5, 
                      help="Dimension reduction factor (0.0-1.0, default: 0.5)")
    
    args = parser.parse_args()
    
    # Validate input parameters
    if args.layers < 1:
        logger.error("Number of layers must be at least 1")
        return
    
    if args.factor <= 0 or args.factor > 1:
        logger.error("Dimension factor must be between 0 and 1")
        return
    
    try:
        result = create_optimized_model(
            args.input, 
            args.output, 
            args.config, 
            num_layers=args.layers, 
            dimension_factor=args.factor
        )
        
        logger.info("\nOptimized model created successfully:")
        logger.info(f"Original parameters: {result['original_params']:,} ({result['original_size_mb']:.2f}MB)")
        logger.info(f"Final parameters: {result['final_params']:,} ({result['final_size_mb']:.2f}MB)")
        logger.info(f"Actual file size: {result['ckpt_size']:.2f}MB (.ckpt)")
        logger.info(f"Reduction: {((result['original_size_mb'] - result['final_size_mb']) / result['original_size_mb'] * 100):.1f}%")
        logger.info(f"Model configuration: {result['n_layers']} layers, {result['hidden_dim']} hidden dim, {result['filter_dim']} filter dim")
        logger.info(f"Created files:")
        logger.info(f"  - {result['ckpt_path']} ({result['ckpt_size']:.2f}MB)")
        logger.info(f"  - {result['pt_path']} ({result['pt_size']:.2f}MB)")
        logger.info(f"  - {result['config_path']}")
        
        # Provide example training command
        logger.info("\nExample command to train with this model:")
        logger.info(f"python -m piper_train --dataset-dir /your/dataset --quality optimized \\\n"
                   f"  --optimized-model-dir {args.output} --batch-size 32 \\\n"
                   f"  --precision 32 --accelerator gpu")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
