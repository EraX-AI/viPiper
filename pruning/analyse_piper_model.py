import torch
import pprint
import collections
from collections import defaultdict
import re

def analyze_model_architecture(model_path):
    # Load the model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Determine the state dict location
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Maybe it's just the state dict directly
            state_dict = checkpoint
    else:
        # If it's not a dict, assume it's the state dict directly
        state_dict = checkpoint
    
    print(f"Model loaded with {len(state_dict)} keys")
    
    # Extract the top-level structure
    top_levels = defaultdict(int)
    for key in state_dict.keys():
        if not isinstance(key, str):
            continue
        parts = key.split('.')
        if parts:
            top_levels[parts[0]] += 1
    
    print("\n--- Top-level components ---")
    for component, count in sorted(top_levels.items(), key=lambda x: x[1], reverse=True):
        print(f"{component}: {count} parameters")
    
    # Find all unique path patterns (without specific layer numbers)
    path_patterns = set()
    layer_counts = defaultdict(set)
    
    for key in state_dict.keys():
        if not isinstance(key, str):
            continue
            
        # Replace layer numbers with 'N'
        pattern = re.sub(r'\.\d+\.', '.N.', key)
        path_patterns.add(pattern)
        
        # Count layer indices for each component
        layer_match = re.search(r'\.(\d+)\.', key)
        if layer_match:
            component = key.split('.')[0]
            layer_num = int(layer_match.group(1))
            layer_counts[component].add(layer_num)
    
    print("\n--- Path patterns ---")
    for pattern in sorted(path_patterns):
        print(pattern)
    
    print("\n--- Layer counts ---")
    for component, layers in layer_counts.items():
        print(f"{component}: {max(layers)+1 if layers else 0} layers (indices: {sorted(layers)})")
    
    # Analyze tensor dimensions
    dimensions = defaultdict(list)
    
    for key, tensor in state_dict.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            continue
            
        component = key.split('.')[0]
        dimensions[f"{component}_dim{len(tensor.shape)}"].append(tensor.shape)
    
    print("\n--- Tensor dimensions ---")
    for dim_key, shapes in sorted(dimensions.items()):
        # Count frequency of each shape
        shape_counts = collections.Counter(shapes)
        most_common = shape_counts.most_common(5)
        
        print(f"{dim_key}: {len(shapes)} tensors")
        for shape, count in most_common:
            print(f"  {shape}: {count} occurrences")
    
    # If config exists, print it
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        print("\n--- Model configuration ---")
        pprint.pprint(checkpoint['config'])

# Usage
analyze_model_architecture("pretrained_high/ljspeech-2000.ckpt")
