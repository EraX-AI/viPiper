import argparse
import json
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ProgressBarBase
import sys

from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)


def init_weights_vits(model):
    """Initialize weights for VITS models using TTS-specific best practices.
    
    Args:
        model: The VITS model to initialize
        
    This follows TTS-specific best practices for VITS models:
    - Flow, posterior encoder, and text encoder: Use Xavier/Glorot normal
    - Decoder and upsampling: Use Xavier uniform with gain based on layer position
    - Convolutional layers: Special handling for ResBlock and WaveNet components
    - Embedding layers: Normal distribution with mean=0, std=0.02
    - LayerNorm and other normalizations: bias=0, weight=1.0
    """
    _LOGGER.info("Initializing VITS weights for training from scratch")
    
    # Initialize all modules with appropriate weight initialization strategies
    for name, module in model.named_modules():
        # Text encoder and posterior encoder transformer layers
        if any(x in name for x in ['enc_p.', 'enc_q.', 'flow.', 'text_enc']):
            if isinstance(module, nn.Linear):
                # Linear layers in encoder get Xavier normal
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.xavier_normal_(module.weight, gain=1.0)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
        # Decoder and upsampling components
        elif any(x in name for x in ['dec.', 'upsample']):
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d):
                # For decoder convolutions, Xavier uniform works better
                gain = 1.0
                # Adjust gain based on position in network
                if 'upsample' in name:
                    gain = 0.5  # Lower gain for upsampling to prevent exploding activations
                
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.xavier_uniform_(module.weight, gain=gain)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Linear):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Special handling for WaveNet and ResBlock components
        elif any(x in name for x in ['wavenet', 'resblock']):
            if isinstance(module, nn.Conv1d):
                # Use He initialization for ReLU activations in WaveNet
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Generic module handling based on type
        elif isinstance(module, nn.Embedding):
            # Standard embedding initialization for TTS
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
        elif isinstance(module, nn.LayerNorm) or 'norm' in name.lower():
            # All normalization layers
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d):
            # Default initialization for other convolutional layers
            if hasattr(module, 'weight') and module.weight is not None:
                fan_in = module.in_channels * module.kernel_size[0]
                if fan_in > 0:
                    std = 1.0 / math.sqrt(fan_in)
                    nn.init.uniform_(module.weight, -std, std)
                else:
                    nn.init.xavier_uniform_(module.weight)
                
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Linear):
            # Default initialization for other linear layers
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    _LOGGER.info("VITS model weights initialized for training from scratch")
    return model


def init_weights_pytorch_default(model):
    """Initialize model weights using PyTorch's default initialization methods.

    Args:
        model: The model (e.g., VitsModel instance) to initialize.

    This function iterates through model modules and applies the standard
    PyTorch initializations for common layer types like Linear, Conv1d,
    ConvTranspose1d, Embedding, and LayerNorm. This serves as a baseline
    initialization strategy.
    """
    _LOGGER.info("Initializing model weights using PyTorch default methods")

    for name, module in model.named_modules():
        try:
            if isinstance(module, nn.Linear):
                # Default: Kaiming uniform for weight, uniform based on fan_in for bias
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if hasattr(module, 'bias') and module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)
                _LOGGER.debug(f"Initialized Linear layer {name} with PyTorch defaults")

            elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                # Default: Kaiming uniform for weight, uniform based on fan_in for bias
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if hasattr(module, 'bias') and module.bias is not None:
                    # Calculate fan_in for Conv layers based on groups
                    # Note: This replicates the internal logic more closely
                    num_input_fmaps = module.weight.size(1) * module.groups
                    receptive_field_size = 1
                    if module.weight.dim() > 2:
                        receptive_field_size = module.weight[0][0].numel()
                    fan_in = num_input_fmaps * receptive_field_size

                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)
                _LOGGER.debug(f"Initialized Conv layer {name} with PyTorch defaults")


            elif isinstance(module, nn.Embedding):
                # Default: Normal distribution mean=0, std=1
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.normal_(module.weight, mean=0.0, std=1.0)
                # Note: std=1.0 is the literal default, but often customized (like your std=0.02)
                _LOGGER.debug(f"Initialized Embedding layer {name} with PyTorch defaults (std=1.0)")

            elif isinstance(module, nn.LayerNorm):
                # Default: weight=1, bias=0
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
                _LOGGER.debug(f"Initialized LayerNorm layer {name} with PyTorch defaults")

            # Add elif blocks here for other layer types if needed (e.g., BatchNorm)

        except Exception as e:
            _LOGGER.warning(f"Could not initialize {name} ({type(module)}): {e}")


    _LOGGER.info("Model weights initialized using PyTorch default methods")
    return model
    
def load_state_dict(model, saved_state_dict):
    """
    Basic state dict loading function.
    
    This allows loading weights even when the model architecture has been modified,
    which is crucial for finetuning pruned/optimized models.
    """
    state_dict = model.state_dict()
    new_state_dict = {}
    
    # Track metrics for reporting
    matched_keys = 0
    missing_keys = 0
    
    for k, v in state_dict.items():
        if k in saved_state_dict:
            # Use saved value
            new_state_dict[k] = saved_state_dict[k]
            matched_keys += 1
        else:
            # Use initialized value
            _LOGGER.debug("%s is not in the checkpoint", k)
            new_state_dict[k] = v
            missing_keys += 1
    
    _LOGGER.info(f"Loaded state dict: {matched_keys} matched keys, {missing_keys} missing keys")
    model.load_state_dict(new_state_dict)


def load_state_dict_flexible(model, saved_state_dict):
    """
    Enhanced state dict loading function for optimized models with better dimension mismatch handling.
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
                    
                    # For incompatible shapes, it's better to use model's initialized weights
                    # rather than attempting partial loading which can lead to instability
                    new_state_dict[k] = v  # Use model's initialized weights for this parameter
            else:
                # Not a tensor, use saved value
                new_state_dict[k] = saved_tensor
                matched_keys += 1
        else:
            # Not in saved state dict, use initialized value
            _LOGGER.debug(f"{k} is not in the checkpoint")
            new_state_dict[k] = v
            missing_keys += 1
    
    _LOGGER.info(f"Loaded state dict: {matched_keys} matched keys, {missing_keys} missing keys, {size_mismatch} size mismatches")
    model.load_state_dict(new_state_dict)

from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import torchaudio
import random

class AudioSampleLogger(Callback):
    """
    Logs audio samples generated from random validation utterances during training
    and saves them to a 'samples' directory. Runs only on rank 0.
    """
    def __init__(self, frequency: int = 2000, num_samples: int = 2, scales: list = [0.667, 1.0, 0.8]):
        """
        Args:
            frequency (int): Log/Save audio every N training steps.
            num_samples (int): Number of random validation samples to generate.
            scales (list): Inference scales [noise_scale, length_scale, noise_scale_w].
        """
        super().__init__()
        self.frequency = frequency
        self.num_samples = num_samples
        self.scales = scales
        _LOGGER.info(f"AudioSampleLogger initialized: logging/saving {num_samples} samples every {frequency} steps.")

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int
    ):
        """Called when the train batch ends. Generates and saves/logs audio samples on rank 0."""

        # --- Only execute on the main process (rank 0) in distributed training ---
        if not trainer.is_global_zero:
            return
        # --------------------------------------------------------------------------

        step = trainer.global_step

        # Check conditions for logging/saving (now only checked on rank 0)
        # 1. Check if validation dataset exists on the model
        if not hasattr(pl_module, '_val_dataset') or pl_module._val_dataset is None:
            if step > 0 and step % self.frequency == 0:
                _LOGGER.warning("Validation dataset not found in model for sample logging/saving (rank 0).")
            return

        # 2. Check if validation dataset is empty
        if len(pl_module._val_dataset) == 0:
            if step > 0 and step % self.frequency == 0:
                _LOGGER.warning("Validation dataset is empty, cannot log/save samples (rank 0).")
            return

        # Proceed if it's the correct step and the validation set is valid
        if step % self.frequency == 0 and step > 0:
            _LOGGER.info(f"Generating validation audio samples at step {step + 1} (rank 0)...")

            # --- Path setup for saving samples ---
            try:
                save_dir_base = Path(trainer.default_root_dir)
                samples_dir = save_dir_base / "samples"
                samples_dir.mkdir(parents=True, exist_ok=True)
                _LOGGER.debug(f"Audio samples will be saved to: {samples_dir}")
            except Exception as e:
                 _LOGGER.error(f"Could not create samples directory under {trainer.default_root_dir}: {e}")
                 return

            logger_available = trainer.logger is not None and hasattr(trainer.logger, 'experiment')
            if not logger_available:
                _LOGGER.warning("Logger not available for sample logging (will only save files).")

            # --- Generate and process samples ---
            original_device = pl_module.device
            try:
                pl_module.eval() # Set model to evaluation mode

                num_available_samples = len(pl_module._val_dataset)
                num_to_generate = min(self.num_samples, num_available_samples)
                if num_to_generate > 0:
                     sample_indices = random.sample(range(num_available_samples), num_to_generate)
                else:
                     sample_indices = []

                with torch.no_grad():
                    for i, idx in enumerate(sample_indices):
                        try:
                            utt = pl_module._val_dataset[idx]

                            text = utt.phoneme_ids.unsqueeze(0).to(pl_module.device)
                            text_lengths = torch.LongTensor([len(utt.phoneme_ids)]).to(pl_module.device)
                            sid = None
                            if pl_module.hparams.num_speakers > 1 and utt.speaker_id is not None:
                                if isinstance(utt.speaker_id, torch.Tensor) and utt.speaker_id.ndim == 0:
                                     sid = utt.speaker_id.unsqueeze(0).to(pl_module.device)
                                elif isinstance(utt.speaker_id, int):
                                     sid = torch.LongTensor([utt.speaker_id]).to(pl_module.device)
                                else:
                                     sid = utt.speaker_id.to(pl_module.device)

                            # --- Perform inference ---
                            # Output might be (1, 1, time)
                            audio_tensor_raw = pl_module(text, text_lengths, self.scales, sid=sid).detach().cpu()
                            _LOGGER.debug(f"Raw audio tensor shape from model: {audio_tensor_raw.shape}")

                            # --- ****** FIX FOR 3D TENSOR ****** ---
                            # Check shape and squeeze if necessary to get (time,) or (channels, time)
                            if audio_tensor_raw.ndim == 3:
                                # Most likely case: (batch=1, channels=1, time)
                                if audio_tensor_raw.shape[0] == 1 and audio_tensor_raw.shape[1] == 1:
                                    audio_tensor = audio_tensor_raw.squeeze(0).squeeze(0) # Squeeze both -> (time,)
                                    _LOGGER.debug(f"Squeezed 3D tensor to shape: {audio_tensor.shape}")
                                # Less likely: (batch=1, channels>1, time)
                                elif audio_tensor_raw.shape[0] == 1:
                                     audio_tensor = audio_tensor_raw.squeeze(0) # Squeeze batch -> (channels, time)
                                     _LOGGER.debug(f"Squeezed 3D tensor (batch dim) to shape: {audio_tensor.shape}")
                                # Handle other unexpected 3D shapes
                                else:
                                     _LOGGER.warning(f"Unexpected 3D audio tensor shape {audio_tensor_raw.shape}. Taking first batch element.")
                                     audio_tensor = audio_tensor_raw[0] # Now 2D: (channels, time)

                            elif audio_tensor_raw.ndim == 2:
                                # Expected 2D: (channels, time) or maybe (1, time)
                                audio_tensor = audio_tensor_raw
                                _LOGGER.debug(f"Audio tensor already 2D: {audio_tensor.shape}")
                            elif audio_tensor_raw.ndim == 1:
                                # Expected 1D: (time,)
                                audio_tensor = audio_tensor_raw
                                _LOGGER.debug(f"Audio tensor already 1D: {audio_tensor.shape}")
                            else:
                                _LOGGER.warning(f"Audio tensor has unexpected dimension {audio_tensor_raw.ndim}. Skipping sample.")
                                continue # Skip this sample
                            # --- ****** END FIX ****** ---


                            # --- Scale audio ---
                            max_amp = torch.max(torch.abs(audio_tensor))
                            if max_amp > 1e-4:
                                audio_tensor = audio_tensor * (0.95 / max_amp)

                            sample_rate = pl_module.hparams.sample_rate
                            tag_name = f"val_sample_{i}_step_{step + 1}"

                            # --- Save the audio file ---
                            output_filename = samples_dir / f"{tag_name}.wav"
                            try:
                                # Ensure tensor is 2D (channels, time) for torchaudio.save
                                if audio_tensor.ndim == 1:
                                    # Add channel dimension: (time,) -> (1, time)
                                    audio_tensor_save = audio_tensor.unsqueeze(0)
                                elif audio_tensor.ndim == 2:
                                    # Assume it's already (channels, time) or (1, time)
                                    audio_tensor_save = audio_tensor
                                else:
                                    # This case should be prevented by the shape check above, but check again for safety
                                    _LOGGER.warning(f"Cannot save audio tensor with dimension {audio_tensor.ndim} for sample {output_filename}.")
                                    continue

                                _LOGGER.debug(f"Saving audio sample {output_filename} with shape: {audio_tensor_save.shape}")
                                torchaudio.save(
                                    str(output_filename),
                                    audio_tensor_save,
                                    sample_rate
                                )
                                _LOGGER.info(f"Saved audio sample: {output_filename}")
                            except Exception as e_save:
                                _LOGGER.error(f"Failed to save audio sample {output_filename}: {e_save}")

                            # --- Log to TensorBoard ---
                            if logger_available:
                                try:
                                    # Ensure tensor is 2D (1, time) for logger.experiment.add_audio
                                    if audio_tensor.ndim == 1:
                                         audio_tensor_log = audio_tensor.unsqueeze(0) # (time,) -> (1, time)
                                    elif audio_tensor.ndim == 2:
                                         # If it's (channels>1, time), take the first channel
                                         if audio_tensor.shape[0] > 1:
                                             _LOGGER.debug(f"Logging first channel of multi-channel audio {tag_name}")
                                             audio_tensor_log = audio_tensor[0, :].unsqueeze(0) # (channels, time) -> (1, time)
                                         else:
                                             # Already (1, time)
                                             audio_tensor_log = audio_tensor
                                    else:
                                        _LOGGER.warning(f"Cannot log audio tensor with dimension {audio_tensor.ndim} for sample {tag_name}.")
                                        continue

                                    _LOGGER.debug(f"Logging audio sample {tag_name} with shape: {audio_tensor_log.shape}")
                                    trainer.logger.experiment.add_audio(
                                        tag_name,
                                        audio_tensor_log,
                                        global_step=step + 1,
                                        sample_rate=sample_rate,
                                    )
                                    _LOGGER.debug(f"Logged audio to logger: {tag_name}")
                                except Exception as e_log:
                                     _LOGGER.error(f"Failed to log audio sample {tag_name} to logger: {e_log}")

                        except Exception as e_inner:
                            _LOGGER.error(f"Error generating/processing sample {i} (index {idx}) at step {step + 1}: {e_inner}", exc_info=True)

            except Exception as e_outer:
                 _LOGGER.error(f"Error during audio sample generation outer loop at step {step + 1}: {e_outer}", exc_info=True)
            finally:
                pl_module.train() # Ensure model is back in train mode

class LitProgressBar(ProgressBarBase):
    def __init__(self):
        super().__init__()
        self.enable = True
    
    def disable(self):
        self.enable = False
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=None):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        percent = (self.train_batch_idx / self.total_train_batches) * 100
        sys.stdout.flush()
        sys.stdout.write(f'{percent:.01f} percent complete \r')
                
def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, help="Path to pre-processed dataset directory"
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        help="Save checkpoint every N epochs (default: 1)",
    )
    parser.add_argument(
        "--quality",
        default="optimized",
        choices=("x-low", "x-low_extra", "medium", "high", "efficient", "optimized"),
        help="Quality/size of model (default: medium)",
    )
    parser.add_argument(
        "--resume_from_single_speaker_checkpoint",
        help="For multi-speaker models only. Converts a single-speaker checkpoint to multi-speaker and resumes training",
    )
    parser.add_argument(
        "--resume_checkpoint",
        help="Resume training from an optimized checkpoint (supports efficient models)",
    )
    parser.add_argument(
        "--pruning-factor",
        type=float,
        default=0.0,
        help="The pruning factor used when creating the efficient model (default: 0.5)",
    )
    parser.add_argument(
        "--optimized-model-dir",
        help="Path to an optimized model directory containing model.pt/ckpt and config.json",
    )
    parser.add_argument(
        "--keep-layers",
        type=int,
        default=6,
        help="Number of transformer layers for optimized models (default: 3)",
    )
    parser.add_argument(
        "--from_scratch", 
        action="store_true",
        help="Train model from scratch with newly initialized weights instead of loading a checkpoint"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float,
        default=2.0e-4,
        help="Learning rate - HIGH for from scratch 2e-4"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float,
        default=0.01,
        help="Weight decay to avoid overfitting"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Cosine Warmup ratio"
    )
    parser.add_argument(
        "--cosine_scheduler",
        type=bool,
        default=True,
        help="Use Cosine instead of standard Piper optimizer/scheduler"
    )
    parser.add_argument(
        "--smart_init",
        action="store_true",
        help="Use Steve's smart initializing or Pytorch"
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=2000,
        help="Generate 2 samples for every sample_steps steps"
    )
    parser.add_argument(
        "--max_epoch_keeps",
        type=int,
        default=50,
        help="Maximum good epochs to keep"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for DataLoader"
    )
    # New arguments for better resume functionality
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="Automatically resume from the most recent checkpoint in the checkpoints directory"
    )
    
    Trainer.add_argparse_args(parser)
    VitsModel.add_model_specific_args(parser)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    _LOGGER.debug(args)

    args.dataset_dir = Path(args.dataset_dir)
    if not args.default_root_dir:
        args.default_root_dir = args.dataset_dir

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    # Determine config path - use optimized model config if provided
    config_path = args.dataset_dir / "config.json"
    if args.optimized_model_dir:
        optimized_config = Path(args.optimized_model_dir) / "config.json"
        if optimized_config.exists():
            config_path = optimized_config
            _LOGGER.info(f"Using optimized model config: {config_path}")

    dataset_path = args.dataset_dir / "dataset.jsonl"

    with open(config_path, "r", encoding="utf-8") as config_file:
        # See preprocess.py for format
        config = json.load(config_file)
        num_symbols = int(config["num_symbols"])
        num_speakers = int(config["num_speakers"])
        sample_rate = int(config["audio"]["sample_rate"])
        
        # Check for optimization metadata
        optimization_info = {}
        if "optimization" in config:
            optimization_info = config["optimization"]
            _LOGGER.info(f"Found optimization info: {optimization_info}")

    # Set up callbacks
    callbacks = []
    
    # Add LR Monitor to track scheduler behavior
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Add progress bars
    bar = LitProgressBar()
    callbacks.append(bar)
    
    # RichProgressBar for prettier display
    if not any(isinstance(cb, RichProgressBar) for cb in callbacks):
        callbacks.append(RichProgressBar())
    
    # Configure checkpoint saving (critical for proper state preservation)
    checkpoint_callback = None
    if args.checkpoint_epochs is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(args.weights_save_path) if args.weights_save_path else Path(args.default_root_dir) / "checkpoints",
            filename='piper-{epoch:03d}-{val_loss:.2f}',  # Unique names
            monitor='val_loss',             # Metric to monitor
            mode='min',                     # Minimize validation loss
            save_top_k=args.max_epoch_keeps, # Keep top N checkpoints
            every_n_epochs=args.checkpoint_epochs, # Save frequency
            save_last=True,                # Also save the very last one
            save_weights_only=False        # This is critical: save optimizer state too
        )
        callbacks.append(checkpoint_callback)
        _LOGGER.info(
            f"Checkpoints will be saved every {args.checkpoint_epochs} epoch(s), "
            f"keeping top {args.max_epoch_keeps} based on val_loss"
        )
    
    # Add audio sample logger
    audio_logger_callback = AudioSampleLogger(frequency=args.sample_steps, num_samples=2)
    callbacks.append(audio_logger_callback)
    
    # Determine if we should resume from a checkpoint
    resume_checkpoint_path = None
    
    # Check for explicit resume path
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        resume_checkpoint_path = args.resume_checkpoint
        _LOGGER.info(f"Will resume from specified checkpoint: {resume_checkpoint_path}")
    
    # Check for auto-resume from last.ckpt
    elif args.auto_resume:
        last_checkpoint_path = Path(args.default_root_dir) / "checkpoints" / "last.ckpt"
        if last_checkpoint_path.exists():
            resume_checkpoint_path = str(last_checkpoint_path)
            _LOGGER.info(f"Auto-resuming from checkpoint: {resume_checkpoint_path}")
        else:
            _LOGGER.info("No checkpoint found for auto-resume. Will start fresh training.")
    
    # Create trainer with appropriate settings for resuming
    trainer_kwargs = vars(args)
    trainer_kwargs['callbacks'] = callbacks
    
    # Add resume_from_checkpoint if needed
    if resume_checkpoint_path:
        trainer_kwargs['resume_from_checkpoint'] = resume_checkpoint_path
    
    trainer = Trainer(**trainer_kwargs)
    
    # Initialize model arguments
    dict_args = vars(args)
    
    # Set model architecture based on quality            
    if args.quality == "x-low":
        dict_args["hidden_channels"] = 96
        dict_args["inter_channels"] = 96
        dict_args["filter_channels"] = 384
    
    # Set model architecture based on quality
    elif args.quality == "x-low_extra":
        dict_args["hidden_channels"] = 96
        dict_args["inter_channels"] = 96
        dict_args["filter_channels"] = 384
        dict_args["n_layers"] = args.keep_layers
    
    elif args.quality == "high":
        dict_args["resblock"] = "1"
        dict_args["resblock_kernel_sizes"] = (3, 7, 11)
        dict_args["resblock_dilation_sizes"] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        )
        dict_args["upsample_rates"] = (8, 8, 2, 2)
        dict_args["upsample_initial_channel"] = 512
        dict_args["upsample_kernel_sizes"] = (16, 16, 4, 4)
        
    elif args.quality == "optimized":
        # Handle optimized models with reduced layers and dimensions
        # First check for optimization info in config
        pruning_factor = args.pruning_factor
        n_layers = args.keep_layers  # Use the value from --keep-layers argument
        
        # Use config values if available
        if optimization_info:
            if "hidden_dim" in optimization_info:
                hidden_dim = int(optimization_info["hidden_dim"])
                dict_args["hidden_channels"] = hidden_dim
                dict_args["inter_channels"] = hidden_dim
                _LOGGER.debug(f"Using hidden dimension from config: {hidden_dim}")
            
            if "filter_dim" in optimization_info:
                filter_dim = int(optimization_info["filter_dim"])
                dict_args["filter_channels"] = filter_dim
                _LOGGER.debug(f"Using filter dimension from config: {filter_dim}")
            
            if "n_layers" in optimization_info:
                n_layers = int(optimization_info["n_layers"])
                dict_args["n_layers"] = n_layers  # Important: explicitly set layers
                _LOGGER.debug(f"Using layer count from config: {n_layers}")
            
            # Get upsample channel info
            if "upsample_initial_channel" in config["audio"]:
                target_channel = int(config["audio"]["upsample_initial_channel"])
                _LOGGER.debug(f"Using upsample channel from config: {target_channel}")
            else:
                # Calculate from pruning factor
                original_channel = 512
                target_channel = int(original_channel * (1 - pruning_factor))
            
            # Always explicitly set the number of layers
            dict_args["n_layers"] = n_layers
            
        else:
            # Calculate dimensions if not found in config
            original_hidden = 192
            target_hidden = int(original_hidden * (1 - pruning_factor))
            dict_args["hidden_channels"] = target_hidden
            dict_args["inter_channels"] = target_hidden
            
            original_filter = 768
            target_filter = int(original_filter * (1 - pruning_factor))
            dict_args["filter_channels"] = target_filter
            
            # Calculate upsampling dimensions
            original_channel = 512
            target_channel = int(original_channel * (1 - pruning_factor))
            
            # Explicitly set the number of layers
            dict_args["n_layers"] = n_layers
        
        # Set remaining parameters
        dict_args["resblock"] = "1"
        dict_args["resblock_kernel_sizes"] = (3, 7, 11)
        dict_args["resblock_dilation_sizes"] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        )
        dict_args["upsample_rates"] = (8, 8, 4)  # 3-stage upsampling
        dict_args["upsample_initial_channel"] = target_channel
        dict_args["upsample_kernel_sizes"] = (16, 16, 8)
        
        _LOGGER.info(
            f"Optimized model settings: layers={dict_args.get('n_layers', n_layers)}, "
            f"hidden={dict_args.get('hidden_channels', target_hidden if 'target_hidden' in locals() else 'unknown')}, "
            f"filter={dict_args.get('filter_channels', target_filter if 'target_filter' in locals() else 'unknown')}, "
            f"channels={target_channel}, "
            f"pruning_factor={pruning_factor:.2f}"
        )
        
    # Create the model
    model = VitsModel(
        num_symbols=num_symbols,
        num_speakers=num_speakers,
        sample_rate=sample_rate,
        dataset=[dataset_path],
        **dict_args,
    )
    
    # Add this debug info
    _LOGGER.info(f"Model created with parameters:")
    _LOGGER.info(f"  - hidden_channels: {model.model_g.hidden_channels}")
    _LOGGER.info(f"  - n_layers: {model.model_g.n_layers}")
    _LOGGER.info(f"  - filter_channels: {model.model_g.filter_channels}")
    
    # Skip weight loading if we're resuming from a checkpoint (Lightning will handle this)
    if resume_checkpoint_path is None:
        # If from_scratch is specified, initialize weights and skip loading checkpoints
        if args.from_scratch:
            if args.smart_init:
                _LOGGER.info("Training from scratch with STEVE's VITS weight initialization")
                model = init_weights_vits(model)
            else:
                _LOGGER.info("Training from scratch with PyTorch's VITS weight initialization")
                model = init_weights_pytorch_default(model)

            _LOGGER.info("Model initialized with random weights for from-scratch training")
        
        # Handle loading from optimized model dir (only if not training from scratch)
        elif args.optimized_model_dir:
            model_path = Path(args.optimized_model_dir) / "model.pt"
            if not model_path.exists():
                model_path = Path(args.optimized_model_dir) / "model.ckpt"
            
            if model_path.exists():
                _LOGGER.info(f"Loading optimized model from {model_path}")
                try:
                    # Load optimized model
                    checkpoint = torch.load(model_path, map_location='cpu')
                    
                    if isinstance(checkpoint, dict) and 'model' in checkpoint:
                        state_dict = checkpoint['model']
                        
                        # Extract generator and discriminator weights
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
                        
                        # Load weights with improved flexibility
                        if model_g_dict:
                            load_state_dict_flexible(model.model_g, model_g_dict)
                        if model_d_dict:
                            load_state_dict_flexible(model.model_d, model_d_dict)
                        
                        _LOGGER.info("Successfully loaded weights from optimized model")
                    else:
                        _LOGGER.error("Invalid model format in optimized model directory")
                except Exception as e:
                    _LOGGER.error(f"Error loading optimized model: {e}")
                    import traceback
                    traceback.print_exc()

        # Handle loading from single-speaker checkpoint (only if not training from scratch)
        elif args.resume_from_single_speaker_checkpoint:
            assert (
                num_speakers > 1
            ), "--resume_from_single_speaker_checkpoint is only for multi-speaker models. Use --resume_checkpoint for single-speaker models."

            # Load single-speaker checkpoint
            _LOGGER.debug(
                "Resuming from single-speaker checkpoint: %s",
                args.resume_from_single_speaker_checkpoint,
            )
            model_single = VitsModel.load_from_checkpoint(
                args.resume_from_single_speaker_checkpoint,
                dataset=None,
            )
            g_dict = model_single.model_g.state_dict()
            for key in list(g_dict.keys()):
                # Remove keys that can't be copied over due to missing speaker embedding
                if (
                    key.startswith("dec.cond")
                    or key.startswith("dp.cond")
                    or ("enc.cond_layer" in key)
                ):
                    g_dict.pop(key, None)

            # Copy over the multi-speaker model, excluding keys related to the
            # speaker embedding (which is missing from the single-speaker model).
            load_state_dict(model.model_g, g_dict)
            load_state_dict(model.model_d, model_single.model_d.state_dict())
            _LOGGER.info(
                "Successfully converted single-speaker checkpoint to multi-speaker"
            )

    # Start training with properly configured trainer
    trainer.fit(model)


if __name__ == "__main__":
    main()