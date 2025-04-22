#!/usr/bin/env python3
"""
Knowledge distillation from a medium-size Piper TTS teacher to an x-low Piper student.

This script provides comprehensive distillation capabilities including:
- Multi-GPU training with DDP
- Configurable distillation parameters (alpha, temperature)
- Advanced optimizer settings (lr, weight decay, scheduler)
- Regular audio sample generation
- TensorBoard logging
- Checkpoint management
- Progress tracking

Usage example:
  python -m piper_train.distill \
    --teacher-checkpoint path/to/teacher.ckpt \
    --output-dir outputs/distill \
    --max-epochs 200 \
    --gpus 2 \
    --batch-size 8 \
    --accumulate-grad-batches 4 \
    --learning-rate 2e-4 \
    --weight-decay 0.01 \
    --distill-alpha 1.0 \
    --distill-temp 1.0 \
    --sample-steps 2000 \
    --from-scratch \
    --student-quality x-low
"""

import argparse
import logging
import math
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from .vits.lightning import VitsModel
from .vits.utils import audio_float_to_int16
from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .config import ModelAudioConfig
from .__main__ import init_weights_vits, init_weights_pytorch_default

_LOGGER = logging.getLogger("piper_train.distill")


class AudioSampleCallback(pl.Callback):
    """Generate and log audio samples during training."""
    
    def __init__(
        self, 
        frequency: int = 2000,
        num_samples: int = 2,
        scales: List[float] = [0.667, 1.0, 0.8]
    ):
        super().__init__()
        self.frequency = frequency
        self.num_samples = num_samples
        self.scales = scales
        _LOGGER.info(f"Will generate {num_samples} samples every {frequency} steps")
        
    def on_train_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: Dict, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """Generate and log samples at specified frequency."""
        if not trainer.is_global_zero:
            return
            
        step = trainer.global_step
        if step > 0 and step % self.frequency == 0:
            self._generate_samples(trainer, pl_module, step)
    
    @rank_zero_only
    def _generate_samples(self, trainer, pl_module, step):
        """Generate audio samples on the main process only."""
        _LOGGER.info(f"Generating student and teacher audio samples at step {step}")
        
        # Ensure we have a validation dataset
        if not hasattr(pl_module.student, '_val_dataset') or pl_module.student._val_dataset is None:
            _LOGGER.warning("No validation dataset available for audio samples")
            return
        
        if len(pl_module.student._val_dataset) == 0:
            _LOGGER.warning("Validation dataset is empty, cannot generate samples")
            return
            
        # Create output directory
        samples_dir = Path(trainer.default_root_dir) / "samples" / f"step_{step}"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Set models to eval mode
        pl_module.eval()
        
        try:
            with torch.no_grad():
                val_dataset = pl_module.student._val_dataset
                num_to_sample = min(self.num_samples, len(val_dataset))
                
                for i in range(num_to_sample):
                    test_utt = val_dataset[i]
                    
                    # Prepare inputs
                    text = test_utt.phoneme_ids.unsqueeze(0).to(pl_module.device)
                    text_lengths = torch.LongTensor([len(test_utt.phoneme_ids)]).to(pl_module.device)
                    sid = test_utt.speaker_id.to(pl_module.device) if test_utt.speaker_id is not None else None
                    
                    # Generate from teacher and student
                    t_audio = pl_module.teacher(text, text_lengths, self.scales, sid=sid)
                    s_audio = pl_module.student(text, text_lengths, self.scales, sid=sid)
                    
                    # Process audio for saving
                    for audio, prefix in [(t_audio, "teacher"), (s_audio, "student")]:
                        # Scale to prevent clipping
                        max_amp = torch.max(torch.abs(audio))
                        if max_amp > 1e-4:
                            audio = audio * (0.95 / max_amp)
                            
                        # Ensure audio is (channels, time)
                        if audio.ndim == 3:  # (batch, channels, time)
                            audio = audio.squeeze(0)
                        elif audio.ndim == 1:  # (time,)
                            audio = audio.unsqueeze(0)
                            
                        # Save audio
                        filename = samples_dir / f"{prefix}_sample_{i}.wav"
                        sample_rate = pl_module.student.hparams.sample_rate
                        torchaudio.save(str(filename), audio.cpu(), sample_rate)
                        
                        # Log to TensorBoard
                        if hasattr(trainer, 'logger') and trainer.logger is not None:
                            if isinstance(trainer.logger, TensorBoardLogger):
                                trainer.logger.experiment.add_audio(
                                    f"{prefix}_sample_{i}", 
                                    audio.unsqueeze(0),
                                    step,
                                    sample_rate=sample_rate
                                )
                                
                    # Log text for reference
                    if hasattr(test_utt, 'text') and test_utt.text:
                        with open(samples_dir / f"sample_{i}_text.txt", 'w', encoding='utf-8') as f:
                            f.write(test_utt.text)
                            
                _LOGGER.info(f"Saved {num_to_sample} sample pairs to {samples_dir}")
        except Exception as e:
            _LOGGER.error(f"Error generating samples: {e}")
        finally:
            # Set models back to training mode
            pl_module.train()


class VitsDistillationModule(pl.LightningModule):
    """
    Lightning module for knowledge distillation from a teacher VITS model to a student.
    """
    
    def __init__(
        self,
        teacher_checkpoint: str,
        student_hparams: Dict[str, Any],
        scales: List[float] = [0.667, 1.0, 0.8],
        distill_alpha: float = 1.0,
        distill_temp: float = 1.0,
        from_scratch: bool = False,
        smart_init: bool = True,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        min_lr_ratio: float = 0.05,
        grad_clip: Optional[float] = None,
        student_ckpt: Optional[str] = None
    ):
        """
        Initialize the distillation module.
        
        Args:
            teacher_checkpoint: Path to the trained medium-size teacher checkpoint
            student_hparams: Hyperparameters for the student model
            scales: Inference scales [noise_scale, length_scale, noise_scale_w]
            distill_alpha: Weight for distillation loss
            distill_temp: Temperature for distillation (soften teacher output)
            from_scratch: Whether to initialize student from scratch
            smart_init: Use smart initialization for student weights
            learning_rate: Peak learning rate for training
            weight_decay: Weight decay coefficient for AdamW
            warmup_ratio: Percentage of training for LR warmup
            min_lr_ratio: Minimum LR as ratio of peak LR
            grad_clip: Gradient clipping value (None to disable)
            student_ckpt: Optional pre-trained student checkpoint to resume from
        """
        super().__init__()
        self.save_hyperparameters(ignore=['teacher_checkpoint', 'student_hparams', 'student_ckpt'])
        
        # Load and freeze teacher
        _LOGGER.info(f"Loading teacher model from {teacher_checkpoint}")
        self.teacher = VitsModel.load_from_checkpoint(teacher_checkpoint)
        self.teacher.eval()  # Set to evaluation mode
        for p in self.teacher.parameters():
            p.requires_grad = False
            
        # Create student model
        _LOGGER.info("Creating student model")
        self.student = VitsModel(**student_hparams)
        
        # Initialize or load student model
        if student_ckpt:
            _LOGGER.info(f"Loading pre-trained student from {student_ckpt}")
            student_state = torch.load(student_ckpt, map_location='cpu')
            if 'state_dict' in student_state:
                self.student.load_state_dict(student_state['state_dict'])
            else:
                _LOGGER.warning("Could not find state_dict in checkpoint, using checkpoint directly")
                self.student.load_state_dict(student_state)
        elif from_scratch:
            _LOGGER.info("Initializing student from scratch")
            if smart_init:
                _LOGGER.info("Using smart weight initialization")
                init_weights_vits(self.student.model_g)
            else:
                _LOGGER.info("Using PyTorch default weight initialization")
                init_weights_pytorch_default(self.student.model_g)
        else:
            _LOGGER.info("Keeping pre-initialized student weights")
            
        # Store hyperparameters
        self.scales = scales
        self.distill_alpha = distill_alpha
        self.distill_temp = distill_temp
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        self.grad_clip = grad_clip
        
        # Initialize metrics
        self.train_step_loss = 0.0
        self.val_step_loss = 0.0

    def training_step(self, batch, batch_idx):
        """Execute a single training step."""
        start_time = time.time()
        
        # Extract input data
        x = batch.phoneme_ids
        x_lengths = batch.phoneme_lengths
        speaker_ids = batch.speaker_ids if batch.speaker_ids is not None else None
        spectrograms = batch.spectrograms
        spec_lengths = batch.spectrogram_lengths
        
        # Teacher inference (with no gradient tracking)
        with torch.no_grad():
            # Get teacher-generated mel spectrograms
            t_audio, _, _, _, _, _, _ = self.teacher.model_g.infer(
                x, x_lengths, 
                noise_scale=self.scales[0],
                length_scale=self.scales[1], 
                noise_scale_w=self.scales[2],
                sid=speaker_ids
            )
            
            # Convert to mel spectrograms
            t_mel = mel_spectrogram_torch(
                t_audio.squeeze(1),
                self.student.hparams.filter_length,
                self.student.hparams.mel_channels,
                self.student.hparams.sample_rate,
                self.student.hparams.hop_length,
                self.student.hparams.win_length,
                self.student.hparams.mel_fmin,
                self.student.hparams.mel_fmax,
                center=False
            )
            
        # Student training forward pass
        s_output = self.student.model_g(x, x_lengths, spectrograms, spec_lengths, speaker_ids)
        s_audio = s_output[0]  # Get generated audio
        
        # Convert student output to mel
        s_mel = mel_spectrogram_torch(
            s_audio.squeeze(1),
            self.student.hparams.filter_length,
            self.student.hparams.mel_channels,
            self.student.hparams.sample_rate,
            self.student.hparams.hop_length,
            self.student.hparams.win_length,
            self.student.hparams.mel_fmin,
            self.student.hparams.mel_fmax,
            center=False
        )
        
        # Original VITS losses from student
        original_losses = self.student.training_step_g(batch)
        
        # Align teacher and student mel time dimension
        min_t = min(t_mel.size(-1), s_mel.size(-1))
        t_mel = t_mel[..., :min_t]
        s_mel = s_mel[..., :min_t]
        
        # Distillation loss (MSE on mel with temperature)
        if self.distill_temp != 1.0:
            # Apply temperature scaling if needed
            distill_loss = F.mse_loss(
                s_mel / self.distill_temp,
                t_mel / self.distill_temp
            )
        else:
            distill_loss = F.mse_loss(s_mel, t_mel)
            
        # Combine losses
        # Scale distillation loss by alpha factor
        total_loss = original_losses + distill_loss * self.distill_alpha
        
        # Calculate duration metrics
        step_time = time.time() - start_time
        examples_per_second = x.size(0) / step_time
        
        # Log metrics
        self.train_step_loss = total_loss.item()
        self.log("train/loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("train/distill_loss", distill_loss, prog_bar=True, sync_dist=True)
        self.log("train/original_loss", original_losses, sync_dist=True)
        self.log("train/examples_per_second", examples_per_second, sync_dist=True)
        
        # Calculate and log gradient norm if clipping is enabled
        if self.grad_clip is not None:
            self.log("train/grad_norm", self.get_grad_norm(), sync_dist=True)
            
        return total_loss

    def validation_step(self, batch, batch_idx):
        """Execute a single validation step."""
        # Extract input data
        x = batch.phoneme_ids
        x_lengths = batch.phoneme_lengths
        speaker_ids = batch.speaker_ids if batch.speaker_ids is not None else None
        spectrograms = batch.spectrograms
        spec_lengths = batch.spectrogram_lengths
        
        # No gradients for validation
        with torch.no_grad():
            # Teacher inference
            t_audio, _, _, _, _, _, _ = self.teacher.model_g.infer(
                x, x_lengths, 
                noise_scale=self.scales[0],
                length_scale=self.scales[1], 
                noise_scale_w=self.scales[2],
                sid=speaker_ids
            )
            t_mel = mel_spectrogram_torch(
                t_audio.squeeze(1),
                self.student.hparams.filter_length,
                self.student.hparams.mel_channels,
                self.student.hparams.sample_rate,
                self.student.hparams.hop_length,
                self.student.hparams.win_length,
                self.student.hparams.mel_fmin,
                self.student.hparams.mel_fmax,
                center=False
            )
            
            # Student inference (using VITS forward mode to get original losses)
            s_output = self.student.model_g(x, x_lengths, spectrograms, spec_lengths, speaker_ids)
            s_audio = s_output[0]
            
            # Original student validation loss
            original_val_loss = self.student.validation_step(batch, batch_idx)
            
            # Calculate distillation loss
            s_mel = mel_spectrogram_torch(
                s_audio.squeeze(1),
                self.student.hparams.filter_length,
                self.student.hparams.mel_channels,
                self.student.hparams.sample_rate,
                self.student.hparams.hop_length,
                self.student.hparams.win_length,
                self.student.hparams.mel_fmin,
                self.student.hparams.mel_fmax,
                center=False
            )
            
            # Align time dimension
            min_t = min(t_mel.size(-1), s_mel.size(-1))
            t_mel = t_mel[..., :min_t]
            s_mel = s_mel[..., :min_t]
            
            # Calculate distillation loss
            distill_loss = F.mse_loss(s_mel, t_mel)
            total_val_loss = original_val_loss + distill_loss * self.distill_alpha
            
            # Store for logging
            self.val_step_loss = total_val_loss.item()
            
            # Log validation metrics
            self.log("val/loss", total_val_loss, prog_bar=True, sync_dist=True)
            self.log("val/distill_loss", distill_loss, prog_bar=True, sync_dist=True)
            self.log("val/original_loss", original_val_loss, sync_dist=True)
            
            return total_val_loss

    def get_grad_norm(self):
        """Calculate gradient norm for all student parameters."""
        total_norm = 0.0
        for p in self.student.parameters():
            if p.grad is not None and p.requires_grad:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def on_train_epoch_start(self):
        """Called at the beginning of each training epoch."""
        # Log current learning rate
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group['lr']
            self.log('train/learning_rate', lr, sync_dist=True)
            break
    
    def configure_optimizers(self):
        """Set up optimizer and learning rate scheduler."""
        # Set up optimizer (AdamW with weight decay)
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.learning_rate,
            betas=self.student.hparams.betas,
            eps=self.student.hparams.eps,
            weight_decay=self.weight_decay
        )
        
        # Calculate training steps for scheduler
        if not hasattr(self.trainer, "estimated_stepping_batches"):
            # For older Lightning versions
            if self.trainer.max_steps and self.trainer.max_steps > 0:
                max_steps = self.trainer.max_steps
            else:
                max_steps = (
                    len(self.trainer.datamodule.train_dataloader()) * 
                    self.trainer.max_epochs / 
                    self.trainer.accumulate_grad_batches
                )
        else:
            # Newer Lightning versions
            max_steps = self.trainer.estimated_stepping_batches
        
        warmup_steps = int(max_steps * self.warmup_ratio)
        
        # Define cosine schedule with warmup
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                # Don't go below min_lr_ratio of the base learning rate
                return max(self.min_lr_ratio, cosine_decay)
        
        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda),
            "interval": "step",
            "frequency": 1,
            "name": "cosine_warmup_lr"
        }
        
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        """Get the training data loader from the student."""
        return self.student.train_dataloader()

    def val_dataloader(self):
        """Get the validation data loader from the student."""
        return self.student.val_dataloader()
    
    def on_after_backward(self):
        """Called after backward pass - clip gradients if enabled."""
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)


class CustomProgressBar(TQDMProgressBar):
    """
    Custom progress bar showing training loss and step throughput.
    """
    def get_metrics(self, trainer, model):
        # Get the metrics from the parent class
        items = super().get_metrics(trainer, model)
        
        # Add additional metrics if available on the model
        if hasattr(model, 'train_step_loss'):
            items["loss"] = f"{model.train_step_loss:.4f}"
        
        return items


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Distill a Piper TTS model from medium to x-low size")
    
    # Model paths
    parser.add_argument(
        "--teacher-checkpoint", type=str, required=True,
        help="Path to the trained medium-size teacher checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--student-checkpoint", type=str, default=None,
        help="Path to an existing student checkpoint to resume from (optional)"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save distilled student checkpoints"
    )
    
    # Student architecture
    parser.add_argument(
        "--student-quality", type=str, 
        choices=["x-low", "x-low_extra", "medium", "high", "efficient", "optimized"],
        default="x-low",
        help="Quality/size of the student model (default: x-low)"
    )
    parser.add_argument(
        "--from-scratch", action="store_true",
        help="Initialize student weights from scratch rather than using pre-initialized weights"
    )
    parser.add_argument(
        "--smart-init", action="store_true",
        help="Use smart weight initialization for the student model (if --from-scratch)"
    )
    parser.add_argument(
        "--keep-layers", type=int, default=3,
        help="Number of transformer layers for optimized/x-low_extra student models (default: 3)"
    )
    parser.add_argument(
        "--pruning-factor", type=float, default=0.0,
        help="Pruning factor for optimized student model (default: 0.0)"
    )
    
    # Training parameters
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--val-check-interval", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=8)
    
    # Distillation parameters
    parser.add_argument("--distill-alpha", type=float, default=1.0,
                       help="Weight for distillation loss term")
    parser.add_argument("--distill-temp", type=float, default=1.0,
                       help="Temperature for distillation (higher = softer logits)")
    parser.add_argument(
        "--scales", type=float, nargs=3, default=[0.667, 1.0, 0.8],
        help="[noise_scale, length_scale, noise_scale_w]"
    )
    
    # Sampling and checkpointing
    parser.add_argument("--sample-steps", type=int, default=2000,
                       help="Generate audio samples every N steps")
    parser.add_argument("--num-samples", type=int, default=2,
                       help="Number of audio samples to generate")
    parser.add_argument("--checkpoint-epochs", type=float, default=0.5,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--save-top-k", type=int, default=5,
                       help="Number of best checkpoints to keep")
    
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--precision", type=str, default="32", 
                        choices=["32", "16", "bf16"],
                        help="Floating point precision for training")
    
    return parser.parse_args()


def setup_student_hparams(args, teacher_hparams):
    """Set up student hyperparameters based on quality setting."""
    student_hparams = teacher_hparams.copy()
    
    # Configure student architecture based on quality setting
    if args.student_quality == "x-low":
        student_hparams.update({
            "hidden_channels": 96,
            "inter_channels": 96,
            "filter_channels": 384,
        })
    
    elif args.student_quality == "x-low_extra":
        student_hparams.update({
            "hidden_channels": 96,
            "inter_channels": 96,
            "filter_channels": 384,
            "n_layers": args.keep_layers,
        })
    
    elif args.student_quality == "high":
        # Configure high-quality model
        student_hparams.update({
            "resblock": "1",
            "resblock_kernel_sizes": (3, 7, 11),
            "resblock_dilation_sizes": (
                (1, 3, 5),
                (1, 3, 5),
                (1, 3, 5),
            ),
            "upsample_rates": (8, 8, 2, 2),
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": (16, 16, 4, 4),
        })
        
    elif args.student_quality == "optimized":
        # Calculate dimensions based on pruning factor
        pruning_factor = args.pruning_factor
        n_layers = args.keep_layers
        
        # Calculate hidden dimension
        original_hidden = 192
        target_hidden = int(original_hidden * (1 - pruning_factor))
        
        # Calculate filter dimension
        original_filter = 768
        target_filter = int(original_filter * (1 - pruning_factor))
        
        # Calculate upsampling dimension
        original_channel = 512
        target_channel = int(original_channel * (1 - pruning_factor))
        
        # Update student parameters
        student_hparams.update({
            "hidden_channels": target_hidden,
            "inter_channels": target_hidden,
            "filter_channels": target_filter,
            "n_layers": n_layers,
            "resblock": "1",
            "resblock_kernel_sizes": (3, 7, 11),
            "resblock_dilation_sizes": (
                (1, 3, 5),
                (1, 3, 5),
                (1, 3, 5),
            ),
            "upsample_rates": (8, 8, 4),  # 3-stage upsampling
            "upsample_initial_channel": target_channel,
            "upsample_kernel_sizes": (16, 16, 8),
        })
        
    # Override batch size and num_workers if specified
    if args.batch_size:
        student_hparams["batch_size"] = args.batch_size
    
    student_hparams["num_workers"] = args.num_workers
        
    return student_hparams


def main():
    """Main entry point for distillation script."""
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Set random seed for reproducibility
    pl.seed_everything(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load teacher to extract hyperparameters
    _LOGGER.info(f"Loading teacher model from {args.teacher_checkpoint}...")
    teacher = VitsModel.load_from_checkpoint(args.teacher_checkpoint)
    teacher_hparams = dict(vars(teacher.hparams))
    
    # Configure student hyperparameters
    _LOGGER.info(f"Setting up student with quality '{args.student_quality}'")
    student_hparams = setup_student_hparams(args, teacher_hparams)
    
    # Log model hyperparameters
    _LOGGER.info(f"Teacher model: hidden={teacher_hparams['hidden_channels']}, "
                 f"layers={teacher_hparams['n_layers']}, "
                 f"filter={teacher_hparams['filter_channels']}")
    
    _LOGGER.info(f"Student model: hidden={student_hparams['hidden_channels']}, "
                 f"layers={student_hparams['n_layers']}, "
                 f"filter={student_hparams['filter_channels']}")
    
    # Initialize distillation module
    model = VitsDistillationModule(
        teacher_checkpoint=args.teacher_checkpoint,
        student_hparams=student_hparams,
        scales=args.scales,
        distill_alpha=args.distill_alpha,
        distill_temp=args.distill_temp,
        from_scratch=args.from_scratch,
        smart_init=args.smart_init,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        grad_clip=args.grad_clip,
        student_ckpt=args.student_checkpoint
    )
    
    # Set up callbacks
    callbacks = [
        # Model checkpoints based on validation loss
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="distill-{epoch:03d}-{val/loss:.4f}",
            save_top_k=args.save_top_k,
            monitor="val/loss",
            mode="min",
            every_n_epochs=args.checkpoint_epochs,
            save_last=True,
        ),
        # Audio sample generation
        AudioSampleCallback(
            frequency=args.sample_steps,
            num_samples=args.num_samples,
            scales=args.scales
        ),
        # Learning rate monitoring
        LearningRateMonitor(logging_interval="step"),
        # Progress bar
        RichProgressBar()
    ]
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="logs"
    )
    
    # Configure trainer
    trainer_kwargs = {
        "max_epochs": args.max_epochs,
        "callbacks": callbacks,
        "logger": logger,
        "log_every_n_steps": 25,
        "val_check_interval": args.val_check_interval,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "gradient_clip_val": args.grad_clip,
        "default_root_dir": output_dir,
    }
    
    # Handle precision
    if args.precision == "16":
        trainer_kwargs["precision"] = 16
    elif args.precision == "bf16":
        trainer_kwargs["precision"] = "bf16-mixed"
    else:
        trainer_kwargs["precision"] = 32
    
    # Add max_steps if specified
    if args.max_steps:
        trainer_kwargs["max_steps"] = args.max_steps
    
    # Configure GPU training
    if args.gpus > 1:
        # Use DDP for multi-GPU training
        trainer_kwargs["devices"] = args.gpus
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=False)
    elif args.gpus == 1:
        trainer_kwargs["devices"] = 1
        trainer_kwargs["accelerator"] = "gpu"
    else:
        trainer_kwargs["accelerator"] = "cpu"
    
    # Initialize trainer
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Start training
    trainer.fit(model, ckpt_path=args.student_checkpoint)
    
    # Log final model path
    best_checkpoint = trainer.checkpoint_callback.best_model_path
    _LOGGER.info(f"Training complete. Best model saved at: {best_checkpoint}")
    
    # Copy the best model to the output directory with a friendly name
    if best_checkpoint:
        # Get validation loss from the checkpoint name
        import re
        val_loss_match = re.search(r'val-loss=(\d+\.\d+)', best_checkpoint)
        if val_loss_match:
            val_loss = val_loss_match.group(1)
        else:
            val_loss = "unknown"
            
        # Create friendly name
        friendly_path = output_dir / f"distilled_{args.student_quality}_model_{val_loss}.ckpt"
        import shutil
        try:
            shutil.copy2(best_checkpoint, friendly_path)
            _LOGGER.info(f"Best model copied to: {friendly_path}")
        except Exception as e:
            _LOGGER.error(f"Failed to copy best model: {e}")


if __name__ == "__main__":
    main()
