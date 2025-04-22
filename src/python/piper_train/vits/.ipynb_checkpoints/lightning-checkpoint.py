import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import pytorch_lightning as pl
import torch
from torch import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from .commons import slice_segments
from .dataset import Batch, PiperDataset, UtteranceCollate
from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .models import MultiPeriodDiscriminator, SynthesizerTrn

_LOGGER = logging.getLogger("vits.lightning")

import warnings
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()` before `optimizer.step()`")

class VitsModel(pl.LightningModule):
    def __init__(
        self,
        num_symbols: int,
        num_speakers: int,
        # audio
        resblock="2",
        resblock_kernel_sizes=(3, 5, 7),
        resblock_dilation_sizes=(
            (1, 2),
            (2, 6),
            (3, 12),
        ),
        upsample_rates=(8, 8, 4),
        upsample_initial_channel=256,
        upsample_kernel_sizes=(16, 16, 8),
        # mel
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        mel_channels: int = 80,
        sample_rate: int = 22050,
        sample_bytes: int = 2,
        channels: int = 1,
        mel_fmin: float = 0.0,
        mel_fmax: Optional[float] = None,
        # model
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        n_layers_q: int = 3,
        use_spectral_norm: bool = False,
        gin_channels: int = 0,
        use_sdp: bool = True,
        segment_size: int = 8192,
        # training
        dataset: Optional[List[Union[str, Path]]] = None,
        learning_rate: float = 2e-4,
        betas: Tuple[float, float] = (0.8, 0.99),
        eps: float = 1e-9,
        batch_size: int = 1,
        lr_decay: float = 0.999875,
        init_lr_ratio: float = 1.0,
        c_mel: int = 45,
        c_kl: float = 1.0,
        grad_clip: Optional[float] = None,
        num_workers: int = 8,
        seed: int = 1234,
        num_test_examples: int = 5,
        validation_split: float = 0.1,
        max_phoneme_ids: Optional[int] = None,
        warmup_epochs: int = 0,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        cosine_scheduler: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        if (self.hparams.num_speakers > 1) and (self.hparams.gin_channels <= 0):
            # Default gin_channels for multi-speaker model
            self.hparams.gin_channels = 512

        # Set up models
        self.model_g = SynthesizerTrn(
            n_vocab=self.hparams.num_symbols,
            spec_channels=self.hparams.filter_length // 2 + 1,
            segment_size=self.hparams.segment_size // self.hparams.hop_length,
            inter_channels=self.hparams.inter_channels,
            hidden_channels=self.hparams.hidden_channels,
            filter_channels=self.hparams.filter_channels,
            n_heads=self.hparams.n_heads,
            n_layers=self.hparams.n_layers,
            kernel_size=self.hparams.kernel_size,
            p_dropout=self.hparams.p_dropout,
            resblock=self.hparams.resblock,
            resblock_kernel_sizes=self.hparams.resblock_kernel_sizes,
            resblock_dilation_sizes=self.hparams.resblock_dilation_sizes,
            upsample_rates=self.hparams.upsample_rates,
            upsample_initial_channel=self.hparams.upsample_initial_channel,
            upsample_kernel_sizes=self.hparams.upsample_kernel_sizes,
            n_speakers=self.hparams.num_speakers,
            gin_channels=self.hparams.gin_channels,
            use_sdp=self.hparams.use_sdp,
        )
        self.model_d = MultiPeriodDiscriminator(
            use_spectral_norm=self.hparams.use_spectral_norm
        )

        # Dataset splits
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        self._load_datasets(validation_split, num_test_examples, max_phoneme_ids)

        # State kept between training optimizers
        self._y = None
        self._y_hat = None
        
        # For tracking resume data
        self._loaded_epoch = 0
        self._loaded_global_step = 0

    def _load_datasets(
        self,
        validation_split: float,
        num_test_examples: int,
        max_phoneme_ids: Optional[int] = None,
    ):
        if self.hparams.dataset is None:
            _LOGGER.debug("No dataset to load")
            return

        full_dataset = PiperDataset(
            self.hparams.dataset, max_phoneme_ids=max_phoneme_ids
        )
        valid_set_size = int(len(full_dataset) * validation_split)
        train_set_size = len(full_dataset) - valid_set_size - num_test_examples

        self._train_dataset, self._test_dataset, self._val_dataset = random_split(
            full_dataset, [train_set_size, num_test_examples, valid_set_size]
        )

    def forward(self, text, text_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio, *_ = self.model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
        )

        return audio

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=self.hparams.num_speakers > 1,
                segment_size=self.hparams.segment_size,
            ),
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=self.hparams.num_speakers > 1,
                segment_size=self.hparams.segment_size,
            ),
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=self.hparams.num_speakers > 1,
                segment_size=self.hparams.segment_size,
            ),
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
        )

    def training_step(self, batch: Batch, batch_idx: int, optimizer_idx: int):
        if optimizer_idx == 0:
            loss_g = self.training_step_g(batch)
            self.log("loss_gen_all", loss_g, prog_bar=True, on_step=True, sync_dist=True)
            return loss_g

        if optimizer_idx == 1:
            loss_d = self.training_step_d(batch)
            self.log("loss_disc_all", loss_d, prog_bar=True, on_step=True, sync_dist=True)
            return loss_d

    def training_step_g(self, batch: Batch):
        x, x_lengths, y, _, spec, spec_lengths, speaker_ids = (
            batch.phoneme_ids,
            batch.phoneme_lengths,
            batch.audios,
            batch.audio_lengths,
            batch.spectrograms,
            batch.spectrogram_lengths,
            batch.speaker_ids if batch.speaker_ids is not None else None,
        )
        (
            y_hat,
            l_length,
            _attn,
            ids_slice,
            _x_mask,
            z_mask,
            (_z, z_p, m_p, logs_p, _m_q, logs_q),
        ) = self.model_g(x, x_lengths, spec, spec_lengths, speaker_ids)
        self._y_hat = y_hat

        mel = spec_to_mel_torch(
            spec,
            self.hparams.filter_length,
            self.hparams.mel_channels,
            self.hparams.sample_rate,
            self.hparams.mel_fmin,
            self.hparams.mel_fmax,
        )
        y_mel = slice_segments(
            mel,
            ids_slice,
            self.hparams.segment_size // self.hparams.hop_length,
        )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            self.hparams.filter_length,
            self.hparams.mel_channels,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.mel_fmin,
            self.hparams.mel_fmax,
        )
        y = slice_segments(
            y,
            ids_slice * self.hparams.hop_length,
            self.hparams.segment_size,
        )  # slice

        # Save for training_step_d
        self._y = y

        _y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.model_d(y, y_hat)

        with autocast(self.device.type, enabled=False):
            # Generator loss
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hparams.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hparams.c_kl

            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

            # Log detailed losses for analysis
            self.log_dict({
                "g/loss_dur": loss_dur,
                "g/loss_mel": loss_mel,
                "g/loss_kl": loss_kl,
                "g/loss_fm": loss_fm,
                "g/loss_gen": loss_gen,
            }, prog_bar=False, on_step=True)

            return loss_gen_all

    def training_step_d(self, batch: Batch):
        # From training_step_g
        y = self._y
        y_hat = self._y_hat
        y_d_hat_r, y_d_hat_g, _, _ = self.model_d(y, y_hat.detach())

        with autocast(self.device.type, enabled=False):
            # Discriminator
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )
            loss_disc_all = loss_disc
            
            # Log discriminator detailed losses
            self.log_dict({
                "d/loss_disc": loss_disc,
                "d/loss_disc_r": torch.mean(torch.stack(losses_disc_r)),
                "d/loss_disc_g": torch.mean(torch.stack(losses_disc_g))
            }, prog_bar=False, on_step=True)

            return loss_disc_all

    def validation_step(self, batch: Batch, batch_idx: int):
        # Calculate validation losses
        val_loss_g = self.training_step_g(batch)
        val_loss_d = self.training_step_d(batch)
        val_loss = val_loss_g + val_loss_d
        
        # Log validation metrics
        self.log_dict({
            "val/loss_g": val_loss_g,
            "val/loss_d": val_loss_d,
            "val_loss": val_loss  # This is monitored by ModelCheckpoint
        }, prog_bar=True, sync_dist=True)

        # Generate audio examples only on the first batch and global rank 0
        if batch_idx == 0 and self.global_rank == 0:
            for utt_idx, test_utt in enumerate(self._test_dataset):
                if utt_idx >= 2:  # Limit to 2 examples
                    break
                    
                text = test_utt.phoneme_ids.unsqueeze(0).to(self.device)
                text_lengths = torch.LongTensor([len(test_utt.phoneme_ids)]).to(self.device)
                scales = [0.667, 1.0, 0.8]
                sid = (
                    test_utt.speaker_id.to(self.device)
                    if test_utt.speaker_id is not None
                    else None
                )
                test_audio = self(text, text_lengths, scales, sid=sid).detach()

                # Scale to make louder in [-1, 1]
                test_audio = test_audio * (1.0 / max(0.01, abs(test_audio.max())))

                tag = test_utt.text or str(utt_idx)
                
                # Log with correct shape for TensorBoard
                audio_tensor = test_audio
                if audio_tensor.ndim > 1:
                    if audio_tensor.shape[0] == 1:
                        audio_tensor = audio_tensor.squeeze(0)
                
                self.logger.experiment.add_audio(
                    f"val_audio/{tag}", 
                    audio_tensor,
                    global_step=self.global_step,
                    sample_rate=self.hparams.sample_rate
                )

        return val_loss

    def configure_optimizers(self):
        """
        Sets up optimizer and learning rate scheduler with state preservation for proper resuming.
        """
        import math
        from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ExponentialLR
        
        # Create optimizers for generator and discriminator
        optimizer_g = torch.optim.AdamW(
            self.model_g.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            eps=self.hparams.eps,
            weight_decay=self.hparams.weight_decay,
        )
        
        optimizer_d = torch.optim.AdamW(
            self.model_d.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            eps=self.hparams.eps,
            weight_decay=self.hparams.weight_decay,
        )
        
        # Setup the appropriate scheduler based on configuration
        if not self.hparams.cosine_scheduler:
            # Standard exponential decay scheduler
            scheduler_g = ExponentialLR(optimizer_g, gamma=self.hparams.lr_decay)
            scheduler_d = ExponentialLR(optimizer_d, gamma=self.hparams.lr_decay)
            
            scheduler_config_g = {
                "scheduler": scheduler_g,
                "interval": "step",
                "frequency": 1,
                "name": "generator_exp_decay",
                "monitor": "val_loss"
            }
            
            scheduler_config_d = {
                "scheduler": scheduler_d,
                "interval": "step",
                "frequency": 1,
                "name": "discriminator_exp_decay",
                "monitor": "val_loss"
            }
        else:
            # Calculate training steps for cosine scheduler
            steps_per_epoch = len(self.train_dataloader()) if self.train_dataloader() is not None else 100
            max_epochs = self.trainer.max_epochs if hasattr(self.trainer, 'max_epochs') else 1000
            num_training_steps = steps_per_epoch * max_epochs
            warmup_steps = int(self.hparams.warmup_ratio * num_training_steps)
            
            # Log scheduler setup information
            _LOGGER.info(f"Cosine scheduler: steps_per_epoch={steps_per_epoch}, "
                        f"max_epochs={max_epochs}, "
                        f"warmup_steps={warmup_steps}, "
                        f"total_steps={num_training_steps}")
            
            # Define lambda function for warmup + cosine decay
            def lr_lambda(current_step):
                # Account for resuming by adding the loaded global step
                if hasattr(self, '_loaded_global_step'):
                    current_step += self._loaded_global_step
                
                if current_step < warmup_steps:
                    # Linear warmup
                    return float(current_step) / float(max(1, warmup_steps))
                
                # Cosine decay after warmup
                progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return max(0.05, cosine_decay)  # Don't go below 5% of max lr
            
            scheduler_g = LambdaLR(optimizer_g, lr_lambda)
            scheduler_d = LambdaLR(optimizer_d, lr_lambda)
            
            scheduler_config_g = {
                "scheduler": scheduler_g,
                "interval": "step",
                "frequency": 1,
                "name": "generator_cosine_warmup",
                "monitor": "val_loss"
            }
            
            scheduler_config_d = {
                "scheduler": scheduler_d,
                "interval": "step", 
                "frequency": 1,
                "name": "discriminator_cosine_warmup",
                "monitor": "val_loss"
            }
        
        # Return optimizers and schedulers with complete config
        return (
            [optimizer_g, optimizer_d],
            [scheduler_config_g, scheduler_config_d]
        )
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Add custom metadata to checkpoint to ensure proper resuming.
        
        Args:
            checkpoint: The checkpoint dictionary being saved
        """
        # Add custom metadata to keep track of training progress
        checkpoint['vits_metadata'] = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'optimizer_states_saved': True,
            'scheduler_states_saved': True,
        }
        
        # Let Lightning handle saving full training state
        super().on_save_checkpoint(checkpoint)
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Load custom metadata from checkpoint for proper resuming.
        
        Args:
            checkpoint: The checkpoint dictionary being loaded
        """
        # Extract and store metadata for scheduler adjustment
        if 'vits_metadata' in checkpoint:
            metadata = checkpoint['vits_metadata']
            
            if 'global_step' in metadata:
                self._loaded_global_step = metadata['global_step']
                _LOGGER.info(f"Resuming from global step: {self._loaded_global_step}")
            
            if 'epoch' in metadata:
                self._loaded_epoch = metadata['epoch']
                _LOGGER.info(f"Resuming from epoch: {self._loaded_epoch}")
                
            _LOGGER.info(f"Checkpoint contains optimizer states: {metadata.get('optimizer_states_saved', False)}")
            _LOGGER.info(f"Checkpoint contains scheduler states: {metadata.get('scheduler_states_saved', False)}")
        
        # Let Lightning handle loading full training state
        super().on_load_checkpoint(checkpoint)
    
    def on_fit_start(self) -> None:
        """Log information about resuming or starting training"""
        if hasattr(self, '_loaded_global_step') and self._loaded_global_step > 0:
            _LOGGER.info(f"Training resumed from step {self._loaded_global_step}, epoch {self._loaded_epoch}")
            
            # Log current learning rates
            optimizers = self.optimizers()
            if isinstance(optimizers, list) and len(optimizers) >= 2:
                _LOGGER.info(f"Generator learning rate: {optimizers[0].param_groups[0]['lr']:.6f}")
                _LOGGER.info(f"Discriminator learning rate: {optimizers[1].param_groups[0]['lr']:.6f}")
        else:
            _LOGGER.info("Starting training from scratch")
            
        super().on_fit_start()
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VitsModel")
        parser.add_argument("--batch-size", type=int, required=True)
        parser.add_argument("--validation-split", type=float, default=0.1)
        parser.add_argument("--num-test-examples", type=int, default=5)
        parser.add_argument(
            "--max-phoneme-ids",
            type=int,
            help="Exclude utterances with phoneme id lists longer than this",
        )
        #
        parser.add_argument("--hidden-channels", type=int, default=192)
        parser.add_argument("--inter-channels", type=int, default=192)
        parser.add_argument("--filter-channels", type=int, default=768)
        parser.add_argument("--n-layers", type=int, default=6)
        parser.add_argument("--n-heads", type=int, default=2)
        #
        return parent_parser