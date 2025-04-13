"""Configuration classes"""
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class MelAudioConfig:
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    mel_channels: int = 80
    sample_rate: int = 22050
    sample_bytes: int = 2
    channels: int = 1
    mel_fmin: float = 0.0
    mel_fmax: Optional[float] = None


@dataclass
class ModelAudioConfig:
    resblock: str
    resblock_kernel_sizes: Tuple[int, ...]
    resblock_dilation_sizes: Tuple[Tuple[int, ...], ...]
    upsample_rates: Tuple[int, ...]
    upsample_initial_channel: int
    upsample_kernel_sizes: Tuple[int, ...]

    @staticmethod
    def low_quality() -> "ModelAudioConfig":
        return ModelAudioConfig(
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
        )

    @staticmethod
    def high_quality() -> "ModelAudioConfig":
        return ModelAudioConfig(
            resblock="1",
            resblock_kernel_sizes=(3, 7, 11),
            resblock_dilation_sizes=(
                (1, 3, 5),
                (1, 3, 5),
                (1, 3, 5),
            ),
            upsample_rates=(8, 8, 2, 2),
            upsample_initial_channel=512,
            upsample_kernel_sizes=(16, 16, 4, 4),
        )
        
    @staticmethod
    def optimized_quality(pruning_factor=0.5) -> "ModelAudioConfig":
        """
        Optimized model quality with customizable pruning factor.
        This handles models that have been created with the model optimizer.
        
        Args:
            pruning_factor: How much the model was reduced (default: 0.5)
        """
        # Calculate target dimensions based on pruning factor
        target_upsample = int(512 * (1 - pruning_factor))
        
        return ModelAudioConfig(
            resblock="1",  # Keep the better resblock type
            resblock_kernel_sizes=(3, 7, 11),  # Same kernels as high quality
            resblock_dilation_sizes=(
                (1, 3, 5),
                (1, 3, 5),
                (1, 3, 5),
            ),
            # Use 3-stage upsampling
            upsample_rates=(8, 8, 4),
            # Apply pruning factor to channel count
            upsample_initial_channel=target_upsample,
            # Match 3-stage upsampling structure
            upsample_kernel_sizes=(16, 16, 8),
        )


@dataclass
class ModelConfig:
    num_symbols: int
    n_speakers: int
    audio: ModelAudioConfig
    mel: MelAudioConfig = field(default_factory=MelAudioConfig)

    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    n_layers_q: int = 3
    use_spectral_norm: bool = False
    gin_channels: int = 0  # single speaker
    use_sdp: bool = True  # StochasticDurationPredictor
    segment_size: int = 8192

    @property
    def is_multispeaker(self) -> bool:
        return self.n_speakers > 1

    @property
    def resblock(self) -> str:
        return self.audio.resblock

    @property
    def resblock_kernel_sizes(self) -> Tuple[int, ...]:
        return self.audio.resblock_kernel_sizes

    @property
    def resblock_dilation_sizes(self) -> Tuple[Tuple[int, ...], ...]:
        return self.audio.resblock_dilation_sizes

    @property
    def upsample_rates(self) -> Tuple[int, ...]:
        return self.audio.upsample_rates

    @property
    def upsample_initial_channel(self) -> int:
        return self.audio.upsample_initial_channel

    @property
    def upsample_kernel_sizes(self) -> Tuple[int, ...]:
        return self.audio.upsample_kernel_sizes

    def __post_init__(self):
        if self.is_multispeaker and (self.gin_channels == 0):
            self.gin_channels = 512


@dataclass
class TrainingConfig:
    learning_rate: float = 2e-4
    betas: Tuple[float, float] = field(default=(0.8, 0.99))
    eps: float = 1e-9
    # batch_size: int = 32
    fp16_run: bool = False
    lr_decay: float = 0.999875
    init_lr_ratio: float = 1.0
    warmup_epochs: int = 0
    c_mel: int = 45
    c_kl: float = 1.0
    grad_clip: Optional[float] = None