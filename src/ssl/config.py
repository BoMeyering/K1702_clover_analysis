"""
src.ssl.config.py
Structured configuration for FlexMatch / supervised training on the HQ Labelbox masks.
Adapted from pgcview_v2_training_framework/src/utils/config.py.
BoMeyering 2025
"""

import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from omegaconf import OmegaConf


class ModelArchitecture(Enum):
    """Valid segmentation architectures from segmentation_models_pytorch."""
    DEEPLABV3 = 'DeepLabV3'
    DEEPLABV3PLUS = 'DeepLabV3Plus'
    FPN = 'FPN'
    LINKNET = 'Linknet'
    MANET = 'MAnet'
    PAN = 'PAN'
    PSPNET = 'PSPNet'
    SEGFORMER = 'Segformer'
    UPERNET = 'UPerNet'
    UNET = 'Unet'
    UNETPLUSPLUS = 'UnetPlusPlus'


class LossCriterion(Enum):
    """Valid loss functions implemented in src.ssl.losses."""
    CELOSS = 'CELoss'
    FOCALLOSS = 'FocalLoss'
    CBLOSS = 'CBLoss'
    ACBLOSS = 'ACBLoss'
    RECALLLOSS = 'RecallLoss'
    DICELOSS = 'DiceLoss'
    TVERSKYLOSS = 'TverskyLoss'
    TVMFDICELOSS = 'TvmfDiceLoss'


@dataclass
class Images:
    resize: List[int] = field(default_factory=lambda: [1024, 1024])   # model input (H, W)
    decode_reduction: int = 2   # JPEGs are decoded at 1/n resolution (1, 2, 4 or 8); 5184x3456 -> 2592x1728 at 2


@dataclass
class ProxyEval:
    """Agreement with the old SAM masks on images that have no HQ label (predictions are collapsed to SAM classes)."""
    enabled: bool = True
    target_dir: str = 'data/processed/targets'
    split: str = 'val'                     # data_split.csv split to evaluate on
    every_n_epochs: int = 5
    max_images: Optional[int] = None       # cap for quick runs
    sam_classes: List[str] = field(default_factory=lambda: ['soil', 'quadrat', 'clover'])
    # HQ class name -> SAM class index
    class_map: Dict[str, int] = field(default_factory=lambda: {
        'soil': 0, 'quadrat': 1, 'clover_leaves': 2, 'clover_stems': 2, 'other_veg': 0, 'person': 0,
    })


@dataclass
class Data:
    image_dir: str = 'data/processed/images'
    hq_target_dir: str = 'data/processed_hq/targets'
    split_csv: str = 'data/data_split.csv'
    holdout_ids: List[str] = field(default_factory=list)   # HQ images held out for validation (never trained on)
    exclude_ids: List[str] = field(default_factory=list)   # images to drop from every pool
    unlabeled_exclude_proxy: bool = False                  # True -> keep proxy-eval images out of the unlabeled pool
    sam_diagnostics: bool = True                           # load SAM masks for unlabeled images to score pseudo-labels
    proxy: ProxyEval = field(default_factory=ProxyEval)


@dataclass
class Directories:
    checkpoint_dir: str = 'checkpoints/ssl'
    log_dir: str = 'logs'
    starting_checkpoint_path: Optional[str] = None


@dataclass
class Training:
    epochs: int = 100
    steps_per_epoch: Optional[int] = None   # default: one pass over the unlabeled pool per rank
    amp: bool = True                        # bf16 autocast
    seed: int = 42
    l_num_workers: int = 4
    u_num_workers: int = 10
    val_num_workers: int = 6
    find_unused_parameters: bool = False


@dataclass
class Loss:
    name: LossCriterion = LossCriterion.CELOSS
    class_sample_count_path: Optional[str] = None
    samples: Optional[List[float]] = field(default_factory=list)
    weights: Optional[List[float]] = field(default_factory=list)
    reduction: str = 'mean'
    loss_type: str = 'CELOSS'
    gamma: float = 1.0
    smooth: float = 0.0
    alpha: float = 0.5
    beta: float = 0.5
    kappa: float = 0.0
    lambda_k: Optional[float] = None
    exclude_empty_target: bool = True


@dataclass
class FlexMatch:
    enabled: bool = True          # False -> supervised-only baseline on the same labeled set and schedule
    tau: float = 0.95             # fixed upper threshold
    tau_min: float = 0.0          # floor on the class thresholds
    mapping: str = 'convex'       # convex | linear | concave
    warmup: bool = True           # FlexMatch threshold warmup (counts unused pixels)
    status: str = 'ema'           # ema: running learning status | batch: current batch only (pgcview behaviour)
    normalize: str = 'ratio'      # ratio: confident / predicted pixels per class | count: raw confident counts (paper)
    momentum: float = 0.99        # EMA momentum for status='ema'
    lam: float = 1.0              # unlabeled loss weight
    scale_by_conf_fraction: bool = True   # multiply the unlabeled loss by the confident-pixel fraction (pgcview behaviour)
    unsup_warmup_epochs: int = 0  # supervised-only epochs before the unlabeled loss starts
    cutmix_p: float = 0.5         # per-sample probability of CutMix on the strong view
    l_batch_size: int = 2
    u_batch_size: int = 4


@dataclass
class SMPModelConfig:
    encoder_name: Optional[str] = 'mit_b1'
    encoder_depth: Optional[int] = 5
    encoder_weights: Optional[str] = 'imagenet'
    in_channels: Optional[int] = 3
    classes: Optional[int] = None   # set from the target mapping when left empty


@dataclass
class Model:
    architecture: ModelArchitecture = ModelArchitecture.SEGFORMER
    config: SMPModelConfig = field(default_factory=SMPModelConfig)


@dataclass
class OptimizerParams:
    lr: float = 0.001
    momentum: float = 0.9
    nesterov: bool = True
    dampening: float = 0
    alpha: float = 0.99
    gamma: float = 0.99
    etas: Tuple[float, float] = field(default_factory=lambda: (0.5, 1.2))
    betas: Tuple[float, float] = field(default_factory=lambda: (0.9, 0.999))
    rho: float = 0.9
    amsgrad: bool = False
    foreach: Optional[bool] = None


@dataclass
class Optimizer:
    name: str = 'AdamW'
    weight_decay: float = 0.001
    original_weight_decay: Optional[float] = None
    filter_bias_and_bn: bool = True
    ema: bool = True
    ema_decay: float = 0.99
    optimizer_params: OptimizerParams = field(default_factory=OptimizerParams)


@dataclass
class Scheduler:
    name: str = 'ExponentialLR'
    gamma: float = 0.99
    step_size: float = 0.00001
    T_max: int = 5
    eta_min: float = 0.0
    T_0: Optional[int] = None
    T_mult: int = 1
    last_epoch: int = -1


@dataclass
class Norm:
    means: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class Metadata:
    norm: Norm = field(default_factory=Norm)
    target_mapping_path: str = 'metadata/hq_target_mapping.json'


@dataclass
class Checkpoint:
    top_k: int = 5
    w_loss: float = 0.5
    w_miou: float = 1.0
    w_gds: float = 1.0


@dataclass
class Wandb:
    enabled: bool = True
    mode: str = 'online'   # online | offline | disabled
    entity: str = 'oxbowsolutions'
    project: str = 'k1702_clover_ssl'
    vis_alpha: float = 0.4
    num_vis_images: int = 4


@dataclass
class TrainSSLConfig:
    model_run: str = 'flexmatch_hq10'
    device: str = 'cuda'
    rank: Optional[int] = None
    local_rank: Optional[int] = None
    world_size: Optional[int] = None
    is_main: Optional[bool] = None
    logging_level: str = 'INFO'
    images: Images = field(default_factory=Images)
    data: Data = field(default_factory=Data)
    metadata: Metadata = field(default_factory=Metadata)
    directories: Directories = field(default_factory=Directories)
    training: Training = field(default_factory=Training)
    model: Model = field(default_factory=Model)
    optimizer: Optimizer = field(default_factory=Optimizer)
    scheduler: Scheduler = field(default_factory=Scheduler)
    loss: Loss = field(default_factory=Loss)
    flexmatch: FlexMatch = field(default_factory=FlexMatch)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    wandb: Wandb = field(default_factory=Wandb)


def set_run_name(conf: OmegaConf):
    """Append a timestamp to conf.model_run."""
    stamp = datetime.datetime.now().isoformat(timespec='seconds', sep='_').replace(':', '.')
    conf.model_run = f'{conf.model_run}_{stamp}'
