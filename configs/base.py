"""
Shared configuration for the Mixed-v2 experiments.

Every run — U-Net, UNet++, SegFormer and SegFormer with an ImageNet-pretrained
encoder — trains on the same 3-way mixed dataset with the same loss, augmentation
and validation protocol; only the model, its initialization and its optimizer
settings differ. Those shared settings live here, the per-run overrides live in
configs/unet.py, configs/unetpp.py, configs/segformer.py and
configs/segformer_pretrained.py.

Data paths are relative to a DATA_ROOT that is resolved at runtime (see
`MixedV2Config.resolve`), so nothing in this repository contains an absolute path.
"""

import os
from pathlib import Path


class MixedV2Config:
    """Base configuration — the Mixed-v2 3-way training recipe."""

    # ── Identity ──────────────────────────────────────────────────────────────
    MODEL = None                # set by the per-model subclasses
    RUN_NAME = 'mixedv2'

    # ── Data layout under DATA_ROOT ───────────────────────────────────────────
    # real/                  ClipLog-normalized hand-labelled sample directories,
    #                        each holding one raw *.tif plus labkit_segmented.tif
    # synth_train.h5         old synthetic frames  (keys: images, masks)
    # synth_val.h5           old synthetic frames  (keys: images, masks)
    # filtered_indices.json  {"train": [...], "val": [...], "test": [...]}
    # ring_train.h5          ring synthetic patches (keys: images, masks)
    # ring_val.h5            ring synthetic patches (keys: images, masks)
    REAL_SUBDIR         = 'real'
    SYNTH_TRAIN_NAME    = 'synth_train.h5'
    SYNTH_VAL_NAME      = 'synth_val.h5'
    SYNTH_INDICES_NAME  = 'filtered_indices.json'
    RING_TRAIN_NAME     = 'ring_train.h5'
    RING_VAL_NAME       = 'ring_val.h5'

    # Filled in by resolve()
    DATA_ROOT      = None
    REAL_DATA_DIR  = None
    SYNTH_TRAIN_H5 = None
    SYNTH_VAL_H5   = None
    SYNTH_INDICES  = None
    RING_TRAIN_H5  = None
    RING_VAL_H5    = None
    OUTPUT_DIR     = None
    CHECKPOINT_DIR = None
    LOG_DIR        = None

    # ── Patch extraction (real hand-labelled data) ────────────────────────────
    PATCH_SIZE      = 512
    STRIDE          = 384
    MIN_FG_FRACTION = 0.005

    # ── Model ─────────────────────────────────────────────────────────────────
    IN_CHANNELS = 1
    NUM_CLASSES = 3             # 0 = background, 1 = HEDM spot, 2 = powder ring
    FEATURES    = 64
    BILINEAR    = True

    # SegFormer only: HuggingFace backbone id, whether to load its ImageNet
    # weights (False takes the architecture from the hub config but leaves every
    # weight random), and how to collapse the pretrained RGB stem to 1 channel.
    BACKBONE            = None
    PRETRAINED_BACKBONE = False
    GRAYSCALE_INIT      = 'sum'

    # ── 3-way mixing ratios (must sum to 1.0) ─────────────────────────────────
    REAL_RATIO      = 0.20      # real hand-labelled patches
    OLD_SYNTH_RATIO = 0.20      # old synthetic frames (diverse materials, HEDM-rich)
    RING_RATIO      = 0.60      # new ring synthetic patches (powder-rich)

    # ── Combined validation metric weights (best-model selection + LR schedule) ─
    REAL_METRIC_WEIGHT      = 0.20
    OLD_SYNTH_METRIC_WEIGHT = 0.20
    RING_METRIC_WEIGHT      = 0.60

    # ── Optimizer (per-model subclasses override) ─────────────────────────────
    OPTIMIZER      = 'adam'     # 'adam' or 'adamw'
    LEARNING_RATE  = 1e-4
    WEIGHT_DECAY   = 1e-5
    GRAD_CLIP_NORM = None       # None disables gradient clipping
    WARMUP_EPOCHS  = 0          # linear LR warmup; scheduler steps only afterwards

    # ── Training ──────────────────────────────────────────────────────────────
    BATCH_SIZE = 4              # per GPU
    NUM_EPOCHS = 150

    # ── LR scheduler — ReduceLROnPlateau on the combined validation mIoU ───────
    SCHEDULER_PATIENCE = 10
    SCHEDULER_FACTOR   = 0.5
    MIN_LR             = 5e-7

    # ── Loss — Combined Focal + Dice ──────────────────────────────────────────
    # alpha[1] = 5.0 compensates for the ~0.4% HEDM pixel fraction.
    FOCAL_ALPHA  = [0.1, 5.0, 1.0]
    FOCAL_GAMMA  = 2.0
    FOCAL_WEIGHT = 0.5
    DICE_WEIGHT  = 0.5

    # ── Deep supervision (UNet++ only) ────────────────────────────────────────
    USE_DEEP_SUPERVISION     = False
    DEEP_SUPERVISION_WEIGHTS = None

    # ── Checkpointing / early stopping ────────────────────────────────────────
    EARLY_STOPPING_PATIENCE = 25
    SAVE_EVERY_N_EPOCHS     = 10

    # ── Augmentation (training samples only) ──────────────────────────────────
    FLIP_PROB        = 0.5
    ROTATE_PROB      = 0.5
    BRIGHTNESS_RANGE = (0.9, 1.1)
    CONTRAST_RANGE   = (0.9, 1.1)

    # ── Warm start ────────────────────────────────────────────────────────────
    PRETRAINED = None           # path to a checkpoint to initialize weights from

    # ── Misc ──────────────────────────────────────────────────────────────────
    USE_AMP     = True
    NUM_WORKERS = 6
    SEED        = 42
    DDP_PORT    = '12356'

    # ── Path resolution ───────────────────────────────────────────────────────

    @classmethod
    def resolve(cls, data_root=None, output_dir=None, check=True):
        """
        Bind the configuration to a dataset directory and an output directory.

        data_root precedence: explicit argument > $POWDERSEG_DATA_ROOT > ./data
        output_dir default:   ./runs/<MODEL>_<RUN_NAME>
        """
        if data_root is None:
            data_root = os.environ.get('POWDERSEG_DATA_ROOT', 'data')
        root = Path(data_root).expanduser().resolve()

        cls.DATA_ROOT      = root
        cls.REAL_DATA_DIR  = root / cls.REAL_SUBDIR
        cls.SYNTH_TRAIN_H5 = root / cls.SYNTH_TRAIN_NAME
        cls.SYNTH_VAL_H5   = root / cls.SYNTH_VAL_NAME
        cls.SYNTH_INDICES  = root / cls.SYNTH_INDICES_NAME
        cls.RING_TRAIN_H5  = root / cls.RING_TRAIN_NAME
        cls.RING_VAL_H5    = root / cls.RING_VAL_NAME

        if output_dir is None:
            output_dir = Path('runs') / f'{cls.MODEL}_{cls.RUN_NAME}'
        cls.OUTPUT_DIR     = Path(output_dir).expanduser().resolve()
        cls.CHECKPOINT_DIR = cls.OUTPUT_DIR / 'checkpoints'
        cls.LOG_DIR        = cls.OUTPUT_DIR / 'logs'

        if check:
            cls.check_data()
        return cls

    @classmethod
    def check_data(cls):
        """Raise a single, readable error listing everything that is missing."""
        required = [
            ('real sample directory', cls.REAL_DATA_DIR),
            ('old synthetic train',   cls.SYNTH_TRAIN_H5),
            ('old synthetic val',     cls.SYNTH_VAL_H5),
            ('synthetic index file',  cls.SYNTH_INDICES),
            ('ring synthetic train',  cls.RING_TRAIN_H5),
            ('ring synthetic val',    cls.RING_VAL_H5),
        ]
        missing = [f'  {label:<24} {path}' for label, path in required
                   if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete dataset under DATA_ROOT={cls.DATA_ROOT}\n"
                "Missing:\n" + "\n".join(missing) +
                "\n\nSee the 'Data' section of README.md for the expected layout, "
                "or point --data-root at the bundled examples/ subset."
            )

    # ── Reporting ─────────────────────────────────────────────────────────────

    @classmethod
    def fields(cls):
        """All configuration fields as a plain dict (upper-case attributes)."""
        out = {}
        for klass in reversed(cls.__mro__):
            for key, value in vars(klass).items():
                if key.isupper():
                    out[key] = value
        return out

    @classmethod
    def describe(cls, stream=None):
        """Print the fully resolved configuration."""
        lines = [
            "=" * 72,
            f"Configuration: {cls.__name__}  (model={cls.MODEL})",
            "=" * 72,
        ]
        for key, value in sorted(cls.fields().items()):
            lines.append(f"  {key:<26} {value}")
        lines.append("=" * 72)
        text = "\n".join(lines)
        if stream is None:
            print(text)
        else:
            stream.write(text + "\n")
        return text
