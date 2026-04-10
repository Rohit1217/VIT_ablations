================================================================================
DLCV Assignment — ViT and ResNet on CIFAR-10
================================================================================

PYTHON VERSION
--------------
Python 3.10.20

PACKAGES
--------
torch       == 2.6.0   (requires CUDA 12.4 for GPU training)
torchvision == 0.21.0
tqdm        == 4.67.3
matplotlib  == 3.10.8
numpy       == 1.24.3

Install all dependencies:
    pip install -r requirements.txt

Note: For GPU support install the matching CUDA build of PyTorch from
      https://pytorch.org/get-started/locally/


FILE ORGANIZATION
-----------------
DLCV/
├── experiment1.py          # Effect of training data size on ViT vs ResNet accuracy
├── experiment2.py          # Effect of ViT patch size (4, 8, 16) on accuracy
├── experiment3.py          # CLS token vs mean pooling in ViT
├── experiment4.py          # Positional embedding type: Learned / Sinusoidal / None
├── experiment5.py          # Attention map visualization + entropy analysis on trained ViT
├── experiment6.py          # Overlapping vs non-overlapping patch projection
├── experiment7.py          # Layer-wise linear probing of ViT representations
├── experiment8.py          # CLS token position: prepend vs append
│
├── config/
│   └── config.py           # Shared hyperparameters for all experiments
│
├── models/
│   ├── __init__.py         # Exports core building blocks from models.py
│   ├── models.py           # CNN_block, FFN, Self_Attention, Transformer_block,
│   │                         Position_embedding (sinusoidal)
│   ├── vit.py              # Vision Transformer (Patchify + VIT class)
│   ├── resnet.py           # Custom ResNet (Resnet_block, Resnet_big_block, Resnet)
│   ├── data.py             # CachedDataset + load_dataloaders for CIFAR-10
│   ├── vit_profiled.py     # Profiling script with CUDA kernel timing metrics
│   └── log.text            # Profiling log — CUDA timing output from vit_profiled.py
│
├── results/
│   ├── experiment1/        # Accuracy vs data fraction plots + results.txt
│   ├── experiment2/        # Loss curves per patch size + results.txt
│   ├── experiment3/        # Loss curves: CLS token vs mean pool + results.txt
│   ├── experiment4/        # Loss curves: positional embedding types + results.txt
│   ├── experiment5/        # Attention map overlays (per-image + grid) + entropy tensors + results.txt
│   ├── experiment6/        # Loss curves: patch overlap ablation + results.txt
│   ├── experiment7/        # Linear probe accuracy/loss per layer + results.txt
│   └── experiment8/        # CLS position comparison + attention heatmaps + results.txt
│
├── cifar10_dataset/        # CIFAR-10 data (auto-downloaded on first run)
├── best_VIT.pt             # Best ViT checkpoint (used by experiments 5, 7)
├── best_RESNET.pt          # Best ResNet checkpoint
├── requirements.txt
└── readme.txt


MODEL SIZES
-----------
ViT    — 4.77M parameters
ResNet — 4.63M parameters


RUNNING THE EXPERIMENTS
-----------------------
Each experiment is self-contained and can be run from the project root:

    python experiment1.py   # ~hours; trains ViT + ResNet at 5 data fractions
    python experiment2.py   # trains ViT at 3 patch sizes x 5 trials
    python experiment3.py   # trains ViT with CLS token and mean pooling
    python experiment4.py   # trains ViT with 3 positional embedding types
    python experiment5.py   # loads best_VIT.pt, saves attention map visualizations + entropy
    python experiment6.py   # trains ViT with/without patch overlap
    python experiment7.py   # loads best_VIT.pt, trains linear probes per layer
    python experiment8.py   # trains ViT with CLS prepend vs append

Experiments 5 and 7 require a trained checkpoint at best_VIT.pt in the
project root. Run experiment1.py (or any other experiment) first and save
the best model, or use the provided checkpoint.

CIFAR-10 is downloaded automatically to cifar10_dataset/ on the first run.

HARDWARE
--------
Experiments are configured to run on CUDA GPUs. The default device is set
in config/config.py (DEVICE = "cuda:3"). Adjust this and the per-experiment
Device variable to match your setup before running.

Training uses bfloat16 autocast and torch.compile() — requires PyTorch 2.0+
and a CUDA-capable GPU (Ampere or newer recommended for bfloat16 support).
================================================================================
