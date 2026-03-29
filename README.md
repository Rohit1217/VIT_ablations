# ViT Ablation Study on CIFAR-10

Vision Transformer (ViT) and ResNet trained from scratch on CIFAR-10, with systematic ablation experiments. Built for a DLCV course assignment.

## Models
- **ViT** — patch embedding, learnable/sinusoidal/no positional encoding, CLS token or mean pooling
- **ResNet** — custom residual blocks with batch norm

## Experiments
| # | Variable | Values |
|---|---|---|
| 1 | Training data size | 5%, 10%, 25%, 50%, 100% |
| 2 | Patch size | 4, 8, 16 |
| 3 | Readout method | CLS token vs mean pool |
| 4 | Positional embedding | Learnable, Sinusoidal, None |
| 5 | Patch overlap | On / Off |
| 6 | Attention entropy analysis | — |
| 8 | CLS token position | Prepend vs Append |

## Setup
```bash
pip install torch torchvision tqdm matplotlib
```

## Usage
```bash
python experiment2.py
python experiment8.py
```
CIFAR-10 downloads automatically on first run.

## Results
Run the experiment scripts to regenerate all plots and logs under `results/`.
