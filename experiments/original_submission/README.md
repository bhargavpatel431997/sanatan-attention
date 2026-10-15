# SANATAN: Supplementary Material

## Contents

### Code
| File | Description | Produces |
|------|-------------|----------|
| `sanatan_training.py` | Main SANATAN (6-path) + softmax baseline | Table 2: softmax (3.27), SANATAN 6p (4.85) |
| `sanatan_ablation_baselines.py` | Ablation study + baselines + 10B extended | Tables 2-3, Figure 2 |
| `sanatan_4path_performer.py` | 4-path variants + stabilized Performer | Table 2: Performer (6.11), Table 3: 4-path |
| `sanatan_4path_extended.py` | 4-path power^4 at 10B tokens | Future work |
| `generate_figures.py` | Baseline + ablation bar charts | Figures 3 |
| `generate_training_curve.py` | Training trajectory figure | Figure 2 |

### Data
| File | Description |
|------|-------------|
| `results.json` | Complete training metrics for all experiments (val/train curves) |

### Figures
| File | Description |
|------|-------------|
| `fig_torus.pdf` | Figure 1: Root-partitioned torus geometry |
| `fig_training_curve.pdf` | Figure 2: Validation loss over 10B tokens |
| `fig_baselines.pdf` | Figure 3: Baseline comparison |
| `fig_ablation.pdf` | Ablation bar chart (supplementary) |

## Requirements

```
jax[tpu]          (TPU support)
flax              (neural network library)
optax             (optimizer)
datasets          (HuggingFace)
transformers      (tokenizers)
tqdm
matplotlib        (figures only)
```

## Reproducing Results

```bash
# Main results (Table 2): ~12h on TPU v4-8
python sanatan_training.py

# All ablations + baselines + 10B (Tables 2-3, Figure 2): ~110h on TPU v4-8
python sanatan_ablation_baselines.py

# 4-path + Performer (remaining Table 2-3 entries): ~18h on TPU v4-8
python sanatan_4path_performer.py
```

All scripts support checkpoint resumption via GCS or local disk.

## Architecture (Table 1)

- 184M parameters (768 dim, 12 heads, 64 head dim, 12 layers)
- SwiGLU FFN (2048 hidden), RMSNorm (pre-norm)
- AdamW (β1=0.9, β2=0.95, wd=0.1), LR 6e-4 with 500-step warmup
- 2048 context length, bfloat16 weights, float32 attention scores
- TPU v4-8 (4 chips)

## Data Mixture

| Source | Weight |
|--------|--------|
| FineWeb | 52% |
| StarCoder | 28% |
| CulturaX-zh | 11% |
| FineWeb-Edu | 6% |
| OpenWebMath | 1.5% |
| FineMath | 1.5% |

All datasets are publicly available.
