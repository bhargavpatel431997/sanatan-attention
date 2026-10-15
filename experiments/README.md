# Experiments

This directory contains the actual experimental code behind the paper, as distinct from
the quick demo notebook at the repository root. These are research scripts: they expect
serious hardware (TPU v4-8 or a 96 GB GPU), stream gated datasets from HuggingFace
(StarCoder requires accepting its license and an HF token), and run for hours to days.

## original_submission/

The code exactly as submitted to NeurIPS 2026 (supplementary material):

- `sanatan_training.py` - SANATAN vs standard softmax training (TPU, 184M, JAX/Flax)
- `sanatan_ablation_baselines.py` - six ablation variants plus Linear Transformer,
  Performer, and cosFormer baselines under the shared recipe
- `sanatan_4path_performer.py`, `sanatan_4path_extended.py` - four-path variants and
  the stabilized Performer baseline
- `generate_figures.py`, `generate_training_curve.py` - figure generation
- `results.json` - metrics as logged by these runs

Note: these scripts use a per-run validation sample drawn from the head of the data
streams. As documented in the paper (Appendix C, changelog), that protocol makes
cross-run numbers non-comparable and produced one retracted result. The numbers in
`results.json` are the superseded originals, kept for transparency.

## rebuttal/

The corrected evaluation produced during the NeurIPS review discussion. All numbers in
the paper come from here.

- `rebuttal_pipeline.ipynb` - the full correction pipeline: builds the common seeded
  held-out set (256 x 2048 tokens, seed 4321, drawn deep in each stream past all
  training consumption), re-evaluates every checkpoint on it, implements and verifies
  the exact p=1 recurrence, measures the p>1 train-inference gap, runs the p=1
  adaptation, measures gate statistics, and benchmarks memory/latency
- `gla_baseline.ipynb` - the matched-recipe Gated Linear Attention baseline
  (flash-linear-attention, 182.2M tied, identical tokenizer/data/optimizer/budget)
- `final_clean_results.json` - the held-out losses of record for every checkpoint;
  these are the numbers in the paper's tables

To rerun, set your own HuggingFace token where the notebooks mark `hf_YOUR_TOKEN_HERE`.
Model checkpoints (~360 MB each) are not distributed in this repository.
