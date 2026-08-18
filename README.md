# PenroseAim

PenroseAim is a flat research trainer for one model and one method: a Direct
Transformer trained with inness-aware optimal-transport flow matching. Batches
come directly from the sibling `PenroseSpur` checkout; no dataset is generated,
loaded, cached, or converted.

The model evolves 120 tile tokens with state
`(x, y, scaled_theta, vertex_in...)`. Tile color and MPEG7 class label are
conditioning inputs. Color-constrained LSA matches structured geometry noise to
data using `(x, y, theta)` only. Each identified center/vertex probe remains
binary during training and is independently revealed from null to data.

## Configuration

Configuration is merged in this order:

1. `default.yaml`
2. one optional `--config experiment.yaml`
3. dotted command-line overrides

The defaults use a dd128 Transformer, hex symmetry, `num_tiles=96` for
PenroseSpur canvas scaling, and `num_ret_tiles=120` actual model tokens. Inness
must arrive normalized to `[0,1]` from PenroseSpur.

```bash
cd /d/work/Diffusion/PenroseAim
PENROSE_SPUR_PATH=../PenroseSpur ~/.aivenv/bin/python -u train.py \
  wandb.enable=false train.num_epochs=1 train.steps_per_epoch=10
```

For hex training:

```bash
PENROSE_SPUR_PATH=../PenroseSpur ~/.aivenv/bin/python -u train.py \
  spur.symmetry=6
```

The default CUDA run is:

```bash
PENROSE_SPUR_PATH=../PenroseSpur ~/.aivenv/bin/python -u train.py
```

The discrete null is a one-inflated Bernoulli model. Fits are cached in
`inness_probs.csv` by symmetry, canvas tile count, and returned tile count.
Missing rows are fitted deterministically from five passes over all masks.

WandB uses project `penrose-aim`. Its run name and run ID both use the
epoch-free experiment identifier `aim_MMDD_HHmm_128x8_t120_l2`. Checkpoints
append `_eNNN.pt` to that same identifier.
WandB logs total, geometric, inness, and sampled lattice losses, mean
pre-clipping gradient norm, and the current learning rate once per epoch.

## Loss

`flow.loss` selects `l1` or `l2`. Geometry velocity error is weighted by target
soft inness and normalized by the total tile weight, preserving greater
importance for interior tiles. Clean identified probes are supervised uniformly
with binary cross-entropy over model logits:

```text
loss = weighted_geometry_error + flow.lambda_inness * inness_error
```

The default `flow.lambda_inness` is `0.25`.

## Resume

Each checkpoint stores model, optimizer, scheduler, epoch/global step, merged
configuration, resolved tile side length, run identity, WandB run ID, and
Python/NumPy/PyTorch/generator random states. Saving is atomic. Only the newest
epoch and lowest-loss epoch are retained, without `latest` or `best` labels.

Resume exactly from a checkpoint:

```bash
PENROSE_SPUR_PATH=../PenroseSpur ~/.aivenv/bin/python -u train.py \
  --resume outputs/aim_MMDD_HHmm_128x8_t120_l2/checkpoints/aim_MMDD_HHmm_128x8_t120_l2_e042.pt
```

Safe runtime settings can be overridden after `--resume`, including extending
the total epoch count:

```bash
PENROSE_SPUR_PATH=../PenroseSpur ~/.aivenv/bin/python -u train.py \
  --resume outputs/aim_MMDD_HHmm_128x8_t120_l2/checkpoints/aim_MMDD_HHmm_128x8_t120_l2_e042.pt \
  train.num_epochs=150
```

Use `--reset-optimizer` to keep the model, epoch, run identity, metrics, and RNG
state while starting with a fresh optimizer and learning-rate scheduler.

Architecture, symmetry, tile counts, loss, and flow settings are validated
against the checkpoint. Older four-channel logit-inness checkpoints are not
compatible with the identified-probe input and output projections.

## Reverse sampling

Reverse sampling integrates the predicted `(x, y, theta)` velocity over the
same sinusoidally warped path-time grid used for training. Binary vertex probes
start from the fitted null distribution, receive independent uniform reveal
times, and are sampled once from the model's clean Bernoulli logits when
revealed. Tile inness is the mean of its generated probes and is used as the
SVG opacity.

```bash
PENROSE_SPUR_PATH=../PenroseSpur ~/.aivenv/bin/python -u sampler.py \
  outputs/aim_.../checkpoints/aim_..._e100.pt \
  -a all -N 200 -o outputs/all_classes
```

Generate all trajectory SVGs, or only the first and last SVG for one class:

```bash
PENROSE_SPUR_PATH=../PenroseSpur ~/.aivenv/bin/python -u sampler.py \
  checkpoints/aim_..._e100.pt -a sample -c apple -N 200 -f all -o outputs/apple

PENROSE_SPUR_PATH=../PenroseSpur ~/.aivenv/bin/python -u sampler.py \
  checkpoints/aim_..._e100.pt -a sample -c apple -N 200 -f ends -o outputs/apple
```

Generate all trajectory SVGs and an MP4 for one class:

```bash
PENROSE_SPUR_PATH=../PenroseSpur ~/.aivenv/bin/python -u sampler.py \
  checkpoints/aim_..._e100.pt -a animate -c apple -N 200 -o outputs/apple
```

## Smoke test

The smoke test uses CPU, tiny model settings, both symmetries, one configured
L2 loss, an optimizer update, null-fit caching, retention, and a fresh-process
checkpoint resume:

```bash
cd /d/work/Diffusion/PenroseAim
PENROSE_SPUR_PATH=../PenroseSpur ~/.aivenv/bin/python tests/test_smoke.py
```

## Deliberate omissions

- No DDPM, DDIM, alternate flow, model registry, or second architecture.
- No XLA, JAX, TPU, Hydra, or configuration matrix.
- No `.npz` files or dataset generation/loading/caching.
- No cloud launcher or storage SDK. Mounted paths such as `/cloud/...` work as
  ordinary output paths; launch scripts belong in `at-cloud-scripts`.
