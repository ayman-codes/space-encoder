# SSIM Boundary Policy — Findings

Generated from `results.json` (same directory). Reproduce with:

```bash
uv run python scripts/benchmark_ssim.py
```

## Question

An 11×11 SSIM window cannot be centered on the outer five pixels of a 64×64
patch. Three defensible policies exist. Does the choice materially change what
the reconstruction loss measures, and therefore what it optimizes?

## Method

256 deterministic test-split patches from `eurosat-rgb-v1.0.0`
(`f7ccc0bc…f9754570`). Each policy scores the same seven perturbations against
the clean target: two local noise fields (border ring vs interior), a constant
border, a global blur, a three-pixel shift, and posterization. Drops are paired
per patch and reported with 2,000-resample bootstrap 95% intervals. The
gradient analysis differentiates the scalar loss with respect to the input and
splits absolute mass between the five-pixel ring and the interior.

## Numbers

| Policy | Map | Border-noise drop | Interior-noise drop | Area-normalized border ratio | Border gradient share |
| :--- | :---: | ---: | ---: | ---: | ---: |
| `valid` | 54×54 | 0.0792 | 0.7117 | 0.275 | 0.635 |
| `zero` | 64×64 | 0.1629 | 0.5483 | 0.734 | 0.798 |
| `masked` | 64×64 | 0.2612 | 0.5671 | 1.138 | 0.803 |

95% intervals on the border drop: `valid` [0.0757, 0.0831], `zero` [0.1566,
0.1697], `masked` [0.2536, 0.2691]. The intervals do not overlap.

## Interpretation

1. **The boundary policy is not cosmetic.** Corrupting the same border ring
   produces a 3.3× larger loss increase under `masked` than under `valid`, and a
   1.6× larger increase than under `zero`. Since every other part of the SSIM
   computation is identical, the difference is entirely attributable to how the
   missing ring is treated.
2. **Zero-padding under-penalizes rather than distorts.** The padded zeros are
   shared by prediction and target, so they enter the local means and variances
   symmetrically. The effect is dilution, not a spurious target: border errors
   are measured over a window that is partly identical constant, so they count
   for less. `zero` retains 62% of the `masked` sensitivity.
3. **Cropping removes border centers, not border influence.** The original
   hypothesis — that `valid` yields exactly zero border gradient — is
   **falsified**. The ring still contributes 63.5% of absolute gradient mass,
   because border pixels participate in the windows of nearby centers even
   though they cannot be centers themselves. What cropping removes is the direct
   evaluation of the ring; sensitivity falls to 30% of `masked`.
4. **Masked is the only policy near area parity.** The border occupies 28.8% of
   the patch. Normalizing each drop by its perturbed area gives 0.275 (`valid`),
   0.734 (`zero`), and 1.138 (`masked`). The slight overshoot above 1.0 is
   expected: border windows average fewer real samples, so their local variance
   estimate is noisier and a fixed-amplitude perturbation degrades them
   marginally more per unit area.

## Verdict

Use `masked` for the loss. Keep `valid` for reporting because it is the
canonical definition and the only number comparable to the literature — but
label it as cropped, and never compare it against the masked training
objective. `zero` has no defensible role beyond backwards compatibility.

## Limitations

- Static perturbations on clean patches, not end-to-end training. This measures
  objective sensitivity, not macro-F1. The causal check is the deferred
  three-arm training ablation.
- The area-normalized ratio assumes a roughly linear response; SSIM is
  nonlinear and saturates near 1, so parity is approximate.
- One dataset, one noise amplitude, one window size. The direction of the effect
  should be stable, the exact factors should not be quoted as universal.
- `interior_noise` drops are not comparable across policies because the map
  support differs (54×54 vs 64×64); only the paired border comparison is used
  for cross-policy claims.
