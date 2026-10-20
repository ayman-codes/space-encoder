# Findings

Internal research notes for the EOS Autoencoder Engine. Each entry records a
question, the experimental setup, the measured result, and the decision it
produced. Artifacts are reproducible from committed scripts.

| ID | Finding | Status |
|---|---|---|
| F-001 | SSIM boundary policy changes border sensitivity by up to 3.3× | Decided |
| F-002 | Evaluation precision, not just training nondeterminism, sets a ~0.005 macro-F1 comparability floor | Decided |
| F-003 | The hybrid's class-level gains localize to the cropland complex, not the river/highway pair | Provisional |

---

# F-001 — The SSIM boundary policy, not the window size, controls whether border reconstruction errors are penalized

**Date:** 2026-10-18
**Artifacts:** `reports/ssim-boundary/{results.json,summary.md,findings.md}`, three figures
**Code:** `src/eos/model/losses.py`, `src/eos/experiments/ssim_boundary.py`, `scripts/benchmark_ssim.py`
**Reproduce:** `uv run python scripts/benchmark_ssim.py`

## 1. Why this question exists

The reconstruction objective is

$$
\mathcal{L}_{\text{recon}} = \alpha\,\mathrm{MSE}(X,\hat X) + (1-\alpha)\,(1-\mathrm{SSIM}(X,\hat X)),
\qquad \alpha = 0.85,
$$

and it enters the total loss with weight $\lambda \in [0.1, 0.5]$. The structural
term is the mechanism behind the project hypothesis: it is supposed to keep
high-frequency boundary topology — river banks, highway corridors, parcel
edges — in the bottleneck. If that term is blind to the patch border, the
hypothesis is weakest exactly where 64×64 crops often place a parcel edge.

SSIM is local. At every center $p$ it compares two windows of size $k \times k$
(default $11 \times 11$, Gaussian $\sigma = 1.5$):

$$
\mu_x(p) = \sum_{q} w(q)\,x(p+q), \qquad
\sigma_x^2(p) = \sum_{q} w(q)\,\bigl(x(p+q)-\mu_x(p)\bigr)^2,
$$

$$
\mathrm{SSIM}(p) =
\frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}
     {(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)},
\qquad C_1 = 0.01^2,\; C_2 = 0.03^2 .
$$

For a center within five pixels of the edge, part of the window lies outside the
image. The definition does not say what is there. That is the entire problem:
the window does not fit, not the image.

At 64×64 the five-pixel ring is **28.8% of the patch area**
($1 - 54^2/64^2 = 0.2881$), so the choice is not a rounding detail.

## 2. The three policies

| Policy | Missing pixels | Resulting map | Consequence |
|---|---|---|---|
| `valid` | none — only centers with full support are evaluated | 54×54 | border pixels never serve as centers; direct evaluation of the ring is absent |
| `zero` | replaced by $0$ in both prediction and target | 64×64 | synthetic constant is shared by both tensors, so it cancels in part but dilutes local statistics |
| `masked` | excluded from the weighted sums and the weights renormalized | 64×64 | every center evaluated from real pixels only |

`masked` uses the support mass

$$
m(p) = \sum_{q \in \Omega(p)} w(q) \le 1,
\qquad
\mu_x(p) = \frac{1}{m(p)}\sum_{q\in\Omega(p)} w(q)\,x(p+q),
$$

and the same normalization for the second moments. At a corner, $m \approx 0.25$
and only 36 of the 121 window positions exist; the statistic is noisier but
unbiased.

## 3. Setup

**Data.** 256 patches sampled deterministically from the EuroSAT test split
(`eurosat-rgb-v1.0.0`, manifest `f7ccc0bc…f9754570`). Selection:
`numpy.random.default_rng(42).choice(test_indices, 256, replace=False)`, sorted.
Inputs are scaled to $[0,1]$ exactly as the model sees them.

**Perturbations.** Applied to the prediction only; the clean patch is the
target. Each probe isolates a spatial region or a degradation type:

| Name | Operation | Purpose |
|---|---|---|
| `identity` | none | sanity: every policy must return SSIM = 1 |
| `border_noise` | $\mathcal{N}(0, 0.1^2)$ on the 5-px ring | does the loss see border error? |
| `interior_noise` | same noise on the inner 54×54 | reference sensitivity |
| `border_constant` | ring set to 0.5 | worst-case border corruption |
| `blur` | 3×3 binomial kernel, replicate padding | global structural degradation |
| `shift` | circular roll by 3 px on both axes | misalignment |
| `quantize` | $\mathrm{round}(3x)/3$ | photometric degradation |

**Metrics.** Per patch, the SSIM map is averaged over channels and positions.
The drop is the paired difference against `identity` (which is exactly 1 for all
policies). Intervals are 95% percentile bootstrap over patches, 2,000 resamples,
seed 42. The area-normalized ratio divides the border drop by 0.2881 and the
interior drop by 0.7119 before taking their ratio.

**Gradient analysis.** For the `border_noise` batch, compute
$\partial (1-\overline{\mathrm{SSIM}})/\partial \hat X$ and split the absolute
mass between the ring and the interior. This measures where the objective
actually pushes, independently of the loss value.

**Environment.** Python 3.12.13, torch 2.14.1+cpu. Runtime ≈ 7 s.

## 4. Results

### 4.1 Headline

| Policy | Map | Border drop | 95% CI | Interior drop | Area-normalized ratio | Border gradient share |
| :--- | :---: | ---: | :--- | ---: | ---: | ---: |
| `valid` | 54×54 | 0.0792 | [0.0757, 0.0831] | 0.7117 | 0.275 | 0.635 |
| `zero` | 64×64 | 0.1629 | [0.1566, 0.1697] | 0.5483 | 0.734 | 0.798 |
| `masked` | 64×64 | 0.2612 | [0.2536, 0.2691] | 0.5671 | 1.138 | 0.803 |

The intervals do not overlap. For identical corruption:

- `masked` reacts **3.30×** more than `valid`;
- `masked` reacts **1.60×** more than `zero`;
- `zero` reacts **2.06×** more than `valid`.

### 4.2 Global perturbations are policy-invariant (control)

| Perturbation | `valid` | `zero` | `masked` | Spread |
| :--- | ---: | ---: | ---: | ---: |
| `blur` | 0.6445 | 0.6273 | 0.6403 | 2.7% |
| `shift` | 0.5184 | 0.4414 | 0.5336 | 17% |
| `quantize` | 0.3806 | 0.3311 | 0.3797 | 14% |

Global degradations that touch the border and the interior equally produce
similar drops under all three policies. The policies diverge **only** on
border-localized error, which is what the experiment was designed to isolate.
This is discriminant validity: the effect is not a generic gain in sensitivity.

### 4.3 Gradient distribution

| Policy | Border mass | Interior mass | Total mass | Per-pixel border / interior |
| :--- | ---: | ---: | ---: | ---: |
| `valid` | 0.675 | 0.387 | 1.063 | 4.3× |
| `zero` | 1.308 | 0.331 | 1.639 | 9.8× |
| `masked` | 1.401 | 0.345 | 1.745 | 10.0× |

Two separate effects appear here. First, **redistribution**: the border's share
of gradient mass rises from 63.5% to 80.3%. Second, and more important,
**attenuation**: the total gradient magnitude of the cropped objective is 39%
smaller than the masked objective for the same corrupted input. Cropping does
not merely move sensitivity inward; it removes objective mass overall.

The per-pixel ratio is reported because the ring is only 28.8% of the area. A
share of 63.5% means the average border pixel carries about 4.3× the average
interior pixel's gradient — the noise perturbation is concentrated there, and
SSIM responds to local variance, so this is expected and is not by itself
evidence of correct weighting.

### 4.4 Identity and numerical behavior

All three policies return exactly SSIM = 1 on identical inputs. `masked`
remains finite on constant patches; its worst deviation from 1.0 on a constant
0.4 patch is $6.6\times10^{-5}$, caused by float32 cancellation in
$\mathbb{E}[x^2]-\mu^2$ against $C_2 = 9\times10^{-4}$. This is two orders of
magnitude below any meaningful SSIM difference.

## 5. Analysis

**A1 — Zero-padding dilutes rather than distorts.** The same zeros are added to
prediction and target, so they cancel in the numerator but still enter the
denominator through the means and variances. A border window therefore compares
two tensors that are partly identical by construction. The result is
under-sensitivity, not a spurious target: `zero` retains 62% of `masked`
sensitivity, and the effect is invisible on global degradations.

**A2 — Cropping removes centers, not influence.** The natural hypothesis — that
`valid` produces zero gradient on the ring — is **false**. Border pixels never
serve as window centers, but they still lie inside the windows of centers up to
five pixels inward, so they keep 63.5% of absolute gradient mass. What cropping
removes is the ability of the ring to be evaluated directly; the ring can still
be penalized indirectly, through its effect on near-border centers.

**A3 — Masked is the only policy near area parity.** Area-normalized ratios are
0.275, 0.734, and 1.138. The 14% overshoot above parity is consistent with
sample count: a border center averages fewer real pixels (36 at a corner versus
121 in the interior), so its local variance is noisier and a fixed-amplitude
perturbation degrades it slightly more per unit area. Parity is approximate
because SSIM is nonlinear and saturates near 1.

**A4 — The practical magnitude is bounded by the loss weight.** With defaults
$\alpha=0.85$ and $\lambda=0.3$, the SSIM term carries coefficient
$\lambda(1-\alpha)=0.045$ relative to cross-entropy. A 3.3× change in border
sensitivity is therefore a change in a few percent of the total objective, not
a change in the dominant signal. The reason to care is not magnitude but
alignment: the structural term is precisely the part of the objective that is
supposed to encode boundary topology, and parcel edges at the crop boundary are
common in EuroSAT.

## 6. Decision record

**Context.** The default SSIM implementation in common libraries pads with zeros
(`torchmetrics`, `kornia`) or crops (`skimage`). Both conventions are in wide
use, and neither is documented as a modelling choice.

**Decision.** Use `masked` for the training loss. Keep `valid` for reporting
only, always labelled as cropped, because it is the canonical definition and the
only number comparable to the SSIM literature. `zero` has no role beyond
backwards compatibility.

**Consequences.**

- The objective evaluates all 64×64 centers without synthetic pixels; the cost
  is one extra depthwise convolution for the support mass.
- Training and reporting SSIM are different quantities. They must be named
  differently in MLflow (`train/ssim_surrogate` vs `val/ssim`) and never
  compared to each other.
- Any future change to the window size or patch geometry invalidates the
  numeric ratios in this note; rerun the benchmark.

## 7. Threats to validity

1. **Static perturbations, not model outputs.** Real reconstructions have
   correlated, structured errors. The sensitivity ordering should survive, but
   the exact factors are not guaranteed to transfer to training.
2. **No training ablation yet.** This note measures objective sensitivity, not
   macro-F1. The causal claim — that `masked` yields better border
   reconstruction or better classification — requires the three-arm training
   experiment.
3. **Area normalization assumes linearity.** SSIM saturates near 1 and is
   nonlinear in noise amplitude; parity is a heuristic, not an identity.
4. **Single noise amplitude and single window size.** The direction of the
   effect is expected to be stable; the factors are conditional on
   $\sigma_{\text{noise}}=0.1$ and an 11×11 window.
5. **Single domain and split.** 256 patches from one test split. The bootstrap
   intervals are tight, but they quantify sampling noise within this dataset,
   not transfer to other sensors or resolutions.
6. **Gradient measured on one batch.** It is a local linearization at one point
   in input space, not an average over the training distribution.

## 8. Next experiment

Three training arms with identical seeds, budgets, and architecture, differing
only in `ssim_boundary`, evaluated on:

- macro-F1 on the test split;
- reconstruction MSE and canonical (`valid`) SSIM, reported separately for the
  border ring and the interior;
- L2 latent drift between arms, to check whether the boundary policy changes
  the learned manifold rather than only the decoder's edge behaviour.

A finding is only promoted from "objective sensitivity" to "model effect" when
that ablation clears its confidence interval.

---

# F-002 — Evaluation precision, not just training nondeterminism, sets a ~0.005 macro-F1 comparability floor

**Date:** 2026-10-20
**Artifacts:** `reports/ablation/evidence/run_metrics.json`
**Code:** `src/eos/evaluation.py`, `scripts/build_evidence.py`
**Reproduce:** `uv run --no-sync python scripts/build_evidence.py` (GPU container)

## 1. Why this question exists

Stage F compared classifier and hybrid across four seeds and found a mean
test macro-F1 advantage of +0.0020 in the hybrid's favour (4/4 seed pairs
positive, one-sided paired p = 0.066). An effect this size is only meaningful
if the measurement apparatus resolves it. The logged `test/macro_f1` values
come from Lightning's test loop under `bf16-mixed`, i.e. autocast bfloat16
inference. Re-evaluating the same checkpoints in float32 measures how much of
the comparison is evaluation noise.

## 2. Setup

The evidence builder reloads each run's best checkpoint from the MLflow
artifact store (weights are the same tensors the trainer logged metrics with),
runs the full 4,050-patch test split in float32, and computes macro-F1 from the
aggregated confusion matrix. No training, no sampling: the only variable is
inference precision. Two additional runs bound other noise sources: the
duplicate seed-42 hybrid pair (`330d3e9c…` vs `56955971…`, identical
configuration and seed, nondeterministic kernels) measures run-to-run training
nondeterminism.

## 3. Results

| Run | bf16-mixed (logged) | float32 (re-eval) | Δ |
|---|---:|---:|---:|
| classifier seed 42 | 0.94867 | 0.94852 | −0.0002 |
| hybrid seed 42 (56955971) | 0.94928 | 0.94947 | +0.0002 |
| hybrid seed 42 (330d3e9c) | 0.94933 | 0.95100 | +0.0017 |
| classifier seed 7 | 0.94627 | 0.94699 | +0.0007 |
| hybrid seed 7 | 0.94711 | 0.94535 | −0.0018 |
| classifier seed 123 | 0.94667 | 0.94421 | −0.0025 |
| hybrid seed 123 | 0.95159 | 0.95051 | −0.0011 |
| classifier seed 2024 | 0.94508 | 0.94033 | −0.0048 |
| hybrid seed 2024 | 0.94687 | 0.94813 | +0.0013 |

Mean |Δ| = 0.0016; maximum = 0.0048 (classifier seed 2024, 20 test samples).
Reconstruction metrics are unaffected: test MSE agrees to 4 decimal places
(0.006303 vs 0.006301) because MSE is a smooth average while macro-F1 counts
argmax flips.

**The arm comparison changes with the regime.** Paired deltas (hybrid −
classifier) per seed:

| Seed | bf16-mixed | float32 |
|---|---:|---:|
| 42 | +0.0006 | +0.0009 |
| 7 | +0.0008 | **−0.0016** |
| 123 | +0.0049 | +0.0063 |
| 2024 | +0.0018 | +0.0078 |

Seed 7 flips sign between regimes. Both regimes agree on the mean advantage
(bf16 +0.0020, float32 +0.0034) and on seeds 123 and 2024, but disagree on
which of the small pairs are positive.

Noise budget for a test macro-F1 comparison at this scale: training
nondeterminism ≤ 0.0015 (duplicate-seed pair, float32), evaluation precision
≈ 0.0016 mean / 0.0048 max, cross-seed σ ≈ 0.0015–0.0022. The effect under
test is +0.002–0.003 — inside the combined floor.

## 4. Analysis

**A1 — The mechanism is argmax instability.** bfloat16 carries 8 significand
bits (~2–3 decimal digits). Encoder logits near a decision boundary shift by
~1e-2 relative magnitude under autocast, flipping a small fraction of argmax
decisions. Macro-F1 is a step function of those decisions, so the metric moves
in ~20-sample quanta on a 4,050-sample split; smooth metrics (MSE) do not.

**A2 — This is a comparability floor, not an accuracy claim.** Float32 is not
"more correct" than bf16 as a statement about the trained model deployed in
bf16. The finding is that numbers from the two regimes are not comparable at
effect sizes below ~0.005, and mixing them manufactures or erases differences.

**A3 — The Stage F conclusion survives, weakened.** The direction (hybrid ≥
classifier) holds in both regimes in aggregate; the 4/4-pair sign consistency
observed in bf16 does not survive float32 (3/4). The honest statement is a mean
advantage with a pair-level sign that is not robust to evaluation precision.

## 5. Decision record

**Context.** MLflow holds bf16-mixed test metrics; the evidence bundle holds
float32 re-evaluations of the same weights.

**Decision.** All cross-arm and cross-seed comparisons are made within one
precision regime and labelled with it. The float32 evidence bundle is canonical
for analysis; logged bf16 values are historical records. Any reported
difference below 0.005 macro-F1 must carry the noise budget above alongside it.

**Consequences.** Future runs should log both precisions (a float32
`trainer.test` pass after fit) so the gap is measured, not inferred. The
notebook reports the regime with every metric.

## 6. Threats to validity

1. One hardware stack (RX 9070 XT, ROCm bf16). The floor magnitude is
   hardware-dependent; the direction of the hazard is not.
2. Nine run pairs is a small sample for the |Δ| distribution.
3. Argmax flips could be analyzed directly (per-sample disagreement) to
   localize which classes are precision-sensitive; not done here.

---

# F-003 — The hybrid's class-level gains localize to the cropland complex, not the river/highway pair

**Date:** 2026-10-20
**Artifacts:** `reports/ablation/evidence/run_metrics.json`
**Code:** `src/eos/evaluation.py`, `scripts/build_evidence.py`
**Reproduce:** per-class F1 in `run_metrics.json`, `classification.per_class_f1`

## 1. Why this question exists

The project hypothesis is mechanistic: spatial-spectral reconstruction
regularization preserves boundary topology (river banks, highway corridors,
parcel edges) in the bottleneck, which should help exactly the visually
adjacent class pairs. Aggregate macro-F1 is a dilute endpoint for that claim
(+0.002–0.003 over 10 classes). Per-class F1 on the hypothesized pairs —
River vs Highway, AnnualCrop vs PermanentCrop — is the sensitive test.

## 2. Setup

Float32 re-evaluation (F-002 regime) of 4 classifier and 4 hybrid runs at
identical budgets. Per-class F1 from the aggregated confusion matrix per run;
paired per-seed differences (hybrid − classifier) per class. Class ids follow
the contract order (AnnualCrop, Forest, HerbaceousVegetation, Highway,
Industrial, Pasture, PermanentCrop, Residential, River, SeaLake).

## 3. Results

Mean over 4 seeds, classifier → hybrid:

| Class | clf | hyb | mean Δ | seeds positive |
|---|---:|---:|---:|---:|
| HerbaceousVegetation | 0.9421 | 0.9505 | **+0.0084** | 3/4 |
| AnnualCrop | 0.9320 | 0.9381 | **+0.0061** | **4/4** |
| PermanentCrop | 0.9040 | 0.9082 | +0.0042 | 3/4 |
| Forest | 0.9757 | 0.9792 | +0.0035 | 3/4 |
| Industrial | 0.9604 | 0.9637 | +0.0034 | 3/4 |
| SeaLake | 0.9883 | 0.9911 | +0.0028 | 2/4 |
| River | 0.9201 | 0.9222 | +0.0021 | 2/4 |
| Highway | 0.9159 | 0.9179 | +0.0021 | 2/4 |
| Residential | 0.9906 | 0.9917 | +0.0011 | 2/4 |
| Pasture | 0.9212 | 0.9210 | −0.0002 | 3/4 |

Adjacent-pair slices, per-seed Δ: AnnualCrop [+0.0094, +0.0004, +0.0129,
+0.0018]; PermanentCrop [−0.0082, +0.0040, +0.0041, +0.0169]; Highway
[−0.0041, −0.0041, +0.0093, +0.0071]; River [−0.0059, +0.0006, +0.0146,
−0.0008].

## 4. Analysis

**A1 — One hypothesized pair responds, the other does not.** The crop pair
(AnnualCrop ↔ PermanentCrop) carries the largest and most consistent gains,
and HerbaceousVegetation — the third member of the crop complex — tops the
table. The river/highway pair shows +0.0021 each with 2/4 seeds positive,
inside the F-002 noise budget. Boundary-topology preservation as written
(river banks, highway corridors) is not where the aggregate advantage comes
from.

**A2 — The likely mechanism is texture discrimination, not boundary topology.**
AnnualCrop, PermanentCrop, and HerbaceousVegetation are distinguished by
field-scale texture and phenology, not by linear boundaries. A reconstruction
term that forces the bottleneck to retain spatial detail plausibly helps
texture classification; that is a weaker and different claim than the one in
`plan.md`.

**A3 — Baseline headroom is not the explanation.** River and Highway have the
lowest classifier baselines (0.92) yet small gains; PermanentCrop (0.904) gains
moderately. The gain pattern tracks class identity, not headroom.

## 5. Decision record

**Context.** The hypothesis names boundary topology as the mechanism.

**Decision.** Record the localization as partial support: the reconstruction
regularization helps, but predominantly on the cropland complex. The notebook
findings cell must state the mechanism as "spatial-spectral detail retention"
rather than "boundary topology", and flag the river/highway pair as
unresolved.

**Consequences.** A decisive test would target the crop confusion matrix
(AnnualCrop ↔ PermanentCrop errors specifically), not macro-F1; per-class
sample counts there (~400/class) need ~12 seeds per arm to resolve 0.005-class
effects.

## 6. Threats to validity

1. n = 4 seeds with per-class deltas up to ±0.017 spread; the table ranks
   trends, it does not establish them.
2. Per-class F1 from a single test split of ~400 samples per class is itself
   noisy at the ±0.01 level.
3. The float32 regime is canonical here (F-002); the same table from bf16
   values reorders rows below rank 3.
