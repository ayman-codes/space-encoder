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

# F-004 — ASVD rank truncation buys 39% dense-parameter reduction at noise-level macro-F1 cost, and only the energy band decides whether M4 is met

**Date:** 2026-10-20
**Artifacts:** `reports/compression/benchmark.json`
**Code:** `src/eos/model/compression.py`, `scripts/run_asvd_benchmark.py`
**Reproduce:** `uv run python scripts/run_asvd_benchmark.py --checkpoint <ckpt> …`
per checkpoint; the JSON carries every layer record, rank, and metric.

## 1. Why this question exists

`plan.md` milestone M4 commits the project to "≥ 30% parameter reduction in
dense heads with < 1.5% macro-F1 loss" after ASVD factorization of the trained
model. The `ASVDConfig` contract constrains the experiment to the energy band
E ∈ [0.90, 0.95], forbids factorizing the calibrated logits layer, and
documents the hidden classifier projection (64×128) as a likely negative
control. Whether the milestone holds — and at which energy — is empirical.

## 2. Setup

Stored Stage-E seed-42 best checkpoints (hybrid, classifier, autoencoder),
freshly reloaded per variant because factorization rewrites weights in place.
Float32 re-evaluation of the 4,050-patch test split (F-002 regime) via
`src/eos/evaluation.py`; every scope (`classifier`, `bottleneck`,
`all_intermediate`) swept over the energy band endpoints {0.90, 0.95}; the
compression-gain guard requires `mn/(k(m+n)) ≥ 1.1` or the layer is recorded as
skipped with its measured rank. Latency: CPU forward, batch 128, median of 10
passes after 3 warm-up passes.

## 3. Results

Hybrid checkpoint (6,453,421 parameters, baseline macro-F1 0.94947):

| Scope | E | Ranks (enc/dec/hidden) | Dense Δ | Model Δ | macro-F1 Δ |
|---|---:|---|---:|---:|---:|
| classifier | 0.90 | — / — / skip(39) | 0.0% | 0.0% | +0.0002 |
| bottleneck | 0.90 | 48 / 105 / — | **−39.15%** | −12.77% | **+0.0002** |
| bottleneck | 0.95 | 80 / skip(115) / — | −18.19% | −5.93% | +0.0000 |
| all_intermediate | 0.90 | 48 / 105 / skip(39) | −38.99% | −12.77% | +0.0002 |

Classifier checkpoint (2,609,546 parameters, baseline macro-F1 0.94852):
bottleneck E=0.90 factorizes `encoder.projection` at rank 47 → **−62.70%**
dense (−25.20% model) at +0.0002 macro-F1; E=0.95 → −36.52%. Autoencoder:
−12.27% at E=0.90 (ranks 109/112), everything gain-guard-skipped at E=0.95.

**M4 verdict: met at E = 0.90** — 39.2% dense reduction at +0.0002 macro-F1
for the champion architecture, and 62.7% for the pure classifier.

## 4. Analysis

**A1 — The energy band, not the scope, decides the milestone.** At E = 0.95
the hybrid retains 80/128 encoder components and the decoder projection misses
the gain floor by 0.004 (1.096 vs 1.1) and is skipped entirely: 18.2% is the
ceiling there. At E = 0.90 the same layers drop to ranks 48 and 105 and clear
30% with margin. Reporting an ASVD result without its energy setting is
meaningless.

**A2 — Macro-F1 is insensitive to rank truncation at this scale.** All deltas
sit inside ±0.0003, an order of magnitude below the F-002 comparability floor
(≈ 0.005). The factorization is near-lossless for argmax decisions at these
ranks; the price of compression on the classification endpoint is not
resolvable at n = 4050 test samples, and the honest statement is preservation,
not improvement (the +0.0002 signs are argmax noise).

**A3 — The classifier-only scope is the documented negative control, exactly
as `ASVDConfig` predicted.** `classifier.hidden` is skipped at every energy —
measured ranks 39/47 against a 64×128 weight — with the skip recorded, never
silent. The bottleneck projections (128×8192 each) carry 99.6% of the dense
set and are the only meaningful ASVD target.

**A4 — Wall-clock speedup is not demonstrated on CPU.** Two measurement passes
under different system load disagree in sign (autoencoder bottleneck: −18% in
one pass, +24% in the other; hybrid 201–254 ms against a 206–265 ms baseline).
The factorized projection cuts MACs ~2.7× at rank 47–48, but the hybrid forward
is convolution-dominated, so the achievable latency win is small and swamped
by measurement noise on a loaded host. The parameter and memory reduction is
the robust claim.

## 5. Decision record

**Context.** The deliverable ablation table needs an ASVD-compressed arm, and
M4 fixes its acceptance bar.

**Decision.** The compressed arm is `bottleneck` scope at E = 0.90 (ranks
48/105 on the hybrid): 39.15% dense-parameter reduction at preserved
macro-F1. Dense-set and whole-model reductions are reported separately
(−39.15% vs −12.77%) because the convolutions dominate the model. The
classifier-scope result ships as the negative control.

**Consequences.** The notebook ablation table's fourth row is filled from
`benchmark.json`; any MMD baseline calibrated on the uncompressed champion is
stale after factorization and must be recalibrated (F-005 demonstrates the
enforcement); latency claims stay out of the headline until the quiet-GPU
measurement lands.

## 6. Threats to validity

1. One checkpoint per arm (seed 42); the multi-seed campaign does not repeat
   ASVD, so rank positions are not distributional claims.
2. Latency evidence is CPU-only under concurrent load and disagrees in sign
   across passes; the quiet-GPU pass is pending and A4 may sharpen into a real
   speedup claim or a bounded null.
3. Macro-F1 preservation could hide logit drift that per-class F1 at n ≈ 405
   cannot resolve; the reconstruction side of the hybrid (decoder rank 105) is
   covered by recon MSE in the artifact and moves by less than 1% relative.
4. The gain-guard threshold (1.1) is a configuration choice; the decoder skip
   at E = 0.95 sits 0.4% from the floor, so the boundary is tight.

# F-005 — The unbiased MMD U-statistic with a live-batch-size bootstrap separates class-mix and radiometric drift from the in-distribution null

**Date:** 2026-10-20
**Artifacts:** `reports/telemetry/mmd_baseline.json`,
`reports/telemetry/mmd_reference.npz`, `reports/telemetry/drift_demo.json`
**Code:** `src/eos/telemetry/mmd.py`, `src/eos/telemetry/mmd_drift.py`,
`scripts/calibrate_mmd.py`
**Reproduce:** `uv run python scripts/calibrate_mmd.py --checkpoint <ckpt>
--demonstrate --push`

## 1. Why this question exists

`plan.md` Phase 4 requires live latent-drift telemetry: a calibrated RKHS
baseline over validation embeddings and a two-sample test with an alarm at
μ_base + 3σ_base. Two estimator choices decide whether that alarm is usable:
the biased V-statistic inflates when baseline and batch sizes differ, and a
null bootstrapped at the wrong batch size calibrates a threshold that no live
batch is ever measured against.

## 2. Setup

Champion hybrid checkpoint; bottleneck embeddings over the full validation
split (4,050 = 405/class, L2-normalized). Bandwidths σ_med × {0.25, 0.5, 1.0,
2.0, 4.0} from the median heuristic over validation pairwise distances.
Reference: seeded 500-row subsample. Null: 500 bootstrap draws of two disjoint
128-row blocks (the live batch size) scored with the unbiased U-statistic
(diagonals removed). Live scoring: batch vs full reference. Baseline
fingerprint binds checkpoint digest + ASVD state + manifest digest.

## 3. Results

Calibrated: μ = 0.00003, σ = 0.00928, threshold = **0.02787** (μ + 3σ).

| Scenario | mmd² | Alarm | × threshold |
|---|---:|:---:|---:|
| Random cross-class validation batch | **−0.0094** | no | — |
| Same batch, +0.25 radiometric pixel shift | **0.6316** | **yes** | 22.7× |
| Single-class slice of the class-ordered loader | **0.9260** | **yes** | 33.2× |

Stale-baseline enforcement: a monitor bound to a post-ASVD fingerprint refuses
the pre-ASVD baseline with "baseline is stale for this model state;
recalibrate after any ASVD pass".

## 4. Analysis

**A1 — The unbiased estimator centers correctly at zero.** The in-distribution
score is slightly negative (−0.0094), which is expected for the U-statistic and
would be impossible for the biased V-statistic; at n = 128 vs N = 500 the
V-form adds a diagonal term that inflates exactly where the alarm threshold is
decided.

**A2 — The monitor detects class-mix shift as well as radiometric shift.** The
initial demonstration batch was drawn from the loader in order and turned out
to be 128 samples of a single class (the curated validation loader streams
class-ordered); it alarmed at 33× threshold. That is correct behavior — a
single-class batch *is* distribution shift against a mixed reference — and it
is now a documented capability rather than a demonstration artifact.

**A3 — The bootstrap batch size is the calibration's load-bearing choice.**
The null is bootstrapped as block-vs-block at n = 128 while live scoring
compares n = 128 against N = 500; the n-vs-n null is the conservative choice
(the estimator variance is larger than the live n-vs-N case), which the clean
null score confirms empirically.

## 5. Decision record

**Context.** Telemetry scope was set to the champion model only (highest macro
and micro F1 across the campaign evidence), and any ASVD pass rewrites z.

**Decision.** MMD telemetry attaches to the champion; the baseline fingerprint
binds checkpoint + ASVD state + manifest, and `DriftMonitor` hard-refuses
mismatched fingerprints instead of silently scoring. The four dashboard series
(`eos_mmd_score`, `eos_mmd_alarm`, `eos_asvd_param_ratio`, `eos_asvd_f1_delta`)
are exported through the Pushgateway with per-scenario grouping labels.

**Consequences.** A final champion calibration runs after the confirmation
campaign; the ASVD-compressed champion variant (F-004) carries its own
recalibrated baseline in the artifact set; Grafana drift/compression panels are
populated live.

## 6. Threats to validity

1. The current baseline is calibrated on the seed-42 hybrid checkpoint; the
   champion designation awaits the n = 12 campaign evidence and the baseline
   will be recalibrated on it.
2. Drift power is demonstrated at one radiometric magnitude (+0.25) and one
   class-mix shift; weaker shifts and other shift families (noise, blur,
   geometric) are unmeasured.
3. The demonstration batches overlap the calibration population (same
   validation split); a true held-out stream would be stricter.
4. Bootstrap draws are 500; σ at that count is stable to ~3% but the extreme
   tail behind a 3σ threshold is estimated, not known.
