# SSIM Boundary Policy Benchmark

Generated: 2026-10-18T16:07:36+00:00
Dataset: `eurosat-rgb-v1.0.0` · 256 patches · manifest `f7ccc0bccafc3b9679c4ef2ac934e0df7d2710628119cf6171784561f9754570`
Window 11x11, sigma=1.5, border 5 px, 2000 bootstrap resamples.

## Headline

| Policy | Map | Border drop | Interior drop | Area-normalized ratio | Border gradient fraction |
| :--- | :---: | ---: | ---: | ---: | ---: |
| `valid` | [54, 54] | 0.07923 | 0.71174 | 0.275 | 0.635 |
| `zero` | [64, 64] | 0.16294 | 0.54830 | 0.734 | 0.798 |
| `masked` | [64, 64] | 0.26120 | 0.56705 | 1.138 | 0.803 |

## Per-perturbation SSIM

| Policy | Perturbation | Mean SSIM | Drop vs identity | 95% CI |
| :--- | :--- | ---: | ---: | :--- |
| `valid` | `identity` | 1.00000 | 0.00000 | [0.00000, 0.00000] |
| `valid` | `border_noise` | 0.92077 | 0.07923 | [0.07565, 0.08312] |
| `valid` | `interior_noise` | 0.28826 | 0.71174 | [0.69211, 0.73182] |
| `valid` | `border_constant` | 0.88095 | 0.11905 | [0.11240, 0.12633] |
| `valid` | `blur` | 0.35554 | 0.64446 | [0.62747, 0.66120] |
| `valid` | `shift` | 0.48159 | 0.51841 | [0.48103, 0.55390] |
| `valid` | `quantize` | 0.61944 | 0.38056 | [0.36808, 0.39300] |
| `zero` | `identity` | 1.00000 | 0.00000 | [0.00000, 0.00000] |
| `zero` | `border_noise` | 0.83706 | 0.16294 | [0.15664, 0.16974] |
| `zero` | `interior_noise` | 0.45170 | 0.54830 | [0.53281, 0.56429] |
| `zero` | `border_constant` | 0.80690 | 0.19310 | [0.18365, 0.20338] |
| `zero` | `blur` | 0.37267 | 0.62733 | [0.61532, 0.63917] |
| `zero` | `shift` | 0.55861 | 0.44139 | [0.41017, 0.47150] |
| `zero` | `quantize` | 0.66890 | 0.33110 | [0.32008, 0.34193] |
| `masked` | `identity` | 1.00000 | 0.00000 | [0.00000, 0.00000] |
| `masked` | `border_noise` | 0.73880 | 0.26120 | [0.25355, 0.26912] |
| `masked` | `interior_noise` | 0.43295 | 0.56705 | [0.55054, 0.58399] |
| `masked` | `border_constant` | 0.73218 | 0.26782 | [0.26114, 0.27501] |
| `masked` | `blur` | 0.35973 | 0.64027 | [0.62359, 0.65626] |
| `masked` | `shift` | 0.46642 | 0.53358 | [0.49701, 0.56850] |
| `masked` | `quantize` | 0.62029 | 0.37971 | [0.36742, 0.39180] |

## Reproduce

```bash
uv run python scripts/benchmark_ssim.py
```
