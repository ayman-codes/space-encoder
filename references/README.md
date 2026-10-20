# References

This directory is **links-only**. Reference corpora and vendored projects are
cited by URL and never copied into git (AGENTS.md §4.5). Code adapted from any
reference goes into `src/` with its provenance noted in the module docstring.

## Related project — Wafer Defect Detection with Ensemble Meta-Learning

**<https://github.com/ayman-codes/Wafer-Defect-detection-with-Ensemble-meta-learning>**

The direct design lineage for this repository's post-training work. That project
delivers a production multi-label wafer-map classifier with the three techniques
this repository's Phase 4 plan reuses for EuroSAT:

| Technique | There | Here |
|---|---|---|
| ASVD — adaptive SVD rank truncation of dense layers | `src/model/compression.py` | planned for the bottleneck projections and classification head |
| MMD latent-drift telemetry | `src/telemetry/model/mmd_drift.py` | planned as RKHS MMD on the 128-d bottleneck |
| Prometheus/Grafana observability | Pushgateway telemetry loop | shipped: job metrics, EOS Training / Latent Drift / Compression dashboards |

## Similar implementations worth reading

Closest to the research question of this repository — does a reconstruction
objective improve a satellite-image classifier at equal budget?

| Reference | Why it is here |
|---|---|
| [Pchambet/MTL_Classification_Reconstruction](https://github.com/Pchambet/MTL_Classification_Reconstruction) | Multi-seed study of reconstruction-based multi-task learning for satellite classification (EuroSAT → AID). Nearest published neighbour to the Hybrid-CAE hypothesis; useful for framing the F-003 result. |
| [AlessandroAI/Compressive-VAE-Satellite-Classification](https://github.com/AlessandroAI/Compressive-VAE-Satellite-Classification) | VAE latent compression followed by classification fine-tuning on EuroSAT, RSI-CB256, and PatternNet; introduces the Rate–Distortion–Accuracy framing. |

Baseline and pedagogical references for the EuroSAT classification side:

| Reference | Why it is here |
|---|---|
| [atharvadhumal03/EuroSAT-CNN-ClassifierModel](https://github.com/atharvadhumal03/EuroSAT-CNN-ClassifierModel) | Straightforward CNN land-use baseline on EuroSAT (≈ 88% test accuracy). |
| [manojkumar229/pytorch-eurosat-classification](https://github.com/manojkumar229/pytorch-eurosat-classification) | End-to-end PyTorch EuroSAT pipeline with tuning and evaluation loops. |
| [mugheesahmad/EuroSat-PyTorch](https://github.com/mugheesahmad/EuroSat-PyTorch) | Notebook-form EuroSAT classifier, the format `notebooks/01_hybrid_autoencoder.ipynb` is compared against. |
| [handson-geoai — CNNs with PyTorch on EuroSAT](https://handson-geoai.readthedocs.io/en/latest/tutorials/L4_1-EUROSAT.html) | Tutorial notebook: CNN progression on EuroSAT in a teaching setting. |
| [tess-vu — EuroSAT Land Cover Classification](https://tess-vu.github.io/vision-EO/eurosat.html) | Walkthrough across model families (classical ML → CNN → multispectral transfer learning) — a useful ablation-table comparison point. |

## Local notes

- `ROCm/README.md` — notes on PyTorch Lightning under ROCm (AMD blog walkthrough),
  the basis for the container training workflow documented in the repository
  README.

## Primary literature

- Helber, P. et al. (2019). *EuroSAT: A Novel Dataset and Deep Learning
  Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS.
- Wang, Z. et al. (2004). *Image Quality Assessment: From Error Visibility to
  Structural Similarity.* IEEE TIP.
- Gretton, A. et al. (2012). *A Kernel Two-Sample Test.* JMLR.
