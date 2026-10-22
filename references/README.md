# References


## Related project — Wafer Defect Detection with Ensemble Meta-Learning

**<https://github.com/ayman-codes/Wafer-Defect-detection-with-Ensemble-meta-learning>**

The direct design lineage for this repository's post-training work. The project is part of my VLBA: systems architecure course.
project delivers a  multi-label wafer-map classifier with the three techniques
this repository's Phase 4 plan reuses for EuroSAT:

| Technique | There | Here |
|---|---|---|
| ASVD — adaptive SVD rank truncation of dense layers | `src/model/compression.py` | planned for the bottleneck projections and classification head |
| MMD latent-drift telemetry | `src/telemetry/model/mmd_drift.py` | planned as RKHS MMD on the 128-d bottleneck |
| Prometheus/Grafana observability | Pushgateway telemetry loop | shipped: job metrics, EOS Training / Latent Drift / Compression dashboards |

## Similar implementations I took inspiration from

does a reconstruction objective improve a satellite-image classifier at equal budget?

| Reference | Why it is here |
|---|---|
| [Pchambet/MTL_Classification_Reconstruction](https://github.com/Pchambet/MTL_Classification_Reconstruction) | Multi-seed study of reconstruction-based multi-task learning for satellite classification (EuroSAT → AID). 
| [AlessandroAI/Compressive-VAE-Satellite-Classification](https://github.com/AlessandroAI/Compressive-VAE-Satellite-Classification) | VAE latent compression followed by classification fine-tuning on EuroSAT, RSI-CB256, and PatternNet; introduces the Rate–Distortion–Accuracy framing. |

Baseline and pedagogical references for the EuroSAT classification side:

| Reference | Why it is here |
|---|---|
| [atharvadhumal03/EuroSAT-CNN-ClassifierModel](https://github.com/atharvadhumal03/EuroSAT-CNN-ClassifierModel) | Straightforward CNN land-use baseline on EuroSAT (≈ 88% test accuracy). |
| [manojkumar229/pytorch-eurosat-classification](https://github.com/manojkumar229/pytorch-eurosat-classification) | End-to-end PyTorch EuroSAT pipeline with tuning and evaluation loops. |
| [mugheesahmad/EuroSat-PyTorch](https://github.com/mugheesahmad/EuroSat-PyTorch) | Notebook-form EuroSAT classifier, the format `notebooks/01_hybrid_autoencoder.ipynb` is inspired from.

## Local notes

- `ROCm/README.md` — notes on PyTorch Lightning under ROCm (AMD blog walkthrough),
  the basis for the container training workflow

