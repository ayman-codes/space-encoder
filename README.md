# EOS Autoencoder Engine

Hybrid convolutional autoencoder for 64x64 Sentinel-2 (EuroSAT) patches.

The engine couples a shared convolutional encoder with a land-cover
classification head and a spatial reconstruction head, then applies adaptive
singular value decomposition (ASVD) compression and RKHS maximum mean
discrepancy (MMD) latent drift telemetry.

## Status

Phase 0 infrastructure only. Dataset, models, and training arrive in later
phases.

## Quick start

```bash
uv python install 3.12
uv sync --all-groups --extra notebook
uv run ruff check .
uv run mypy
uv run pytest
uv run python scripts/validate_contracts.py
```

## Services

```bash
docker compose -f docker/docker-compose.yml --env-file .env up -d
```

MLflow runs on port 5001, Prometheus on 9092, Pushgateway on 9091, and
Grafana on 3001. Ports 5000/3000/9090 are intentionally avoided because they
are commonly occupied by other local stacks.

MLflow artifacts are stored in the `mlflow_data` volume. An S3-compatible
artifact store is deferred until a supported server image is selected.

## ROCm GPU runtime

```bash
docker compose -f docker/docker-compose.yml --env-file .env --profile gpu build eos-gpu
docker compose -f docker/docker-compose.yml --env-file .env --profile gpu run --rm --no-deps \
  eos-gpu uv run --no-sync python scripts/verify_rocm.py
```

The service publishes JupyterLab on port 8888 with the token from
`JUPYTER_TOKEN`, mounts the repository at `/workspace`, and requests
`/dev/dxg`. WSL2 resolves the ROCm userspace through `/usr/lib/wsl/lib`, so
that directory is mounted read-only and added to `LD_LIBRARY_PATH`; without it
the HIP loader cannot find `libdxcore.so`.

`HSA_OVERRIDE_GFX_VERSION` is deliberately unset: `gfx1201` is native on ROCm
10.1.0 and overriding it degrades performance.
