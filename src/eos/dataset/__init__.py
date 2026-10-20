"""Dataset contracts, ingestion, curation, and the EuroSAT LightningDataModule."""

from eos.dataset.contract import DatasetContract, load_contract
from eos.dataset.eurosat_datamodule import EuroSATDataModule, EuroSATDataset
from eos.dataset.manifest import ImageRecord, Manifest, ManifestMetadata, read_manifest
from eos.dataset.transforms import build_eval_transform, build_train_transform

__all__ = [
    "DatasetContract",
    "EuroSATDataModule",
    "EuroSATDataset",
    "ImageRecord",
    "Manifest",
    "ManifestMetadata",
    "build_eval_transform",
    "build_train_transform",
    "load_contract",
    "read_manifest",
]
