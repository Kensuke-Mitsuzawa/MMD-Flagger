from pathlib import Path
import toml

from hallucination_mt.dale_2023_halomi.load_dataset import load_dataset


def test_load_dataset(resource_path_root: Path):

    path_dataset_tsv = Path(resource_path_root / 'eval_datasets/halomi_core.tsv')
    assert path_dataset_tsv.exists(), f"Dataset file {path_dataset_tsv} does not exist."
    assert path_dataset_tsv.is_file(), f"Dataset file {path_dataset_tsv} is not a file."

    # Load the dataset
    seq_halomi_dataset = load_dataset(path_halomi_dataset_tsv=path_dataset_tsv)

    assert len(seq_halomi_dataset) > 0, "No records found in the dataset."

