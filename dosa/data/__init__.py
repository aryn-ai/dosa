from pathlib import Path

from dosa.data.dataset import DOSADataset
from dosa.models import PreProcessor


def build_dataset(image_set, args):
    root = Path(args.data_path)
    assert (
        root.exists()
    ), f"provided path {root} to custom dataset does not exist"

    preprocessor = PreProcessor(args.n_sequence, args.semantic_model_name)

    training = "train.json"
    validation = "val.json"
    paths = {
        "train": (root / "images", root / "annotations" / training),
        "val": (root / "images", root / "annotations" / validation),
    }
    img_folder, ann_file = paths[image_set]
    return DOSADataset(img_folder, ann_file, preprocessor, args.n_sequence)
