from pathlib import Path

from data.dataset import DOSADataset
from models import (
    PreProcessor,
    FPNFeatureMapExtractor,
    SentenceTransformerEmbeder,
)


def build_dataset(image_set, args):
    root = Path(args.data_path)
    assert (
        root.exists()
    ), f"provided path {root} to custom dataset does not exist"

    extractor = FPNFeatureMapExtractor(batch_size=args.visual_batch_size)
    embedder = SentenceTransformerEmbeder(batch_size=args.semantic_batch_size)
    preprocessor = PreProcessor(
        args.n_sequence,
        extractor,
        embedder,
        device=args.device,
    )

    training = "train.json"
    validation = "val.json"
    paths = {
        "train": (root / "images", root / "annotations" / training),
        "val": (root / "images", root / "annotations" / validation),
    }
    img_folder, ann_file = paths[image_set]
    return DOSADataset(img_folder, ann_file, preprocessor, args.n_sequence)
