# DOSA

## Introduction

**Abstract.** Document structure analysis is fundamental to information retrieval and document understanding. In complex documents, information is encoded not only in individual page objects such as tables, headers, and text blocks, but also in the structural relations among them. We propose a novel framework, termed **DOcument Structure Analyzer (DOSA)**, for inferring relations between page objects and reconstructing document-level semantic structures. DOSA combines a transformer-based multimodal architecture with a tree-guided context-building algorithm designed to address the attention and sequence-length limitations of transformer when processing long documents. Specifically, DOSA extracts and fuses visual, textual, and layout features for each page object, refines these representations using a transformer encoder to capture contextual dependencies, and predicts hierarchical and ordering relations. To scale to long documents, DOSA processes page objects incrementally by dividing documents into chunks and sequentially performing inference, while dynamically constructing a semantic tree from previously processed chunks to guide subsequent predictions. Experimental results on multiple benchmarks demonstrate the effectiveness of the proposed approach.

## Usage

### Dataset preparation
Organize data as following:
```
code_root/
└── dataset/
    └── dochienet/
        ├── images/
        │   ├── prefix1/
        │   │   ├── page1.png
        │   │   ├── page2.png
        │   │   └── ...
        │   └── ...
        └── annotations/
            ├── train.json
            ├── val.json
            └── benchmark.json

```

### Training

For example, the command for training DOSA on single GPUs is as following:

```bash
GPUS=1 GPUS_PER_NODE=1 poetry run ./dosa/tools/run_dist_launch.sh --batch_size 2 --data_path ./dataset/dochienet --output_dir output --n_sequence 256 --semantic_model_name sentence-transformers/distiluse-base-multilingual-cased-v2 --d_semantic_in 768 --d_semantic_out 512 --num_workers 0
```

### Evaluation
To evaluate DOSA on DocHieNet with a single GPU run:
```bash
poetry run python -m dosa.data.dochienet.benchmark --context_window 8 --resume output/checkpoint.pth --n_sequence 256 --semantic_model_name sentence-transformers/distiluse-base-multilingual-cased-v2 --d_semantic_in 768 --d_semantic_out 512 --data_path ./dataset/dochienet
```
