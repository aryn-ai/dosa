# DOSA

## Introduction

**Abstract.** Document Layout parsing is a crucial task for document understanding.
Most work today focus on page object detection but ignore the rich relation information
among those page objects in a document. In this paper, we present a multi-stage end
to end document structure analysis system called DOSA(DOcument Structure Analyzer).
DOSA is able to localize page objects in a document, detect relationship among these
objects and build a semantic hierarchical tree based on the detected relationship.
DOSA also provides a set of operators leveraging document tree structure for solving
tricky semantic filtering and chunking problem.

## Usage

### Dataset preparation
Organize data as following:
```
code_root/
└── data/
    └── doc/
        ├── images/
        └── annotations/
        	├── train.json
        	└── val.json
```

### Training

#### Training on single node

For example, the command for training DOSA on 8 GPUs is as following:

```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./tools/train_dosa.sh
```

#### Training on multiple nodes

For example, the command for training DOSA on 2 nodes of each with 8 GPUs is as following:

On node 1:

```bash
MASTER_ADDR=<IP address of node 1> NODE_RANK=0 GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 16 ./tools/train_dosa.sh
```

On node 2:

```bash
MASTER_ADDR=<IP address of node 1> NODE_RANK=1 GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 16 ./tools/train_dosa.sh
```
