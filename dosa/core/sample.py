from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Samples:
    visuals: list[tuple[Tensor, list[Tensor], list[tuple[int, int]]]]
    semantics: list[dict[str, Tensor]]
    measurement: Tensor
    category: Tensor
    positions: Tensor
    masks: Tensor
    """
    One Samples object corresponds to a batch of N samples, each sample contains
    M pages, but M may vary in different samples.

    visuals: a list of N tuples, each tuple has an image tensor containing M
        page images in one sample of shape M x W x H x C, a list of M Tensors
        for bounding boxes in each page, a list of M tuples for each page shape
    semantics: a list of N dict, each dict is the tokenized representation for
        all page objects in one sample, each has 3 fields including
        {'input_ids': [...], 'attention_mask': [...], 'token_type_ids': [...]}
    measurement: aligned dimension embedding of shape: N x n_sequence x 3
        including width, height and area
    positions: aligned position embedding of shape: N x n_sequence x 5 including
        x1, y1, x2, y2, page_no /page_count
    masks: padding mask of shape: N x n_sequence
    """

    def to(self, device):
        visuals = []
        for images, boxes, shapes in self.visuals:
            images = images.to(device)
            boxes = [box.to(device) for box in boxes]
            visuals.append((images, boxes, shapes))
        self.visuals = visuals

        self.semantics = [
            {k: v.to(device) for k, v in sample.items()}
            for sample in self.semantics
        ]

        self.measurement = self.measurement.to(device)
        self.category = self.category.to(device)
        self.positions = self.positions.to(device)
        self.masks = self.masks.to(device)

    @staticmethod
    def collate(samples: list["Samples"]):
        batch_visuals = []
        batch_semantics = []
        batch_measurement = []
        batch_category = []
        batch_positions = []
        batch_masks = []

        for sample in samples:
            batch_visuals.extend(sample.visuals)
            batch_semantics.extend(sample.semantics)
            batch_measurement.append(sample.measurement)
            batch_category.append(sample.category)
            batch_positions.append(sample.positions)
            batch_masks.append(sample.masks)

        collated = Samples(
            batch_visuals,
            batch_semantics,
            torch.concat(batch_measurement),
            torch.concat(batch_category),
            torch.concat(batch_positions),
            torch.concat(batch_masks),
        )
        return collated

    def __iter__(self):
        return iter(
            (
                self.visuals,
                self.semantics,
                self.measurement,
                self.category,
                self.positions,
                self.masks,
            )
        )
