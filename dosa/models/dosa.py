import torch
from torch import Tensor
import torch.nn.functional as F

from dosa.core.sample import Samples
from dosa.models import Backbone
from dosa.models.fuse import Fuse
from dosa.models.position_encoding import PositionEncoder
from dosa.models.transformer import Transformer
from dosa.models.relation import RelationMLP


class DOSA(torch.nn.Module):
    def __init__(
        self,
        backbone: Backbone,
        fuse: Fuse,
        position: PositionEncoder,
        transformer: Transformer,
        parent: RelationMLP,
    ):
        super().__init__()
        self._backbone = backbone
        self._fuse = fuse
        self._position = position
        self._transformer = transformer
        self._parent = parent

    def forward(self, samples: Samples):
        """
        :param samples: The forward expects five tensors in samples
            visual feature map of shape [batch_size, sequence_length, 12544]
            semantic embedding of shape
            [batch_size, sequence_length, embedding_size], the embedding size
            is determined by embedding model
            measurement information of shape
            [batch_size, sequence_length, measurement]
            position tensors of shape
            [batch_size, sequence_length, position], position corresponds
            [page_no, x1, y1, x2, y2]
            masks: padding mask of shape [batch_size, sequence_length]
        :return: a dict of relation logits, each relation logits is of shape
            [batch_size, sequence_length, sequence_length]
        """
        visuals, semantics, measurement, category, positions, masks = samples
        visuals, semantics = self._backbone(visuals, semantics)
        fused_features = self._fuse(visuals, semantics, measurement, category)
        position_encodings = self._position(positions)
        enhanced = self._transformer(
            src=fused_features,
            pos_embed=position_encodings,
            padding_mask=masks,
        )
        logits = {
            "parent": self._parent(enhanced),
            "masks": masks,
        }
        return logits


class PostProcess(torch.nn.Module):
    @torch.no_grad()
    def forward(self, outputs: dict[str, Tensor]):
        """
        The forward expects relation logits output from DOSA
        :param outputs: a dict of relation logits, each corresponds to one class
            of relation output from model dosa, of shape
            [batch_size, n_sequence, n_sequence]
        :return: a list of sample result, each sample result is a dict:
            {
                "parent": {"labels": xx, "scores": xx},
            }
        """
        cloned = outputs.copy()
        masks = cloned.pop("masks")
        results = []
        for name, logits in cloned.items():
            result = []
            for logit, mask in zip(logits, masks):
                # truncate the tensor to non-padding token only
                n_truncated = mask.shape[-1] - mask.sum()
                logit = logit[:n_truncated, :n_truncated, ...]
                prob = F.softmax(logit, -1)
                scores, labels = prob.max(-1)
                result.append({name: {"scores": scores, "labels": labels}})
            results.append(result)
        transposed = []
        for row in zip(*results):
            transposed.append({k: v for d in row for k, v in d.items()})
        return transposed


class DOSACriterion(torch.nn.Module):
    def __init__(
        self,
        n_sequence: int,
        loss_weight: dict[str, int],
        focal_alpha: float,
    ):
        super().__init__()
        self._n_sequence = n_sequence
        self._loss_weight = loss_weight
        self._focal_alpha = focal_alpha

    @property
    def loss_weight(self):
        return self._loss_weight

    @staticmethod
    def mask_cross_entropy(logit: Tensor, target: Tensor, mask: Tensor):
        """
        :param logit: relation logit of shape
            [batch_size, n_sequence, n_sequence]
        :param target: target of shape, [batch_size, n_sequence]
        :param mask: mask of shape, [batch_size, n_sequence]
        :return: a scalar number Tensor
        """
        # Compute the standard cross-entropy loss without reduction
        # looks like this is correct, but the target data points to some
        # invalid position of -inf, and therefore has -inf/-inf to get nan
        loss = F.cross_entropy(logit, target, reduction="none")

        # Apply the mask
        masked_loss = loss * mask

        # Normalize by the number of valid (non-masked) elements
        valid_elements = mask.sum()
        masked_loss = masked_loss.sum() / valid_elements

        return masked_loss

    @staticmethod
    def weighted_mask_ce(
        logits: Tensor,
        targets: Tensor,
        masks: Tensor,
        eos: float = None,
    ):
        """This handles self relation imbalance issue"""
        # sequence is of variable length, we process each sample individually
        # and sum together across the batch
        loss = []
        for logit, target, mask in zip(logits, targets, masks):
            n_truncated = masks.shape[-1] - mask.sum()
            lt = logit[:n_truncated, :n_truncated]
            tt = target[:n_truncated]
            # loss.append(F.cross_entropy(lt.transpose(0, 1), tt))
            if eos:
                ce = F.cross_entropy(lt, tt, reduction="none")
                condition = (
                    torch.arange(
                        0, target.shape[-1], dtype=lt.dtype, device=masks.device
                    )[:n_truncated]
                    == tt
                )
                tw = torch.where(condition, eos, 1.0 / eos)
                loss.append((ce * tw).mean())
            else:
                loss.append(F.cross_entropy(lt, tt))
        loss_ce = torch.stack(loss).mean()
        return loss_ce

    @staticmethod
    def _sigmoid_focal_loss(
        inputs: Tensor, targets: Tensor, alpha: float = 0.25, gamma: float = 2
    ):
        """
        Employ focal loss considering the imbalance of relation and non-relation
        entities, modified from torchvision.ops.sigmoid_focal_loss.
        :param inputs: in theory, any shape is accepted, here, we take relation
            entity float Tensor of shape [n_sequence, n_sequence, num_classes]
        :param targets: with same shape as inputs, Stores the one hot vector
            representation of each classification label.
        :param alpha: (optional) Weighting factor in range (0,1) to balance
            positive vs negative examples. Default = -1 (no weighting).
        :param gamma: Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples.
        :return: a scalar Tensor for loss sum
        """
        p = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss

    def focal_relation_loss(self, logits, targets, masks):
        """
        Compute relation loss based on sigmoid focal loss
        1. Create on hot vector from targets and run focal loss
        2. Create a filter of shape
            from mask by inverting 0s and 1s and apply on the focal loss result
        :param logits: relation logits tensor of shape
            [batch_size, sequence_size, 11]
        :param targets: targets tensor of shape [batch_size, sequence_size]
        :param masks: masks tensor of shape [batch_size, sequence_size
        :return: loss tensor
        """
        onehot = torch.zeros(
            logits.size(),
            dtype=logits.dtype,
            layout=logits.layout,
            device=logits.device,
        )
        onehot.scatter_(2, targets.unsqueeze(-1), 1)
        loss = self._sigmoid_focal_loss(
            logits, onehot, alpha=self._focal_alpha, gamma=2
        )

        filters = 1 - masks.to(torch.int)
        filters = torch.stack([torch.outer(row, row) for row in filters])
        loss = loss * filters
        return loss.sum() / filters.sum()

    def forward(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]):
        """
        :param outputs: a dict of relation logits output from dosa corresponding
            to parent, sibling and continuation, each is of shape
            [batch_size, n_sequence, n_sequence]
        :param targets: a dict of targets corresponding to parent, sibling and
            continuation, each is of shape [batch_size, n_sequence]
        :return: a scalar Tensor value for loss of a batch
        """

        masks = outputs["masks"]
        loss = {
            "parent": self.focal_relation_loss(
                outputs["parent"], targets["parent"], masks
            )
            * self._loss_weight["parent"],
        }
        return loss
