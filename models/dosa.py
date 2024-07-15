import torch
from torch import Tensor
import torch.nn.functional as F

from models.fuse import Fuse
from models.position_encoding import PositionEncoder
from models.transformer import Transformer
from models.relation import RelationMLP


class DOSA(torch.nn.Module):
    def __init__(
        self,
        fuse: Fuse,
        position: PositionEncoder,
        transformer: Transformer,
        parent: RelationMLP,
        sibling: RelationMLP,
        continuation: RelationMLP,
    ):
        super().__init__()
        self._fuse = fuse
        self._position = position
        self._transformer = transformer
        self._parent = parent
        self._sibling = sibling
        self._continuation = continuation

    def forward(
        self,
        samples: dict[str, Tensor],
    ):
        """
        :param samples: The forward expects five tensors in samples
            visual feature map of shape [batch_size, sequence_length, 12544]
            semantic embedding of shape
            [batch_size, sequence_length, embedding_size], the embedding size
            is determined by embedding model
            dimensions information of shape
            [batch_size, sequence_length, dimension]
            position tensors of shape
            [batch_size, sequence_length, position], position corresponds
            [page_no, x1, y1, x2, y2]
            masks: padding mask of shape [batch_size, sequence_length]
        :return: a dict of relation logits, each relation logits is of shape
            [batch_size, sequence_length, sequence_length]
        """
        fused_features = self._fuse(
            samples["visuals"], samples["semantics"], samples["dimensions"]
        )
        position_encodings = self._position(samples["positions"])
        enhanced = self._transformer(
            src=fused_features,
            pos_embed=position_encodings,
            padding_mask=samples["masks"],
        )
        logits = {
            "parent": self._parent(enhanced),
            "sibling": self._sibling(enhanced),
            "continuation": self._continuation(enhanced),
        }
        return logits


class PostProcess(torch.nn.Module):
    @torch.no_grad()
    def forward(self, inputs: dict[str, Tensor], masks: Tensor):
        """
        The forward expects relation logits output from DOSA
        :param inputs: a dict of relation logits, each corresponds to one class
            of relation we infer, of shape [batch_size, n_sequence, n_sequence]
        :param masks of shape, a boolean Tensor of shape
            [batch_size, n_sequence]
        :return: a dict of result per sample, each sample is a dict:
            {
                "parent": {"labels": xx, "scores": xx},
                "sibling": {"labels": xx, "scores": xx},
                "continuation": {"labels": xx, "scores": xx},
            }
        """
        results = []
        for name, logits in inputs.items():
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
    def __init__(self, n_sequence: int, loss_weight: dict[str, int]):
        super().__init__()
        self._n_sequence = n_sequence
        self.loss_weight = loss_weight

    @staticmethod
    def _focal_loss(
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

        return loss.mean(-1).sum()

    @staticmethod
    def _mask_cross_entropy(logit: Tensor, target: Tensor, mask: Tensor):
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
    def loss_relation(logits: Tensor, targets: Tensor, masks: Tensor):
        # sequence is of variable length, we process each sample individually
        # and sum together across the batch
        loss = []
        for sl, st, sm in zip(logits, targets, masks):
            n_t = masks.shape[-1] - sm.sum()
            lt = sl[:n_t, :n_t]
            tt = st[:n_t]
            loss.append(F.cross_entropy(lt.transpose(0, 1), tt))

        loss_ce = torch.stack(loss).mean()
        return loss_ce

    def forward(
        self,
        logits: dict[str, Tensor],
        targets: dict[str, Tensor],
        masks: Tensor,
    ):
        # TODO, right now it's only relation detection, we should be able to
        #   bring in page object classification enhancement loss
        """
        :param logits: a list of relation logits, each is of shape
            [batch_size, n_sequence, n_sequence]
        :param targets: a list of targets, each is of shape
            [batch_size, n_sequence]
        :param masks: mask, boolean Tensor of shape [batch_size, n_sequence]
        :return: a scalar Tensor value for loss of a batch
        """
        loss = {}
        for name, logit in logits.items():
            loss[name] = self.loss_relation(logit, targets[name], masks)
        return loss
