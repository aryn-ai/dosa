import torch
import torch.distributed as dist

from util.misc import is_dist_avail_and_initialized


class DosaEvaluator(object):
    """
    Evaluator for different stats:
    relation detection is based solely on the mAP, we compute it based on each
    single relation type and average over them.
    """

    def __init__(self, device):
        self.parent = 0.0
        self.sibling = 0.0
        self.continuation = 0.0
        self.count = 0
        self.device = device

    def update(self, predictions, targets, masks):
        for sid, prediction in enumerate(predictions):
            mask = masks[sid]
            truncated = mask.shape[-1] - mask.sum()

            parent = targets["parent"][sid][:truncated]
            sibling = targets["sibling"][sid][:truncated]
            continuation = targets["continuation"][sid][:truncated]

            # TODO, should compute the precision each sample and average
            #   over the whole
            self.parent += (
                prediction["parent"]["labels"] == parent
            ).sum() / truncated
            self.sibling += (
                prediction["sibling"]["labels"] == sibling
            ).sum() / truncated
            self.continuation += (
                prediction["continuation"]["labels"] == continuation
            ).sum() / truncated
            self.count += 1

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return

        t = torch.tensor(
            [self.parent, self.sibling, self.continuation, self.count],
            dtype=torch.float64,
            device=self.device,
        )
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.parent = float(t[0])
        self.sibling = float(t[1])
        self.continuation = float(t[2])
        self.count = int(t[3])

    def summarize(self):
        print("Dosa Evaluation Stats:")
        row = " {:<18} for [ relation={:>12s}]= {:0.3f}"
        title = "Average Precision (AP)"
        parent = self.parent / self.count
        sibling = self.sibling / self.count
        continuation = self.continuation / self.count
        overall = (parent + sibling + continuation) / 3
        print(row.format(title, "overall", overall))
        print(row.format(title, "parent", parent))
        print(row.format(title, "sibling", sibling))
        print(row.format(title, "continuation", continuation))
