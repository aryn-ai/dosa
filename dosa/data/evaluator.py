import torch
import torch.distributed as dist

from dosa.util.misc import is_dist_avail_and_initialized


class DosaEvaluator(object):
    """Evaluator for training and validation, including F1 only"""

    def __init__(self, device):
        self.parent = 0.0
        self.count = 0
        self.context_parent = 0.0
        self.context_count = 0.0
        self.device = device

    def update(self, predictions, targets):
        for sid, prediction in enumerate(predictions):
            # Get the real length after truncating padding
            p_truncated = len(prediction["parent"]["labels"])
            truncated = p_truncated

            parent = targets["parent"][sid][:truncated]
            context = targets["context"][sid][:truncated]

            self.parent += (prediction["parent"]["labels"] == parent).sum()
            self.count += truncated

            self.context_parent += (
                (prediction["parent"]["labels"] == parent) * context
            ).sum()
            self.context_count += context.sum()

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return

        t = torch.tensor(
            [
                self.parent,
                self.count,
                self.context_parent,
                self.context_count,
            ],
            dtype=torch.float64,
            device=self.device,
        )
        # TODO, this barrier is duplicated?
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.parent = float(t[0])
        self.count = int(t[1])
        self.context_parent = float(t[2])
        self.context_count = int(t[3])

    def summarize(self):
        print("Dosa Evaluation Stats:")
        row = " {:<18} for [ relation={:>12s}]= {:0.3f}"
        title = "Average Precision (AP)"

        parent = self.parent / self.count
        print(row.format(title, "parent", parent))

        context_parent = (
            self.context_parent / self.context_count
            if self.context_count
            else 1.0
        )

        print(row.format(title, "context parent", context_parent))

        non_context_parent = (self.parent - self.context_parent) / (
            self.count - self.context_count
        )
        print(row.format(title, "non context parent", non_context_parent))
