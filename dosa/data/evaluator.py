import torch
import torch.distributed as dist

from dosa.util.misc import is_dist_avail_and_initialized


class DosaEvaluator(object):
    """Evaluator for training and validation, including F1 only"""

    def __init__(self, device):
        self.parent = 0.0
        self.sibling = 0.0
        self.count = 0
        self.context_parent = 0.0
        self.context_sibling = 0.0
        self.context_count = 0.0
        self.device = device

    def update(self, predictions, targets):
        for sid, prediction in enumerate(predictions):
            # Get the real length after truncating padding
            p_truncated = len(prediction["parent"]["labels"])
            s_truncated = len(prediction["sibling"]["labels"])
            assert p_truncated == s_truncated
            truncated = p_truncated

            parent = targets["parent"][sid][:truncated]
            sibling = targets["sibling"][sid][:truncated]
            context = targets["context"][sid][:truncated]

            self.parent += (prediction["parent"]["labels"] == parent).sum()
            self.sibling += (prediction["sibling"]["labels"] == sibling).sum()
            self.count += truncated

            self.context_parent += (
                (prediction["parent"]["labels"] == parent) * context
            ).sum()
            self.context_sibling += (
                (prediction["sibling"]["labels"] == sibling) * context
            ).sum()
            self.context_count += context.sum()

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return

        t = torch.tensor(
            [
                self.parent,
                self.sibling,
                self.count,
                self.context_parent,
                self.context_sibling,
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
        self.sibling = float(t[1])
        self.count = int(t[2])
        self.context_parent = float(t[3])
        self.context_sibling = float(t[4])
        self.context_count = int(t[5])

    def summarize(self):
        print("Dosa Evaluation Stats:")
        row = " {:<18} for [ relation={:>12s}]= {:0.3f}"
        title = "Average Precision (AP)"

        parent = self.parent / self.count
        sibling = self.sibling / self.count
        overall = (parent + sibling) / 2
        print(row.format(title, "overall", overall))
        print(row.format(title, "parent", parent))
        print(row.format(title, "sibling", sibling))

        context_parent = (
            self.context_parent / self.context_count
            if self.context_count
            else 1.0
        )
        context_sibling = (
            self.context_sibling / self.context_count
            if self.context_count
            else 1.0
        )
        context_overall = (context_parent + context_sibling) / 2
        print(row.format(title, "context overall", context_overall))
        print(row.format(title, "context parent", context_parent))
        print(row.format(title, "context sibling", context_sibling))

        non_context_parent = (self.parent - self.context_parent) / (
            self.count - self.context_count
        )
        non_context_sibling = (self.sibling - self.context_sibling) / (
            self.count - self.context_count
        )
        non_context_overall = (non_context_parent + non_context_sibling) / 2
        print(row.format(title, "non context overall", non_context_overall))
        print(row.format(title, "non context parent", non_context_parent))
        print(row.format(title, "non context sibling", non_context_sibling))
