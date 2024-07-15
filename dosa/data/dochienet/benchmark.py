import argparse
import json
import multiprocessing as mp
from pathlib import Path

from tqdm import tqdm

from args import get_args_parser
from dosa.core.context import build_context
from dosa.core.node import Tree, Node
from dosa.data.dataset import generate_sample
from dosa.data.dochienet.chunk import (
    create_tree as create_predicted_tree,
    normalize,
)
from infer import init_model


def create_gt_tree(doc: dict):
    """build the whole ground truth tree from doc annotation"""
    nodes = {0: Node(name=0)}
    for page in doc["pages"]:
        for obj in page["objects"]:
            nodes[obj["id"]] = Node(
                name=obj["id"],
                page_id=obj["page_id"],
                bbox=obj["bbox"],
                order=obj["order"],
                category_id=obj["category_id"],
                content=obj["content"],
            )
    for page in doc["pages"]:
        for obj in page["objects"]:
            child = nodes[obj["id"]]
            parent = nodes[obj["parent_id"]]
            child.parent = parent
            if child not in parent.children:
                parent.children.append(child)

    for node in nodes.values():
        node.children.sort(key=lambda c: (c.page_id, c.order))
    return Tree(nodes)


def update_tree(tree: Tree, chunk: dict, parents: dict[int, tuple[int, float]]):
    """
    Update nodes relation based on chunk annotation's ordering information and
    inferred parents relations
    """
    nodes = tree.nodes

    # update the order
    for page in chunk["pages"]:
        for obj in page["objects"]:
            nodes[obj["g_id"]].order = obj["order"]

    # update parent and children
    for cid, (pid, _) in parents.items():
        # in case some annotation self pointing
        pid = 0 if cid == pid else pid
        child = nodes[cid]
        if child.parent is not None:
            continue
        parent = nodes[pid]
        child.parent = parent
        if child not in parent.children:
            parent.children.append(child)

    def fix_branch(node: Node):
        stack = []
        while node is not None:
            if node in stack:
                idx = stack.index(node)
                cycle = stack[idx:]
                bad = min(cycle, key=lambda t: (-t.page_id, t.order))
                bad.parent.children.remove(bad)
                bad.parent = tree.root
                tree.root.children.append(node)
                return
            else:
                stack.append(node)
                node = node.parent

    visited = []
    for cid, _ in parents.items():
        if cid not in visited:
            fix_branch(nodes[cid])

    for node in nodes.values():
        node.children.sort(key=lambda c: (c.page_id, c.order))


def collect_metrics(
    task: tuple[str, Tree, Tree, bool]
) -> tuple[float, float, float, float, int]:
    doc, predicted, gt, long = task
    try:
        positive_objects = 0
        for oid, gt_node in gt.nodes.items():
            if oid == 0:
                continue
            p_node = predicted.nodes[oid]
            if gt_node.parent.name == p_node.parent.name:
                positive_objects += 1

        if predicted.cycled():
            raise Exception("predict has cycle, this is wrong")
        if gt.cycled():
            raise Exception("GT has cycle, this is wrong")

        teds = Node.tree_edit_distance(predicted.root, gt.root)
        if long:
            return (
                teds,
                positive_objects / (len(gt.nodes) - 1),
                teds,
                positive_objects / (len(gt.nodes) - 1),
                1,
            )
        else:
            return teds, positive_objects / (len(gt.nodes) - 1), 0, 0, 0
    except Exception as e:
        raise Exception(f"Got Exception while evaluating {doc}", e)


class DocHieNetBenchmarker:
    def __init__(self, args):
        self._capacity = args.n_sequence
        self._context_window = args.context_window
        preprocessor, model, postprocessor = init_model(args)
        self._preprocessor = preprocessor
        self._model = model
        self._postprocessor = postprocessor
        self._root = Path(args.data_path)
        self._device = args.device
        self._num_workers = mp.cpu_count()

        self._teds = 0
        self._f1 = 0
        self._total_docs = 0

        self._long_positive_objects = 0
        self._long_teds = 0
        self._long_total_docs = 0

    def process_chunk(self, predicted: Tree, chunk: dict):
        assert (
            chunk["count"] <= self._capacity
        ), f"Too many {chunk['count']} objects in a chunk"

        # build a map from chunk local id to doc global id
        cid2gid = normalize(chunk)

        sample = generate_sample(
            chunk, self._preprocessor, self._root / "images"
        )
        sample.to(self._device)
        results = self._postprocessor(self._model(sample))
        result = results[0]

        # Extract Parents
        parents = {
            cid2gid[idx]: (cid2gid[label] if idx != label else 0, score)
            for idx, (label, score) in enumerate(
                zip(
                    result["parent"]["labels"].cpu().tolist(),
                    result["parent"]["scores"].cpu().tolist(),
                )
            )
        }

        update_tree(predicted, chunk, parents)

        pages, obj_count = build_context(predicted, self._context_window)
        # append context pages with file path info
        for p in pages:
            p["file_path"] = f"{chunk['id']}/page{p['page_id']}.png"

        return pages, obj_count

    def process_doc(self, doc: dict):
        gt = create_gt_tree(doc)
        predicted = create_predicted_tree(doc)

        pages = []
        obj_count = 0
        for page in doc["pages"]:
            if obj_count + len(page["objects"]) > self._capacity:
                chunk = {
                    "id": doc["id"],
                    "count": obj_count,
                    "pages": pages,
                }
                pages, obj_count = self.process_chunk(predicted, chunk)

            obj_count += len(page["objects"])
            pages.append(page)

        # leftover pages
        if pages:
            chunk = {
                "id": doc["id"],
                "count": obj_count,
                "pages": pages,
            }
            self.process_chunk(predicted, chunk)

        return predicted, gt

    def sum_results(self, results):
        for teds, f1, long_teds, long_f1, long in results:
            self._teds += teds
            self._f1 += f1
            self._total_docs += 1
            if long == 1:
                self._long_positive_objects += long_f1
                self._long_teds += long_teds
                self._long_total_docs += 1

    def evaluate(self, name: str):
        ann_file = self._root / "annotations" / name

        buffer = []

        with mp.Pool(self._num_workers) as pool:
            with open(ann_file) as ifd:
                docs = json.load(ifd)
                for doc in tqdm(docs, desc="Infer against DOSA"):
                    try:
                        predicted, gt = self.process_doc(doc)
                    except Exception as e:
                        print(
                            f"Got exception in {doc.get('id', 'unknown')}, {e}"
                        )
                        raise e
                    buffer.append(
                        (doc["id"], predicted, gt, doc["count"] > 256)
                    )
                    if len(buffer) == self._num_workers:
                        results = pool.map(collect_metrics, buffer)
                        self.sum_results(results)
                        buffer.clear()

                if buffer:
                    results = pool.map(collect_metrics, buffer)
                    self.sum_results(results)

        print(
            f"F1 score: {self._f1 / self._total_docs if self._total_docs > 0 else 0}"
        )
        print(
            f"TEDS score: {self._teds / self._total_docs if self._total_docs > 0 else 0}"
        )
        print(
            f"Long F1 score: {self._long_positive_objects / self._long_total_docs if self._long_total_docs > 0 else 0}"
        )
        print(
            f"Long TEDS score: {self._long_teds / self._long_total_docs if self._long_total_docs > 0 else 0}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DocHieNet Benchmark", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    evaluator = DocHieNetBenchmarker(args)
    evaluator.evaluate("benchmark.json")
