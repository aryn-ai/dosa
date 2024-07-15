from functools import cmp_to_key
import argparse
import json
import multiprocessing as mp
from pathlib import Path

from tqdm import tqdm

from args import get_args_parser
from dosa.core.context import build_context
from dosa.core.node import Node, Tree
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


def update_tree(
    tree: Tree,
    parents: dict[int, tuple[int, float]],
    siblings: dict[int, tuple[int, float]],
):
    """
    Update nodes relation based on inferred parents and siblings relations.
    """
    nodes = tree.nodes

    # 1. Update Parent Relations
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

    def prev(siblings: dict[int, tuple[int, float]], left: int, right: int):
        lt = left
        visited = []
        while lt != 0 and lt in siblings and lt not in visited:
            if siblings[lt][0] == right:
                return 1
            visited.append(lt)
            lt = siblings[lt][0]
        visited.clear()
        rt = right
        while rt != 0 and rt in siblings and rt not in visited:
            if siblings[rt][0] == left:
                return -1
            visited.append(rt)
            rt = siblings[rt][0]
        return 0

    def compare_nodes(left: Node, right: Node):
        if left.order == -1 and right.order != -1:
            return 1
        if left.order != -1 and right.order == -1:
            return -1
        if left.order != -1 and right.order != -1:
            return left.order - right.order

        if left.page_id != right.page_id:
            return left.page_id - right.page_id

        res = prev(siblings, left.name, right.name)
        if res != 0:
            return res

        if left.bbox[0] != right.bbox[0]:
            return left.bbox[0] - right.bbox[0]
        return left.bbox[1] - right.bbox[1]

    # 2. Fix Cycles (same as DocHieNet)
    def fix_branch(curr: Node):
        stack = []
        while curr is not None:
            if curr in stack:
                # Cycle detected
                idx = stack.index(curr)
                cycle = stack[idx:]
                bad = min(cycle, key=lambda t: (-t.page_id, parents[t.name][1]))
                bad.parent.children.remove(bad)
                bad.parent = tree.root
                tree.root.children.append(bad)
                return
            else:
                stack.append(curr)
                curr = curr.parent

    # We only need to check nodes involved in this chunk
    visited = []
    for cid in parents.keys():
        if cid not in visited:
            fix_branch(nodes[cid])
            visited.append(cid)

    # sort children for each node
    for node in nodes.values():
        node.children.sort(key=cmp_to_key(compare_nodes))

    order_id = 0

    def dfs(curr: Node):
        nonlocal order_id
        curr.order = order_id
        order_id += 1
        for child in curr.children:
            dfs(child)

    dfs(tree.root)


def collect_metrics(task: tuple[str, Tree, Tree]) -> tuple[float, float]:
    doc, predicted, gt = task
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
        return teds, positive_objects / (len(gt.nodes) - 1)
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

        # Extract Siblings
        # sibling prediction: label is index of previous sibling
        # label=0 usually implies no previous sibling (first child)
        siblings = {
            cid2gid[idx]: (cid2gid[label] if idx != label else 0, score)
            for idx, (label, score) in enumerate(
                zip(
                    result["sibling"]["labels"].cpu().tolist(),
                    result["sibling"]["scores"].cpu().tolist(),
                )
            )
        }

        update_tree(predicted, parents, siblings)

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
        for teds, f1 in results:
            self._teds += teds
            self._f1 += f1
            self._total_docs += 1

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
                        # It's better to log the exception rather than just raising generic
                        print(
                            f"Got exception in {doc.get('id', 'unknown')}, {e}"
                        )
                        raise e
                    buffer.append((doc["id"], predicted, gt))
                    if len(buffer) == self._num_workers:
                        results = pool.map(collect_metrics, buffer)
                        self.sum_results(results)
                        buffer.clear()

                if buffer:
                    results = pool.map(collect_metrics, buffer)
                    self.sum_results(results)

        print(
            f"TEDS score: {self._teds / self._total_docs if self._total_docs > 0 else 0}"
        )
        print(
            f"F1 score: {self._f1 / self._total_docs if self._total_docs > 0 else 0}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DocHieNet Benchmark", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    benchmarker = DocHieNetBenchmarker(args)
    benchmarker.evaluate("benchmark.json")
