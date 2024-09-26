from collections import defaultdict, OrderedDict
from functools import cmp_to_key
from itertools import groupby

import torch

from infer import init_model, infer
from processing.categories import Category
from processing.object_extractor import Page


class Analyzer:
    def __init__(self, args):
        self._capacity = args.n_sequence
        preprocessor, model, postprocessor = init_model(args)
        self._preprocessor = preprocessor
        self._model = model
        self._postprocessor = postprocessor

    @staticmethod
    def _filter(pages):
        # We currently handle only text, formula, section header and title
        for page in pages:
            objects = []
            for page_object in page.objects:
                label = page_object["label"]
                if label == 3 or label == 8 or label == 10 or label == 11:
                    objects.append(page_object)

            page.objects = objects

    @staticmethod
    def build_batch_tree(pids, sids):
        """
        Build a tree based on current batch of parent and sibling relations.
        It's not that easy to build the tree even given the parent, sibling and
        continuation relation, one challenge is potential miss classification.
        It's possible siblings belong to different parents and children of same
        parent are not siblings, in theory, score should always be used to
        decide
        TODO, we should build a more sophisticated way for reconstruction
            considering the relation confidence score, additional heuristics etc
        """

        def find_parents():
            parents = defaultdict(list)
            for idx, pid in pids.items():
                if idx == pid:
                    parents[0].append(idx)
                else:
                    parents[pid].append(idx)
            return parents

        def comparator(x1, x2):
            def predecessor(x1, x2):
                # check if x1 is predecessor of x2
                x = x2
                cycles = [x]
                while sids[x] != x:
                    x = sids[x]
                    if x == x1:
                        return True
                    if x in cycles:
                        return False
                    cycles.append(x)
                return False

            if predecessor(x1, x2):
                return -1
            elif predecessor(x2, x1):
                return 1
            else:
                return 0

        parents = find_parents()
        for k in parents.keys():
            siblings = sorted(parents[k], key=cmp_to_key(comparator))
            parents[k] = siblings

        return parents

    @staticmethod
    def merge_tree(tree, batch_tree):
        # merge the batch tree into
        for pid, children in batch_tree.items():
            global_children = tree[pid]["children"]
            for c in children:
                if c not in global_children:
                    global_children.append(c)

    def build_tree(self, tree, pids, sids):
        batch_tree = self.build_batch_tree(pids, sids)
        self.merge_tree(tree, batch_tree)

    @staticmethod
    def translate(result, id2id):
        def helper(ids):
            ids = ids.detach().cpu().numpy().tolist()
            gids = {}
            for idx, relation in enumerate(ids):
                gids[id2id[idx]] = id2id[relation]
            return gids

        result = result[0]
        result["parent"] = helper(result["parent"]["labels"])
        result["sibling"] = helper(result["sibling"]["labels"])
        result["continuation"] = helper(result["continuation"]["labels"])
        return result

    @staticmethod
    def prepare_sample(sample, pages):
        id2id = [obj["gid"] for obj in sample]
        inputs = []
        for page_no, page_objects in groupby(
            sorted(sample, key=lambda x: x["page_no"]),
            key=lambda x: x["page_no"],
        ):
            image = pages[page_no].image
            page_objects = list(page_objects)
            boxes = torch.tensor(
                [page_object["box"] for page_object in page_objects]
            )
            boxes[:, 2:] -= boxes[:, :2]
            labels = torch.tensor(
                [page_object["label"] for page_object in page_objects]
            )
            contents = [page_object["content"] for page_object in page_objects]
            fonts = torch.tensor(
                [page_object["font"] for page_object in page_objects]
            )
            inputs.append(
                {
                    "image": image,
                    "boxes": boxes,
                    "labels": labels,
                    "contents": contents,
                    "fonts": fonts,
                }
            )
        return id2id, inputs

    @staticmethod
    def build_context(tree):
        # TODO, add context for all potential categories appear so far
        context = []

        # find last branch of the tree
        cur = tree[0]
        while cur["children"]:
            last = cur["children"][-1]
            last = tree[last]
            context.append(last)
            cur = last

        return context

    @staticmethod
    def convert(tree):
        def recursive(pid):
            parent = tree[pid]
            # finally convert the dict tree to actual tree with sections
            if parent["children"] and pid not in parent["children"]:
                children = []
                for child in parent["children"]:
                    child = recursive(child)
                    children.append(child)
                parent["children"] = children

            if parent["gid"] == 0:
                return parent
            else:
                return OrderedDict(
                    [
                        ("gid", parent["gid"]),
                        ("label", Category(parent["label"]).name),
                        ("page_no", parent["page_no"]),
                        ("box", parent["box"]),
                        ("text", parent["text"]),
                        ("children", parent["children"]),
                    ]
                )

        return recursive(0)

    def analyze(self, pages: list[Page]):
        self._filter(pages)
        page_lengths = [len(page.objects) for page in pages]
        for page_no, page_length in enumerate(page_lengths):
            # it's better to check this before running the inference
            if page_length >= self._capacity - 8:
                raise Exception(
                    f"page {page_no} in document has {page_length} objects to handle"
                )

        # build index for each page object across all pages, add the root with 0
        tree = [{"gid": 0, "children": []}]
        gid = 1
        for page in pages:
            for page_object in page.objects:
                page_object["gid"] = gid
                page_object["children"] = []
                gid += 1
                tree.append(page_object)

        # split pages and objects based on model capacity
        sample = []
        for page_no, page_length in enumerate(page_lengths):
            if len(sample) + page_length > self._capacity:
                id2id, sample = self.prepare_sample(sample, pages)
                result = infer(
                    self._preprocessor,
                    self._model,
                    self._postprocessor,
                    [sample],
                )
                result = self.translate(result, id2id)
                self.build_tree(tree, result["parent"], result["sibling"])
                sample = self.build_context(tree)
            sample.extend(pages[page_no].objects)
        if sample:
            id2id, sample = self.prepare_sample(sample, pages)
            result = infer(
                self._preprocessor, self._model, self._postprocessor, [sample]
            )
            result = self.translate(result, id2id)
            self.build_tree(tree, result["parent"], result["sibling"])

        self.convert(tree)

        return tree
