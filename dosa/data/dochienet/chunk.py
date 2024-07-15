import json
from tqdm import tqdm
from typing import Optional

from dosa.core.node import Node, Tree
from dosa.core.context import build_context


def create_tree(doc: dict):
    """
    Create tree nodes from a doc annotation with a dummy root node with id 0
    """
    nodes = {0: Node(name=0)}
    for page in doc["pages"]:
        for obj in page["objects"]:
            nodes[obj["id"]] = Node(
                name=obj["id"],
                page_id=obj["page_id"],
                bbox=obj["bbox"],
                category_id=obj["category_id"],
                content=obj["content"],
            )
    return Tree(nodes)


def update_tree(tree: Tree, chunk: dict):
    """
    Update nodes relation with a chunk of annotations, the chunk is before normalization
    """
    nodes = tree.nodes
    for page in chunk["pages"]:
        for obj in page["objects"]:
            child = nodes[obj["id"]]
            child.order = obj["order"]
            # in case annotation self pointing, point it to 0 i.e. the dummy root
            pid = 0 if obj["id"] == obj["parent_id"] else obj["parent_id"]
            parent = nodes[pid]
            child.parent = parent
            if child not in parent.children:
                parent.children.append(child)

    for node in nodes.values():
        node.children.sort(key=lambda c: (c.page_id, c.order))


def normalize(chunk: dict):
    """
    Given a chunk of document, convert all labels from global id to continuous
    in chunk local id starting from 0 and return the mapping from local id to
    global id for converting back, this is used in training and evaluation only
    """
    # in case adding context makes the sibling id change, update the sibling
    objs = [obj for page in chunk["pages"] for obj in page["objects"]]
    objs.sort(key=lambda x: (x["page_id"], x["order"]))
    for i, obj in enumerate(objs):
        obj["sibling_id"] = (
            0 if i == 0 or obj["order"] == 0 else objs[i - 1]["id"]
        )

    # construct doc global object id to in chunk local object id mapping
    gid2cid = {obj["id"]: i for i, obj in enumerate(objs)}
    cid2gid = {v: k for k, v in gid2cid.items()}

    # do all id conversion
    for obj in objs:
        # save the global id for merge back the splits
        obj["g_id"] = obj["id"]
        obj["g_parent_id"] = obj["parent_id"]
        obj["g_sibling_id"] = obj["sibling_id"]

        # create parent id, if it's top level object, self pointing to itself
        obj["id"] = gid2cid[obj["id"]]
        if obj["parent_id"] == 0:
            obj["parent_id"] = obj["id"]
        elif obj["parent_id"] not in gid2cid:
            print(
                f"object {obj['g_id']} parent {obj['g_parent_id']} not in context of chunk {chunk['id']}"
            )
            obj["parent_id"] = obj["id"]
        else:
            obj["parent_id"] = gid2cid[obj["parent_id"]]

        # create sibling id, if it's first object, self pointing to itself
        obj["sibling_id"] = (
            obj["id"] if obj["sibling_id"] == 0 else gid2cid[obj["sibling_id"]]
        )

    return cid2gid


def split(doc: dict, capacity: int, window: int):
    """
    Split one single document into chunks for training and validation
    """
    # build id to obj mapping for the whole doc
    tree = create_tree(doc)
    chunks = []

    pages = []
    obj_count = 0
    for page in doc["pages"]:
        # if new page + current count > capacity, create a new chunk and build
        # context from this new chunk
        if obj_count + len(page["objects"]) > capacity:
            chunk = {
                "id": doc["id"],
                "split": len(chunks),
                "count": obj_count,
                "pages": pages,
                "language": doc["language"],
            }
            chunks.append(chunk)
            # update tree nodes relation incrementally using annotation
            update_tree(tree, chunk)
            pages, obj_count = build_context(tree, window)

            # append context pages with file path info
            for context_page in pages:
                context_page["file_path"] = (
                    f"{doc['id']}/page{context_page['page_id']}.png"
                )
                del context_page["page_id"]

        for obj in page["objects"]:
            # mark this as non context object
            obj["context"] = 0
        obj_count += len(page["objects"])
        pages.append(page)

    # leftover pages
    if pages:
        chunks.append(
            {
                "id": doc["id"],
                "split": len(chunks),
                "count": obj_count,
                "pages": pages,
                "language": doc["language"],
            }
        )

    # Normalize ids after all chunks in one doc are all processed
    for chunk in chunks:
        assert (
            chunk["count"] <= capacity
        ), f"chunk {chunk['id']} has {chunk['count']} chunks"
        normalize(chunk)

    return chunks


def build_chunks(
    docs: list,
    out_path: str,
    limit: int,
    window: int,
    skips: Optional[list[str]],
):
    """
    Create a list of chunks for training, each chunk fits into the sequence
    limit, each sequence contains id starting from 0
    """
    chunks = []
    for doc in tqdm(docs, "Chunking Dosa annotations"):
        if skips and doc["id"] in skips:
            continue
        try:
            chunks.extend(split(doc, limit, window))
        except Exception as e:
            print(f"exception {e} in {doc['id']}")

    with open(out_path, "w") as ofd:
        json.dump(chunks, ofd, indent=2)
