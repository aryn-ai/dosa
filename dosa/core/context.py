from itertools import groupby

from dosa.core.node import Tree


def build_context(tree: Tree, window: int):
    """
    Build context based on current state of a semantic tree
    :param tree: a tree with all nodes, but only partial relation get defined
    :param window: window size, be at most 0.25 of the sequence length
    :return: a list of pages, each page is a dictionary
    """
    context = {}

    # collect all nodes which already resolve relations, this depends on the fact the current tree is updated
    # root dummy node always has a None parent.
    nodes = [node for node in tree.nodes.values() if node.parent is not None]
    nodes.sort(key=lambda node: (node.page_id, node.order))

    for node in reversed(nodes):
        # skip all the non-hierarchy nodes to find the last branch of the tree, and potentially some
        # previous branches
        if node.non_hierarchy():
            continue
        while node.parent is not None and node.name not in context:
            context[node.name] = node
            node = node.parent
        if len(context) > window:
            break

    assert 0 not in context
    count = len(context)
    orders = list(context.values())
    orders.sort(key=lambda x: (x.page_id, x.order))

    pages = []
    for page_id, page_nodes in groupby(orders, key=lambda x: x.page_id):
        objs = []
        for node in list(page_nodes):
            objs.append(
                {
                    "id": node.name,
                    "parent_id": (
                        node.parent.name if node.parent is not None else 0
                    ),
                    "sibling_id": objs[-1]["id"] if objs else 0,
                    "category_id": node.category,
                    "bbox": node.bbox,
                    "content": node.content,
                    "page_id": node.page_id,
                    "context": 1,
                    "order": node.order,
                }
            )
        pages.append({"page_id": page_id, "objects": objs})

    return pages, count
